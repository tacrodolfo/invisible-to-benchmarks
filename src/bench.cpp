// bench: YOLOX + TensorRT 10 latency harness in C++ (replaces bench.py and bench_pipe.py).
//
// Same interface and CSV as bench.py (extra columns are appended at the end, so the older
// analysis scripts keep working). Clock: CLOCK_MONOTONIC in ns, the same as Python's
// time.perf_counter_ns() and the probe.
//
// Columns: pass,frame,drop,t_rel,t_start,t_dec,t_pre,t_deq,t_h2d,t_inf,t_d2h,t_end,n_det,
//          g_dec,g_pre,g_h2d,g_inf,g_d2h
//   t_*  host timestamps (ns). t_deq = dequeue time (pipeline) or = t_pre (sequential).
//   g_*  duration of each stage on the GPU (ns), measured with cudaEvent (0 if the stage does not use the GPU).
//   With --sync edge, t_h2d and t_inf are the instants when enqueueing returned; use g_*
//   for the duration of the GPU stages.
//
// Main options (in addition to those of bench.py):
//   --pipeline seq|pipe        sequential (1 thread) or two stages with a 1-slot queue
//   --input jpeg|raw|tensor    JPEG in memory | already decoded frames (BGR) in memory |
//                              already pre-processed tensor (network only: H2D + inference + D2H)
//   --decode cpu|gpu           OpenCV (libjpeg-turbo) | nvJPEG on the GPU (requires pre gpu)
//   --pre cpu|gpu              letterbox in OpenCV | CUDA kernel
//   --post cpu|none            NMS on the CPU | no post-processing
//   --sync stage|edge          synchronize after each GPU operation | only once per frame
//   --sched auto|spin|yield|block   CUDA runtime wait policy
//   --wait sleep|hybrid        wait until release: absolute clock_nanosleep | sleep and
//                              busy-wait the last 1 ms (as bench.py)
#include <NvInfer.h>
// TensorRT < 8.5 (JetPack 4/5: Jetson Nano, Xavier): bindings API and enqueueV2
#define TRT_LEGACY (NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5))
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef WITH_NVJPEG
#include <nvjpeg.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "letterbox.h"
#include "post.h"
#include "preproc.h"

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
#include <experimental/filesystem>  // gcc 7 (Ubuntu 18.04, JetPack 4)
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// ---------------------------------------------------------------- utilities
static inline int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

static void ck_cuda(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s in %s (line %d)\n", cudaGetErrorString(e), what, line);
        std::exit(2);
    }
}
#define CK(x) ck_cuda((x), #x, __LINE__)

#ifdef WITH_NVJPEG
static void ck_nvj(nvjpegStatus_t s, const char* what, int line) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        fprintf(stderr, "nvJPEG error %d in %s (line %d)\n", int(s), what, line);
        std::exit(2);
    }
}
#define NVJ(x) ck_nvj((x), #x, __LINE__)
#endif

[[noreturn]] static void die(const std::string& m) {
    fprintf(stderr, "ERROR: %s\n", m.c_str());
    std::exit(1);
}

static void wait_until(int64_t t, bool hybrid) {
    if (!hybrid) {
        timespec ts{time_t(t / 1000000000LL), long(t % 1000000000LL)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
        return;
    }
    // as in bench.py: sleeps until ~1 ms before and finishes with a busy wait
    for (;;) {
        int64_t rem = t - now_ns();
        if (rem <= 0) return;
        if (rem > 2000000) {
            int64_t s = rem - 1000000;
            timespec ts{time_t(s / 1000000000LL), long(s % 1000000000LL)};
            nanosleep(&ts, nullptr);
        }
    }
}

// ---------------------------------------------------------------- arguments
struct Args {
    std::string engine, frames = "frames", mode = "periodic", tag, outdir = "results";
    double period_ms = 1000.0 / 12.0, deadline_ms = -1;
    int passes = 3, warmup = 100, limit = 0;
    std::string pipeline = "seq", input = "jpeg", decode = "cpu", pre = "cpu", post = "cpu";
    std::string sync = "stage", sched = "auto", wait = "sleep", nvjpeg_backend = "gpu_hybrid";
    int cv_threads = -1;
    std::map<std::string, std::string> all;
};

static void usage() {
    fprintf(stderr,
            "usage: bench --engine E.engine --tag T [--frames DIR] [--mode b2b|periodic]\n"
            "  [--period-ms P] [--deadline-ms D] [--passes N] [--warmup W] [--limit L] [--outdir O]\n"
            "  [--pipeline seq|pipe] [--input jpeg|raw|tensor] [--decode cpu|gpu] [--pre cpu|gpu]\n"
            "  [--post cpu|none] [--sync stage|edge] [--sched auto|spin|yield|block]\n"
            "  [--wait sleep|hybrid] [--nvjpeg-backend gpu_hybrid|hybrid|default] [--cv-threads N]\n");
    std::exit(1);
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i], v;
        if (k == "-h" || k == "--help") usage();
        if (k.rfind("--", 0) != 0) usage();
        auto eq = k.find('=');
        if (eq != std::string::npos) {
            v = k.substr(eq + 1);
            k = k.substr(0, eq);
        } else {
            if (i + 1 >= argc) usage();
            v = argv[++i];
        }
        k = k.substr(2);
        a.all[k] = v;
        if (k == "engine") a.engine = v;
        else if (k == "frames") a.frames = v;
        else if (k == "mode") a.mode = v;
        else if (k == "period-ms") a.period_ms = std::stod(v);
        else if (k == "deadline-ms") a.deadline_ms = std::stod(v);
        else if (k == "passes") a.passes = std::stoi(v);
        else if (k == "warmup") a.warmup = std::stoi(v);
        else if (k == "limit") a.limit = std::stoi(v);
        else if (k == "tag") a.tag = v;
        else if (k == "outdir") a.outdir = v;
        else if (k == "pipeline") a.pipeline = v;
        else if (k == "input") a.input = v;
        else if (k == "decode") a.decode = v;
        else if (k == "pre") a.pre = v;
        else if (k == "post") a.post = v;
        else if (k == "sync") a.sync = v;
        else if (k == "sched") a.sched = v;
        else if (k == "wait") a.wait = v;
        else if (k == "nvjpeg-backend") a.nvjpeg_backend = v;
        else if (k == "cv-threads") a.cv_threads = std::stoi(v);
        else die("unknown option --" + k);
    }
    auto in = [](const std::string& v, std::initializer_list<const char*> ok) {
        for (auto o : ok)
            if (v == o) return true;
        return false;
    };
    if (a.engine.empty() || a.tag.empty()) usage();
    if (!in(a.mode, {"b2b", "periodic"})) die("--mode");
    if (!in(a.pipeline, {"seq", "pipe"})) die("--pipeline");
    if (!in(a.input, {"jpeg", "raw", "tensor"})) die("--input");
    if (!in(a.decode, {"cpu", "gpu"})) die("--decode");
    if (!in(a.pre, {"cpu", "gpu"})) die("--pre");
    if (!in(a.post, {"cpu", "none"})) die("--post");
    if (!in(a.sync, {"stage", "edge"})) die("--sync");
    if (!in(a.sched, {"auto", "spin", "yield", "block"})) die("--sched");
    if (!in(a.wait, {"sleep", "hybrid"})) die("--wait");
    if (a.decode == "gpu" && a.input != "jpeg") die("--decode gpu only makes sense with --input jpeg");
    if (a.decode == "gpu" && a.pre != "gpu") die("--decode gpu requires --pre gpu (the image stays on the GPU)");
    if (a.input == "tensor" && a.pre != "cpu") die("--input tensor uses the ready tensor; leave --pre cpu");
#ifndef WITH_NVJPEG
    if (a.decode == "gpu") die("built without nvJPEG (see CMake)");
#endif
    return a;
}

// ---------------------------------------------------------------- TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};

struct Engine {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    std::string in_name, out_name;
    bool in_half = false, out_half = false;
    size_t in_elems = 0, out_elems = 0, in_bytes = 0, out_bytes = 0;
    int n_anchors = 0, n_ch = 0, in_idx = -1, out_idx = -1;
    void* bindings[2] = {nullptr, nullptr};  // only on TensorRT < 8.5 (enqueueV2)
    std::string in_shape, out_shape;

    static size_t esize(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            default: die("unsupported tensor type (only FP32/FP16 for I/O)");
        }
    }

    explicit Engine(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) die("could not open " + path);
        std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(blob.data(), blob.size()));  // 8.x e 10.x
        if (!engine) die("failed to deserialize the engine");
        ctx.reset(engine->createExecutionContext());
#if TRT_LEGACY
        const int nio = engine->getNbBindings();
#else
        const int nio = engine->getNbIOTensors();
#endif
        for (int i = 0; i < nio; ++i) {
#if TRT_LEGACY
            const char* n = engine->getBindingName(i);
            auto d = engine->getBindingDimensions(i);
            auto t = engine->getBindingDataType(i);
            const bool is_in = engine->bindingIsInput(i);
#else
            const char* n = engine->getIOTensorName(i);
            auto d = engine->getTensorShape(n);
            auto t = engine->getTensorDataType(n);
            const bool is_in = engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT;
#endif
            size_t e = 1;
            std::string sh = "(";
            for (int j = 0; j < d.nbDims; ++j) {
                e *= size_t(d.d[j]);
                sh += std::to_string(d.d[j]) + (j + 1 < d.nbDims ? "," : "");
            }
            sh += ")";
            if (is_in) {
                in_name = n; in_idx = i; in_elems = e; in_half = t == nvinfer1::DataType::kHALF;
                in_bytes = e * esize(t); in_shape = sh;
            } else {
                out_name = n; out_idx = i; out_elems = e; out_half = t == nvinfer1::DataType::kHALF;
                out_bytes = e * esize(t); out_shape = sh;
                n_ch = int(d.d[d.nbDims - 1]);
                n_anchors = int(e / size_t(n_ch));
            }
        }
        if (nio != 2 || in_idx < 0 || out_idx < 0) die("the engine must have exactly one input and one output");
        if (in_elems != size_t(3) * yolox::IN_W * yolox::IN_H) die("unexpected input " + in_shape);
        printf("engine: input %s %s %s, output %s %s %s (TensorRT %d.%d)\n", in_name.c_str(), in_shape.c_str(),
               in_half ? "fp16" : "fp32", out_name.c_str(), out_shape.c_str(), out_half ? "fp16" : "fp32",
               NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
    }

    // Fixed output address (once) and inference with the slot's input.
    void set_output(void* d_out) {
#if TRT_LEGACY
        bindings[out_idx] = d_out;
#else
        if (!ctx->setTensorAddress(out_name.c_str(), d_out)) die("setTensorAddress(out)");
#endif
    }
    void infer(void* d_in, cudaStream_t stream) {
#if TRT_LEGACY
        bindings[in_idx] = d_in;
        if (!ctx->enqueueV2(bindings, stream, nullptr)) die("enqueueV2 failed");
#else
        if (!ctx->setTensorAddress(in_name.c_str(), d_in)) die("setTensorAddress(in)");
        if (!ctx->enqueueV3(stream)) die("enqueueV3 failed");
#endif
    }
};

// ---------------------------------------------------------------- frames
struct Frames {
    std::vector<std::string> files;
    std::vector<std::vector<uint8_t>> jpeg;
    uint8_t* raw = nullptr;
    bool raw_pinned = false;
    size_t raw_each = 0;
    int w = 0, h = 0;
    size_t n() const { return files.size(); }
    const uint8_t* raw_ptr(size_t k) const { return raw + k * raw_each; }
    ~Frames() {
        if (raw) {
            if (raw_pinned) cudaFreeHost(raw);
            else free(raw);
        }
    }
};

static void load_frames(const Args& a, Frames& F) {
    for (auto& e : fs::directory_iterator(a.frames))
        if (e.path().extension() == ".jpg") F.files.push_back(e.path().string());
    std::sort(F.files.begin(), F.files.end());
    if (a.limit > 0 && size_t(a.limit) < F.files.size()) F.files.resize(a.limit);
    if (F.files.empty()) die("no .jpg in " + a.frames);
    for (auto& p : F.files) {
        std::ifstream f(p, std::ios::binary);
        F.jpeg.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    cv::Mat first = cv::imdecode(cv::Mat(1, int(F.jpeg[0].size()), CV_8U, F.jpeg[0].data()), cv::IMREAD_COLOR);
    F.w = first.cols;
    F.h = first.rows;
    if (a.input == "raw") {
        F.raw_each = size_t(F.w) * F.h * 3;
        size_t total = F.raw_each * F.n();
        if (total > (size_t(3) << 30))
            die("raw frames would take " + std::to_string(total >> 20) + " MB; use --limit (e.g. 400)");
        // pinned memory mimics a camera DMA buffer; if it fails, regular memory
        if (cudaMallocHost(reinterpret_cast<void**>(&F.raw), total) == cudaSuccess) {
            F.raw_pinned = true;
        } else {
            cudaGetLastError();
            F.raw = static_cast<uint8_t*>(malloc(total));
            if (!F.raw) die("out of memory for the raw frames");
        }
        for (size_t k = 0; k < F.n(); ++k) {
            cv::Mat img = cv::imdecode(cv::Mat(1, int(F.jpeg[k].size()), CV_8U, F.jpeg[k].data()), cv::IMREAD_COLOR);
            if (img.cols != F.w || img.rows != F.h || !img.isContinuous()) die("frames of different sizes");
            memcpy(F.raw + k * F.raw_each, img.data, F.raw_each);
        }
        printf("%zu raw frames %dx%d in memory (%zu MB, %s)\n", F.n(), F.w, F.h, total >> 20,
               F.raw_pinned ? "pinned" : "pageable");
    } else {
        printf("%zu frames loaded (%dx%d)\n", F.n(), F.w, F.h);
    }
}

// ---------------------------------------------------------------- output rows
struct Row {
    int pass = 0, frame = 0, drop = 0;
    int64_t t_rel = 0, t_start = 0, t_dec = 0, t_pre = 0, t_deq = 0, t_h2d = 0, t_inf = 0, t_d2h = 0, t_end = 0;
    int n_det = -1;
    int64_t g_dec = 0, g_pre = 0, g_h2d = 0, g_inf = 0, g_d2h = 0;
};

// ---------------------------------------------------------------- slots (up to 3 frames in flight)
struct Slot {
    void* h_in = nullptr;  // pre-processed tensor on the host (pinned), used with --pre cpu
    void* d_in = nullptr;  // input tensor on the GPU (address passed to TensorRT)
    cudaEvent_t e0{}, e1{}, e2{};  // GPU decode start, decode end/pre start, pre end
    bool gpu_dec = false, gpu_pre = false;
    float ratio = 1.f;
};

// ---------------------------------------------------------------- stage A: decode + pre-processing
struct PreCtx {
    const Args& a;
    const Frames& F;
    const Engine& E;
    cudaStream_t stream{};
    uint8_t* d_bgr = nullptr;  // BGR image on the GPU (GPU decode or upload for GPU pre-processing)
    cv::Mat resized, padded;
    LetterboxGeom geom;
    float ratio;
    bool sync_stage;
#ifdef WITH_NVJPEG
    nvjpegHandle_t nvj{};
    nvjpegJpegDecoder_t dec{};
    nvjpegJpegState_t dstate{};
    nvjpegBufferPinned_t pbuf{};
    nvjpegBufferDevice_t dbuf{};
    nvjpegJpegStream_t jstream{};
    nvjpegDecodeParams_t params{};
#endif

    PreCtx(const Args& a_, const Frames& F_, const Engine& E_) : a(a_), F(F_), E(E_) {
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        geom = make_geom(F.w, F.h, F.w * 3);
        ratio = letterbox_ratio(F.w, F.h);
        sync_stage = a.sync == "stage";
        padded = cv::Mat(yolox::IN_H, yolox::IN_W, CV_8UC3, cv::Scalar(114, 114, 114));
        if (a.pre == "gpu") CK(cudaMalloc(&d_bgr, size_t(F.w) * F.h * 3));
#ifdef WITH_NVJPEG
        if (a.decode == "gpu") {
            nvjpegBackend_t be = NVJPEG_BACKEND_GPU_HYBRID;
            if (a.nvjpeg_backend == "hybrid") be = NVJPEG_BACKEND_HYBRID;
            else if (a.nvjpeg_backend == "default") be = NVJPEG_BACKEND_DEFAULT;
            NVJ(nvjpegCreateSimple(&nvj));
            NVJ(nvjpegDecoderCreate(nvj, be, &dec));
            NVJ(nvjpegDecoderStateCreate(nvj, dec, &dstate));
            NVJ(nvjpegBufferPinnedCreate(nvj, nullptr, &pbuf));
            NVJ(nvjpegBufferDeviceCreate(nvj, nullptr, &dbuf));
            NVJ(nvjpegStateAttachPinnedBuffer(dstate, pbuf));
            NVJ(nvjpegStateAttachDeviceBuffer(dstate, dbuf));
            NVJ(nvjpegJpegStreamCreate(nvj, &jstream));
            NVJ(nvjpegDecodeParamsCreate(nvj, &params));
            NVJ(nvjpegDecodeParamsSetOutputFormat(params, NVJPEG_OUTPUT_BGRI));
        }
#endif
    }

    // CPU: letterbox + HWC->CHW straight into the slot's pinned buffer
    void pre_cpu(const cv::Mat& img, Slot& s) {
        if (resized.cols != geom.nw || resized.rows != geom.nh) resized.create(geom.nh, geom.nw, CV_8UC3);
        cv::resize(img, resized, cv::Size(geom.nw, geom.nh), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(padded(cv::Rect(0, 0, geom.nw, geom.nh)));
        const size_t plane = size_t(yolox::IN_W) * yolox::IN_H;
        const uint8_t* p = padded.data;
        if (!E.in_half) {
            float* d = static_cast<float*>(s.h_in);
            for (size_t i = 0; i < plane; ++i) {
                d[i] = p[3 * i];
                d[plane + i] = p[3 * i + 1];
                d[2 * plane + i] = p[3 * i + 2];
            }
        } else {
            __half* d = static_cast<__half*>(s.h_in);
            for (size_t i = 0; i < plane; ++i) {
                d[i] = __float2half(float(p[3 * i]));
                d[plane + i] = __float2half(float(p[3 * i + 1]));
                d[2 * plane + i] = __float2half(float(p[3 * i + 2]));
            }
        }
        s.ratio = ratio;
    }

    // Fills t_dec and t_pre of r and leaves frame k's tensor ready in slot s.
    void run(size_t k, Slot& s, Row& r) {
        s.gpu_dec = s.gpu_pre = false;
        if (a.input == "tensor") {  // network only: the fixed tensor is already in s.h_in
            r.t_dec = r.t_pre = now_ns();
            return;
        }
        cv::Mat img;
        if (a.input == "raw") {
            img = cv::Mat(F.h, F.w, CV_8UC3, const_cast<uint8_t*>(F.raw_ptr(k)));
            r.t_dec = now_ns();
        } else if (a.decode == "cpu") {
            img = cv::imdecode(cv::Mat(1, int(F.jpeg[k].size()), CV_8U, const_cast<uint8_t*>(F.jpeg[k].data())),
                               cv::IMREAD_COLOR);
            r.t_dec = now_ns();
        } else {
#ifdef WITH_NVJPEG
            const auto& j = F.jpeg[k];
            NVJ(nvjpegJpegStreamParse(nvj, j.data(), j.size(), 0, 0, jstream));
            NVJ(nvjpegDecodeJpegHost(nvj, dec, dstate, params, jstream));  // CPU phase
            CK(cudaEventRecord(s.e0, stream));  // g_dec = GPU part only
            NVJ(nvjpegDecodeJpegTransferToDevice(nvj, dec, dstate, jstream, stream));
            nvjpegImage_t out{};
            out.channel[0] = d_bgr;
            out.pitch[0] = size_t(F.w) * 3;
            NVJ(nvjpegDecodeJpegDevice(nvj, dec, dstate, &out, stream));
            CK(cudaEventRecord(s.e1, stream));
            s.gpu_dec = true;
            if (sync_stage) CK(cudaStreamSynchronize(stream));
            r.t_dec = now_ns();
#endif
        }
        if (a.pre == "cpu") {
            pre_cpu(img, s);
        } else {
            if (!s.gpu_dec) {  // image on the host: send it to the GPU (pinned raw = direct DMA copy)
                CK(cudaEventRecord(s.e1, stream));
                CK(cudaMemcpyAsync(d_bgr, img.data, size_t(F.w) * F.h * 3, cudaMemcpyHostToDevice, stream));
            }
            letterbox_gpu(d_bgr, geom, s.d_in, E.in_half, stream);
            CK(cudaGetLastError());
            CK(cudaEventRecord(s.e2, stream));
            s.gpu_pre = true;
            s.ratio = ratio;
            if (sync_stage) CK(cudaStreamSynchronize(stream));
        }
        r.t_pre = now_ns();
    }
};

// ---------------------------------------------------------------- stage B: H2D + network + D2H + NMS
struct InfCtx {
    const Args& a;
    Engine& E;
    cudaStream_t stream{};
    void* d_out = nullptr;
    void* h_out = nullptr;
    std::vector<float> out_f;
    cudaEvent_t h0{}, h1{}, i1{}, d1{};
    yolox::Grid grid;
    yolox::PostScratch scratch;
    bool sync_stage;

    InfCtx(const Args& a_, Engine& E_) : a(a_), E(E_) {
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CK(cudaMalloc(&d_out, E.out_bytes));
        CK(cudaMallocHost(&h_out, E.out_bytes));
        out_f.resize(E.out_elems);
        for (auto* e : {&h0, &h1, &i1, &d1}) CK(cudaEventCreate(e));
        E.set_output(d_out);
        if (size_t(E.n_anchors) != grid.size()) die("number of anchors does not match the 8/16/32 grid");
        sync_stage = a.sync == "stage";
    }

    static int64_t ev_ns(cudaEvent_t a, cudaEvent_t b) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        return int64_t(double(ms) * 1e6);
    }

    void run(Slot& s, Row& r) {
        CK(cudaEventRecord(h0, stream));
        if (s.gpu_pre) CK(cudaStreamWaitEvent(stream, s.e2, 0));
        else CK(cudaMemcpyAsync(s.d_in, s.h_in, E.in_bytes, cudaMemcpyHostToDevice, stream));
        CK(cudaEventRecord(h1, stream));
        if (sync_stage) CK(cudaStreamSynchronize(stream));
        r.t_h2d = now_ns();

        E.infer(s.d_in, stream);
        CK(cudaEventRecord(i1, stream));
        if (sync_stage) CK(cudaStreamSynchronize(stream));
        r.t_inf = now_ns();

        CK(cudaMemcpyAsync(h_out, d_out, E.out_bytes, cudaMemcpyDeviceToHost, stream));
        CK(cudaEventRecord(d1, stream));
        CK(cudaStreamSynchronize(stream));  // the only mandatory synchronization
        r.t_d2h = now_ns();

        if (a.post == "cpu") {
            const float* o = static_cast<const float*>(h_out);
            if (E.out_half) {
                const __half* hh = static_cast<const __half*>(h_out);
                for (size_t i = 0; i < E.out_elems; ++i) out_f[i] = __half2float(hh[i]);
                o = out_f.data();
            }
            r.n_det = yolox::postprocess(o, E.n_anchors, E.n_ch, s.ratio, grid, scratch);
        } else {
            r.n_det = 0;
        }
        r.t_end = now_ns();

        r.g_h2d = s.gpu_pre ? 0 : ev_ns(h0, h1);
        r.g_inf = ev_ns(h1, i1);
        r.g_d2h = ev_ns(i1, d1);
        r.g_dec = s.gpu_dec ? ev_ns(s.e0, s.e1) : 0;
        r.g_pre = s.gpu_pre ? ev_ns(s.e1, s.e2) : 0;
    }
};

// ---------------------------------------------------------------- 1-slot queue
struct Item {
    int kind = 0;  // 0 frame, 1 end of pass, 2 end
    Row row;
    int slot = 0;
};

class OneSlotQueue {
    std::mutex m;
    std::condition_variable cv_put, cv_get;
    std::deque<Item> q;

   public:
    void put(Item it) {
        std::unique_lock<std::mutex> l(m);
        cv_put.wait(l, [&] { return q.empty(); });
        q.push_back(std::move(it));
        cv_get.notify_one();
    }
    Item get() {
        std::unique_lock<std::mutex> l(m);
        cv_get.wait(l, [&] { return !q.empty(); });
        Item it = std::move(q.front());
        q.pop_front();
        cv_put.notify_one();
        return it;
    }
};

// ---------------------------------------------------------------- summary
static double pct(std::vector<double> v, double q) {  // linear interpolation, as numpy
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    double pos = q / 100.0 * (v.size() - 1);
    size_t lo = size_t(std::floor(pos)), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

static void json_escape(FILE* f, const std::string& s) {
    fputc('"', f);
    for (char c : s) {
        if (c == '"' || c == '\\') fputc('\\', f);
        fputc(c, f);
    }
    fputc('"', f);
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    unsigned flag = cudaDeviceScheduleAuto;
    if (a.sched == "spin") flag = cudaDeviceScheduleSpin;
    else if (a.sched == "yield") flag = cudaDeviceScheduleYield;
    else if (a.sched == "block") flag = cudaDeviceScheduleBlockingSync;
    CK(cudaSetDeviceFlags(flag));
    CK(cudaFree(nullptr));  // creates the context
    if (a.cv_threads >= 0) cv::setNumThreads(a.cv_threads);

    Frames F;
    load_frames(a, F);
    Engine E(a.engine);
    PreCtx pc(a, F, E);
    InfCtx ic(a, E);

    const int NS = 3;
    Slot slots[NS];
    for (auto& s : slots) {
        CK(cudaMallocHost(&s.h_in, E.in_bytes));
        CK(cudaMalloc(&s.d_in, E.in_bytes));
        for (auto* e : {&s.e0, &s.e1, &s.e2}) CK(cudaEventCreate(e));
    }
    if (a.input == "tensor") {  // fixed tensor of frame 0, like trtexec with a fixed input
        cv::Mat img = cv::imdecode(cv::Mat(1, int(F.jpeg[0].size()), CV_8U, F.jpeg[0].data()), cv::IMREAD_COLOR);
        for (auto& s : slots) pc.pre_cpu(img, s);
    }

    auto run_frame = [&](size_t k, Slot& s, Row& r) {
        r.t_start = now_ns();
        pc.run(k, s, r);
        r.t_deq = r.t_pre;
        ic.run(s, r);
    };

    const size_t N = F.n();
    for (int i = 0; i < a.warmup; ++i) {
        Row r;
        run_frame(size_t(i) % N, slots[0], r);
    }

    // start and end of the measurement on both clocks, to align tegrastats
    // (which logs local time) with the monotonic timestamps
    auto real_ns = [] {
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    };
    const int64_t meas_mono0 = now_ns(), meas_real0 = real_ns();
    const int64_t period = int64_t(std::llround(a.period_ms * 1e6));
    const bool periodic = a.mode == "periodic", hybrid = a.wait == "hybrid";
    std::vector<Row> rows;
    rows.reserve(size_t(a.passes) * N);

    if (a.pipeline == "seq") {
        for (int p = 0; p < a.passes; ++p) {
            int64_t base = now_ns() + 10000000;
            for (size_t k = 0; k < N; ++k) {
                Row r;
                r.pass = p;
                r.frame = int(k);
                if (periodic) {
                    r.t_rel = base + int64_t(k) * period;
                    int64_t t = now_ns();
                    if (t < r.t_rel) wait_until(r.t_rel, hybrid);
                    else if (t >= r.t_rel + period) {
                        r.drop = 1;
                        rows.push_back(r);
                        continue;
                    }
                }
                run_frame(k, slots[0], r);
                if (!periodic) r.t_rel = r.t_start;
                rows.push_back(r);
            }
            printf("pass %d/%d done\n", p + 1, a.passes);
            fflush(stdout);
        }
    } else {
        OneSlotQueue q;
        std::mutex lk;
        std::vector<Row> dropped;
        std::thread producer([&] {
            int used = 0;
            for (int p = 0; p < a.passes; ++p) {
                int64_t base = now_ns() + 20000000;
                for (size_t k = 0; k < N; ++k) {
                    Item it;
                    it.row.pass = p;
                    it.row.frame = int(k);
                    if (periodic) {
                        it.row.t_rel = base + int64_t(k) * period;
                        int64_t t = now_ns();
                        if (t < it.row.t_rel) wait_until(it.row.t_rel, hybrid);
                        else if (t >= it.row.t_rel + period) {
                            it.row.drop = 1;
                            std::lock_guard<std::mutex> g(lk);
                            dropped.push_back(it.row);
                            continue;
                        }
                    }
                    it.slot = used++ % NS;
                    it.row.t_start = now_ns();
                    if (!periodic) it.row.t_rel = it.row.t_start;
                    pc.run(k, slots[it.slot], it.row);
                    q.put(it);
                }
                Item e;
                e.kind = 1;
                e.row.pass = p;
                q.put(e);
            }
            Item d;
            d.kind = 2;
            q.put(d);
        });
        for (;;) {
            Item it = q.get();
            if (it.kind == 2) break;
            if (it.kind == 1) {
                printf("pass %d/%d done\n", it.row.pass + 1, a.passes);
                fflush(stdout);
                continue;
            }
            it.row.t_deq = now_ns();
            ic.run(slots[it.slot], it.row);
            rows.push_back(it.row);
        }
        producer.join();
        rows.insert(rows.end(), dropped.begin(), dropped.end());
        std::sort(rows.begin(), rows.end(),
                  [](const Row& x, const Row& y) { return x.pass != y.pass ? x.pass < y.pass : x.frame < y.frame; });
    }

    const int64_t meas_mono1 = now_ns(), meas_real1 = real_ns();

    // ------------------------------------------------ writing
    fs::create_directories(a.outdir);
    std::string path = a.outdir + "/" + a.tag + ".csv";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) die("could not open " + path);
    fprintf(f, "pass,frame,drop,t_rel,t_start,t_dec,t_pre,t_deq,t_h2d,t_inf,t_d2h,t_end,n_det,"
               "g_dec,g_pre,g_h2d,g_inf,g_d2h\n");
    for (auto& r : rows)
        fprintf(f, "%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%lld,%lld,%lld,%lld,%lld\n", r.pass,
                r.frame, r.drop, (long long)r.t_rel, (long long)r.t_start, (long long)r.t_dec, (long long)r.t_pre,
                (long long)r.t_deq, (long long)r.t_h2d, (long long)r.t_inf, (long long)r.t_d2h,
                (long long)r.t_end, r.n_det, (long long)r.g_dec, (long long)r.g_pre, (long long)r.g_h2d,
                (long long)r.g_inf, (long long)r.g_d2h);
    fclose(f);

    // ------------------------------------------------ summary (deadline = period, unless --deadline-ms)
    double D = a.deadline_ms > 0 ? a.deadline_ms : a.period_ms;
    std::vector<double> resp, ginf;
    size_t drops = 0, late = 0;
    int64_t t0 = INT64_MAX, t1 = 0;
    for (auto& r : rows) {
        t0 = std::min(t0, r.t_rel);
        if (r.drop) { ++drops; continue; }
        double x = (r.t_end - r.t_rel) / 1e6;
        resp.push_back(x);
        ginf.push_back(r.g_inf / 1e6);
        if (x > D) ++late;
        t1 = std::max(t1, r.t_end);
    }
    double dur = (t1 - t0) / 1e9;
    double mr = double(late + drops) / rows.size();
    double mean = resp.empty() ? NAN : std::accumulate(resp.begin(), resp.end(), 0.0) / resp.size();
    double mginf = ginf.empty() ? NAN : std::accumulate(ginf.begin(), ginf.end(), 0.0) / ginf.size();
    char summ[512];
    snprintf(summ, sizeof summ,
             "SUMMARY tag=%s n=%zu drops=%zu late=%zu mr=%.5f mean=%.3f p99=%.3f thr=%.3f good=%.3f "
             "g_inf=%.3f deadline=%.3f",
             a.tag.c_str(), rows.size(), drops, late, mr, mean, pct(resp, 99), resp.size() / dur,
             (resp.size() - late) / dur, mginf, D);
    printf("%s\nsaved to %s\n", summ, path.c_str());

    std::string mpath = a.outdir + "/" + a.tag + ".meta.json";
    FILE* m = fopen(mpath.c_str(), "w");
    if (m) {
        char host[256] = {0};
        gethostname(host, sizeof host - 1);
        fprintf(m, "{\n  \"harness\": \"cpp\",\n  \"host\": ");
        json_escape(m, host);
        fprintf(m, ",\n  \"n_frames\": %zu, \"width\": %d, \"height\": %d,\n", N, F.w, F.h);
        fprintf(m, "  \"in_half\": %s, \"out_half\": %s, \"raw_pinned\": %s, \"cv_threads\": %d,\n",
                E.in_half ? "true" : "false", E.out_half ? "true" : "false", F.raw_pinned ? "true" : "false",
                cv::getNumThreads());
        fprintf(m, "  \"meas_mono_ns\": [%lld, %lld], \"meas_real_ns\": [%lld, %lld],\n", (long long)meas_mono0,
                (long long)meas_mono1, (long long)meas_real0, (long long)meas_real1);
        fprintf(m, "  \"summary\": ");
        json_escape(m, summ);
        fprintf(m, ",\n  \"args\": {");
        bool first = true;
        // effective values, defaults included
        std::map<std::string, std::string> eff = {
            {"engine", a.engine}, {"frames", a.frames}, {"mode", a.mode}, {"period-ms", std::to_string(a.period_ms)},
            {"deadline-ms", std::to_string(D)}, {"passes", std::to_string(a.passes)},
            {"warmup", std::to_string(a.warmup)}, {"limit", std::to_string(a.limit)}, {"tag", a.tag},
            {"pipeline", a.pipeline}, {"input", a.input}, {"decode", a.decode}, {"pre", a.pre},
            {"post", a.post}, {"sync", a.sync}, {"sched", a.sched}, {"wait", a.wait},
            {"nvjpeg-backend", a.nvjpeg_backend}};
        for (auto& kv : eff) {
            fprintf(m, "%s\n    ", first ? "" : ",");
            json_escape(m, kv.first);
            fprintf(m, ": ");
            json_escape(m, kv.second);
            first = false;
        }
        fprintf(m, "\n  }\n}\n");
        fclose(m);
    }
    return 0;
}
