// Fake implementation of the CUDA runtime and TensorRT on the CPU. Inference: sleeps MOCK_INF_MS
// (default 8) and writes a fixed output with a few boxes. "GPU" pre-processing: letterbox_px on the CPU.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
#include "preproc.h"
static int64_t mono() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec; }
static bool half_io() { const char* e = getenv("MOCK_HALF"); return e && *e == '1'; }
extern "C" {
cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
cudaError_t cudaFree(void* p) { free(p); return cudaSuccess; }
cudaError_t cudaMalloc(void** p, size_t n) { *p = calloc(1, n); return cudaSuccess; }
cudaError_t cudaMallocHost(void** p, size_t n) { *p = calloc(1, n); return cudaSuccess; }
cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = reinterpret_cast<cudaStream_t>(new int(0)); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) { memcpy(d, s, n); return cudaSuccess; }
cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = reinterpret_cast<cudaEvent_t>(new int64_t(0)); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { *reinterpret_cast<int64_t*>(e) = mono(); return cudaSuccess; }
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  *ms = float((*reinterpret_cast<int64_t*>(b) - *reinterpret_cast<int64_t*>(a)) / 1e6); return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "mock"; }
}
void letterbox_gpu(const uint8_t* src, const LetterboxGeom& g, void* dst, bool half_out, cudaStream_t) {
  size_t plane = size_t(g.dw) * g.dh;
  for (int y = 0; y < g.dh; ++y) for (int x = 0; x < g.dw; ++x) for (int c = 0; c < 3; ++c) {
    float v = letterbox_px(src, g, x, y, c);
    if (half_out) static_cast<__half*>(dst)[c * plane + size_t(y) * g.dw + x] = __float2half(v);
    else static_cast<float*>(dst)[c * plane + size_t(y) * g.dw + x] = v;
  }
}
namespace nvinfer1 {
IRuntime* createInferRuntime(ILogger&) { return new IRuntime(); }
#ifndef MOCK_TRT_LEGACY
TensorIOMode ICudaEngine::getTensorIOMode(const char* n) const { return std::string(n) == "images" ? TensorIOMode::kINPUT : TensorIOMode::kOUTPUT; }
#endif
Dims ICudaEngine::getTensorShape(const char* n) const {
  Dims d{}; if (std::string(n) == "images") { d.nbDims = 4; d.d[0] = 1; d.d[1] = 3; d.d[2] = 640; d.d[3] = 640; }
  else { d.nbDims = 3; d.d[0] = 1; d.d[1] = 8400; d.d[2] = 85; } return d; }
DataType ICudaEngine::getTensorDataType(const char*) const { return half_io() ? DataType::kHALF : DataType::kFLOAT; }
#ifdef MOCK_TRT_LEGACY
bool IExecutionContext::enqueueV2(void* const* b, cudaStream_t, void*) { in = b[0]; out = b[1]; return run_mock(); }
#else
bool IExecutionContext::setTensorAddress(const char* n, void* p) { (std::string(n) == "images" ? in : out) = p; return true; }
bool IExecutionContext::enqueueV3(cudaStream_t) { return run_mock(); }
#endif
bool IExecutionContext::run_mock() {
  static std::vector<float> o;
  if (o.empty()) {
    std::mt19937 r(3); std::uniform_real_distribution<float> u(0, 1);
    o.assign(8400 * 85, 0.f);
    for (int i = 0; i < 8400; ++i) { float* p = &o[i * 85]; p[0] = u(r) - .5f; p[1] = u(r) - .5f; p[2] = u(r); p[3] = u(r);
      p[4] = u(r) < 0.01f ? 0.9f : 0.01f; p[5 + (i % 7)] = 0.8f; }
  }
  const char* e = getenv("MOCK_INF_MS");
  std::this_thread::sleep_for(std::chrono::microseconds(int((e ? atof(e) : 8.0) * 1000)));
  if (half_io()) { auto* h = static_cast<__half*>(out); for (size_t i = 0; i < o.size(); ++i) h[i] = __float2half(o[i]); }
  else memcpy(out, o.data(), o.size() * 4);
  return true;
}
}  // namespace nvinfer1
