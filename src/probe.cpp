// probe: samples GPU frequency and load, the frequency of each CPU cluster, input power and
// memory (EMC) frequency directly from sysfs/debugfs, with absolute clock_nanosleep.
// CLOCK_MONOTONIC clock, the same as the bench.
//
// CSV: t_ns,gpu_hz,gpu_load,cpu0_khz,<cpuN_khz per policy>,vin_mw,emc_hz
//   gpu_load in 0..1000 (as in sysfs); vin_mw and emc_hz = -1 in samples without that reading
//   (power every --power-every samples; EMC every --emc-every, because on Orin the read goes
//   through the BPMP and is expensive).
// Power: Orin (JetPack 6) = hwmon INA3221 "VDD_IN" (mV x mA);
//        Nano/Xavier (JetPack 4) = iio ina3221x, channel 0 (POM_5V_IN / VDD_IN) in mW.
// Usage: sudo ./probe --out file.csv [--interval-ms 1] [--power-every 10] [--emc-every 100] [--cpu N]
#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glob.h>
#include <string>
#include <vector>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

static std::vector<std::string> glob_all(const std::string& pat) {
    std::vector<std::string> out;
    glob_t g;
    if (glob(pat.c_str(), 0, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i) out.push_back(g.gl_pathv[i]);
    globfree(&g);
    std::sort(out.begin(), out.end());
    return out;
}

static bool exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

static std::string read_str(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (!f) return "";
    char b[256] = {0};
    if (!fgets(b, sizeof b, f)) b[0] = 0;
    fclose(f);
    std::string s(b);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

static long long rd(int fd) {
    if (fd < 0) return -1;
    char b[64];
    ssize_t n = pread(fd, b, sizeof b - 1, 0);
    if (n <= 0) return -1;
    b[n] = 0;
    return atoll(b);
}

int main(int argc, char** argv) {
    std::string out;
    double interval_ms = 1.0;
    int power_every = 10, emc_every = 100, cpu = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--out") out = argv[i + 1];
        else if (k == "--interval-ms") interval_ms = atof(argv[i + 1]);
        else if (k == "--power-every") power_every = atoi(argv[i + 1]);
        else if (k == "--cpu") cpu = atoi(argv[i + 1]);
        else if (k == "--emc-every") emc_every = atoi(argv[i + 1]);
    }
    if (out.empty()) {
        fprintf(stderr, "usage: probe --out file.csv [--interval-ms 1] [--power-every 10] [--cpu N]\n");
        return 1;
    }
    if (cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpu, &s);
        sched_setaffinity(0, sizeof s, &s);
    }

    auto gpus = glob_all("/sys/class/devfreq/*gpu*");
    std::string gdev = gpus.empty() ? "" : gpus[0];
    std::string gload;
    for (auto& c : std::vector<std::string>{gdev + "/device/load"})
        if (!gdev.empty() && exists(c)) gload = c;
    if (gload.empty())
        for (auto& pat : {"/sys/devices/platform/bus@0/*.gpu/load", "/sys/devices/platform/*gpu*/load",
                          "/sys/devices/gpu.0/load", "/sys/devices/*.gpu/load"}) {
            auto v = glob_all(pat);
            if (!v.empty()) { gload = v[0]; break; }
        }
    int fg = gdev.empty() ? -1 : open((gdev + "/cur_freq").c_str(), O_RDONLY);
    int fl = gload.empty() ? -1 : open(gload.c_str(), O_RDONLY);

    // one frequency per CPU policy (Orin Nano: policy0 = cpu0-3, policy4 = cpu4-5)
    std::vector<int> fc;
    std::vector<std::string> cnames;
    for (auto& p : glob_all("/sys/devices/system/cpu/cpufreq/policy*")) {
        fc.push_back(open((p + "/scaling_cur_freq").c_str(), O_RDONLY));
        cnames.push_back("cpu" + p.substr(p.rfind("policy") + 6) + "_khz");
    }
    if (fc.empty()) {
        fc.push_back(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", O_RDONLY));
        cnames.push_back("cpu0_khz");
    }

    // VDD_IN on the INA3221
    int fv = -1, fi = -1;
    for (auto& hw : glob_all("/sys/class/hwmon/hwmon*"))
        for (auto& lab : glob_all(hw + "/in*_label"))
            if (read_str(lab) == "VDD_IN") {
                std::string idx = lab.substr(lab.rfind("/in") + 3);
                idx = idx.substr(0, idx.find('_'));
                fv = open((hw + "/in" + idx + "_input").c_str(), O_RDONLY);
                fi = open((hw + "/curr" + idx + "_input").c_str(), O_RDONLY);
            }

    // JetPack 4: INA3221 iio, power already in mW (channel 0 = input)
    int fp = -1;
    if (fv < 0 || fi < 0) {
        auto v = glob_all("/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input");
        if (!v.empty()) fp = open(v[0].c_str(), O_RDONLY);
    }
    const bool has_pw = (fv >= 0 && fi >= 0) || fp >= 0;

    // EMC: Orin = BPMP (debugfs), Nano/Xavier = kernel clk (debugfs)
    int fe = -1;
    for (auto& c : {"/sys/kernel/debug/bpmp/debug/clk/emc/rate", "/sys/kernel/debug/clk/emc/clk_rate"})
        if (fe < 0 && exists(c)) fe = open(c, O_RDONLY);

    printf("probe: gpu=%s load=%s cpus=%zu power=%s emc=%s interval=%.2f ms\n", gdev.c_str(),
           fl >= 0 ? "ok" : "n/a", fc.size(), fp >= 0 ? "iio" : (has_pw ? "hwmon" : "n/a"),
           fe >= 0 ? "ok" : "n/a", interval_ms);
    fflush(stdout);

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    const size_t nc = fc.size();
    std::vector<long long> data;
    data.reserve(size_t(5 + nc) * 600000);
    const int64_t step = int64_t(interval_ms * 1e6);
    int64_t nxt = now_ns();
    long long i = 0;
    while (!g_stop) {
        data.push_back(now_ns());
        data.push_back(rd(fg));
        data.push_back(rd(fl));
        for (int fd : fc) data.push_back(rd(fd));
        long long mw = -1, emc = -1;
        if (power_every > 0 && i % power_every == 0) {
            if (fp >= 0) {
                mw = rd(fp);
            } else if (fv >= 0 && fi >= 0) {
                long long mv = rd(fv), ma = rd(fi);
                if (mv >= 0 && ma >= 0) mw = mv * ma / 1000;
            }
        }
        if (emc_every > 0 && i % emc_every == 0) emc = rd(fe);
        data.push_back(mw);
        data.push_back(emc);
        ++i;
        nxt += step;
        int64_t t = now_ns();
        if (nxt > t) {
            timespec ts{time_t(nxt / 1000000000LL), long(nxt % 1000000000LL)};
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
        } else {
            nxt = t;
        }
    }

    FILE* f = fopen(out.c_str(), "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "t_ns,gpu_hz,gpu_load");
    for (auto& c : cnames) fprintf(f, ",%s", c.c_str());
    fprintf(f, ",vin_mw,emc_hz\n");
    const size_t w = 3 + nc + 2;
    for (size_t r = 0; r + w <= data.size(); r += w) {
        for (size_t c = 0; c < w; ++c) fprintf(f, c ? ",%lld" : "%lld", data[r + c]);
        fputc('\n', f);
    }
    fclose(f);
    printf("probe: %lld samples in %s\n", i, out.c_str());
    return 0;
}
