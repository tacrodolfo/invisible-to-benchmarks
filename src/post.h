// YOLOX post-processing (grid decoding + per-class NMS), equivalent to
// postprocess() in bench.py. Header-only so it can be tested without a GPU.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace yolox {

constexpr int IN_W = 640, IN_H = 640;
constexpr float SCORE_THR = 0.3f;
constexpr float NMS_THR = 0.45f;

struct Grid {
    std::vector<float> gx, gy, stride;
    Grid() {
        for (int s : {8, 16, 32}) {
            int h = IN_H / s, w = IN_W / s;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    gx.push_back(float(x));
                    gy.push_back(float(y));
                    stride.push_back(float(s));
                }
        }
    }
    size_t size() const { return gx.size(); }
};

// out: [n_anchors x n_ch] (n_ch = 5 + n_classes), float32.
// Returns the number of detections after NMS (same as n_det in bench.py).
// scratch avoids per-frame allocations.
struct PostScratch {
    std::vector<float> boxes;   // 4 per candidate, already offset per class
    std::vector<float> scores;
    std::vector<int> order;
    std::vector<char> removed;
};

inline int postprocess(const float* out, int n_anchors, int n_ch, float ratio, const Grid& g,
                       PostScratch& s) {
    const int n_cls = n_ch - 5;
    s.boxes.clear();
    s.scores.clear();
    for (int i = 0; i < n_anchors; ++i) {
        const float* p = out + size_t(i) * n_ch;
        // argmax over classes (first maximum, as numpy)
        int best = 0;
        float bv = p[5];
        for (int c = 1; c < n_cls; ++c)
            if (p[5 + c] > bv) { bv = p[5 + c]; best = c; }
        float score = p[4] * bv;
        if (!(score > SCORE_THR)) continue;
        float st = g.stride[i];
        float cx = (p[0] + g.gx[i]) * st;
        float cy = (p[1] + g.gy[i]) * st;
        float w = std::exp(p[2]) * st;
        float h = std::exp(p[3]) * st;
        float off = float(best) * 4096.0f;
        float x1 = (cx - w / 2) / ratio, y1 = (cy - h / 2) / ratio;
        float x2 = (cx + w / 2) / ratio, y2 = (cy + h / 2) / ratio;
        s.boxes.insert(s.boxes.end(), {x1 + off, y1 + off, x2 + off, y2 + off});
        s.scores.push_back(score);
    }
    const int n = int(s.scores.size());
    if (n == 0) return 0;
    s.order.resize(n);
    std::iota(s.order.begin(), s.order.end(), 0);
    std::sort(s.order.begin(), s.order.end(),
              [&](int a, int b) { return s.scores[a] > s.scores[b]; });
    s.removed.assign(n, 0);
    int keep = 0;
    for (int a = 0; a < n; ++a) {
        int i = s.order[a];
        if (s.removed[i]) continue;
        ++keep;
        const float* bi = &s.boxes[4 * i];
        float ai = (bi[2] - bi[0]) * (bi[3] - bi[1]);
        for (int b = a + 1; b < n; ++b) {
            int j = s.order[b];
            if (s.removed[j]) continue;
            const float* bj = &s.boxes[4 * j];
            float xx1 = std::max(bi[0], bj[0]), yy1 = std::max(bi[1], bj[1]);
            float xx2 = std::min(bi[2], bj[2]), yy2 = std::min(bi[3], bj[3]);
            float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
            float aj = (bj[2] - bj[0]) * (bj[3] - bj[1]);
            float ovr = inter / (ai + aj - inter + 1e-9f);
            if (ovr > NMS_THR) s.removed[j] = 1;
        }
    }
    return keep;
}

}  // namespace yolox
