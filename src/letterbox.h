// Letterbox (bilinear resize + padding with 114) per output pixel.
// Same coordinate convention as cv::resize(INTER_LINEAR): fx = (dx + 0.5) * sx - 0.5,
// with borders handled as in OpenCV. OpenCV uses fixed-point coefficients (11 bits),
// so the result may differ by ±1 gray level in some pixels.
// Compiles as __host__ __device__ (CUDA) or as plain C++ (CPU test).
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define LB_HD __host__ __device__ __forceinline__
#else
#define LB_HD inline
#endif

struct LetterboxGeom {
    int sw, sh, spitch;  // source (interleaved BGR, uint8), pitch in bytes
    int nw, nh;          // resized size (int(sw*r), int(sh*r))
    int dw, dh;          // output size (640x640)
    float scale_x, scale_y;  // sw/nw, sh/nh
};

LB_HD int lb_floor(float v) {
#ifdef __CUDA_ARCH__
    return __float2int_rd(v);
#else
    return int(std::floor(v));
#endif
}

// Returns the value (0..255, already rounded) of channel c at output pixel (dx, dy).
LB_HD float letterbox_px(const uint8_t* src, const LetterboxGeom& g, int dx, int dy, int c) {
    if (dx >= g.nw || dy >= g.nh) return 114.0f;
    float fx = (dx + 0.5f) * g.scale_x - 0.5f;
    int sx = lb_floor(fx);
    fx -= sx;
    if (sx < 0) { fx = 0.f; sx = 0; }
    if (sx >= g.sw - 1) { fx = 0.f; sx = g.sw - 1; }
    float fy = (dy + 0.5f) * g.scale_y - 0.5f;
    int sy = lb_floor(fy);
    fy -= sy;
    if (sy < 0) { fy = 0.f; sy = 0; }
    if (sy >= g.sh - 1) { fy = 0.f; sy = g.sh - 1; }
    int sx1 = sx + (fx > 0.f ? 1 : 0);
    int sy1 = sy + (fy > 0.f ? 1 : 0);
    const uint8_t* r0 = src + size_t(sy) * g.spitch;
    const uint8_t* r1 = src + size_t(sy1) * g.spitch;
    float v00 = r0[sx * 3 + c], v01 = r0[sx1 * 3 + c];
    float v10 = r1[sx * 3 + c], v11 = r1[sx1 * 3 + c];
    float v = (v00 * (1.f - fx) + v01 * fx) * (1.f - fy) + (v10 * (1.f - fx) + v11 * fx) * fy;
    v = lb_floor(v + 0.5f);
    return v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}

inline LetterboxGeom make_geom(int sw, int sh, int spitch, int dw = 640, int dh = 640) {
    LetterboxGeom g;
    g.sw = sw; g.sh = sh; g.spitch = spitch; g.dw = dw; g.dh = dh;
    // Python: int(img.shape[1] * r), with r as double
    double rd = double(dh) / sh < double(dw) / sw ? double(dh) / sh : double(dw) / sw;
    g.nw = int(sw * rd);
    g.nh = int(sh * rd);
    g.scale_x = float(double(sw) / g.nw);
    g.scale_y = float(double(sh) / g.nh);
    return g;
}

inline float letterbox_ratio(int sw, int sh, int dw = 640, int dh = 640) {
    double a = double(dh) / sh, b = double(dw) / sw;
    return float(a < b ? a : b);
}
