#include <cuda_fp16.h>
#include "preproc.h"

template <typename T>
__device__ __forceinline__ T to_out(float v);
template <>
__device__ __forceinline__ float to_out<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half to_out<__half>(float v) { return __float2half_rn(v); }

template <typename T>
__global__ void letterbox_kernel(const uint8_t* __restrict__ src, LetterboxGeom g,
                                 T* __restrict__ dst) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= g.dw || dy >= g.dh) return;
    size_t plane = size_t(g.dw) * g.dh;
    size_t o = size_t(dy) * g.dw + dx;
#pragma unroll
    for (int c = 0; c < 3; ++c) dst[c * plane + o] = to_out<T>(letterbox_px(src, g, dx, dy, c));
}

void letterbox_gpu(const uint8_t* d_src, const LetterboxGeom& g, void* d_dst, bool half_out,
                   cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((g.dw + block.x - 1) / block.x, (g.dh + block.y - 1) / block.y);
    if (half_out)
        letterbox_kernel<__half><<<grid, block, 0, stream>>>(d_src, g, static_cast<__half*>(d_dst));
    else
        letterbox_kernel<float><<<grid, block, 0, stream>>>(d_src, g, static_cast<float*>(d_dst));
}
