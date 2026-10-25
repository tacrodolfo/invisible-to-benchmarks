#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "letterbox.h"

// Letterbox on the GPU: interleaved BGR uint8 source (device memory) -> CHW tensor
// 3x640x640 in float32 (half_out = false) or float16 (half_out = true), same BGR order and
// same 0..255 scale as the CPU pre-processing.
void letterbox_gpu(const uint8_t* d_src, const LetterboxGeom& g, void* d_dst, bool half_out,
                   cudaStream_t stream);
