#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// GPU: crop + bilinear resize + ImageNet normalize on RGBA image.
// Output is a CHW float buffer (3 x dst_h x dst_w).
void preprocess_image_rgba_cuda(
    const uint8_t* d_rgba,     // GPU RGBA source image
    int src_h, int src_w,      // source resolution
    size_t src_pitch,          // source row stride in bytes
    float* d_out,              // GPU output CHW float buffer
    int dst_h, int dst_w,      // output resolution (448x448)
    float crop_ratio,          // bottom crop fraction (4.8/16.0)
    cudaStream_t stream);
