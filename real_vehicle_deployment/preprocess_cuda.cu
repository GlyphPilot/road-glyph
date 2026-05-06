#include "preprocess_cuda.cuh"

// ImageNet normalization constants
__constant__ float c_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float c_std[3]  = {0.229f, 0.224f, 0.225f};

__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    int src_h, int src_w, size_t src_pitch,
    float* __restrict__ dst,
    int dst_h, int dst_w,
    int crop_h,
    float scale_y, float scale_x)
{
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow >= dst_w || oh >= dst_h) return;

    const float iy = (oh + 0.5f) * scale_y - 0.5f;
    const int y0 = max(static_cast<int>(iy), 0);
    const int y1 = min(y0 + 1, src_h - 1);
    const float dy = iy - static_cast<float>(y0);

    const float ix = (ow + 0.5f) * scale_x - 0.5f;
    const int x0 = max(static_cast<int>(ix), 0);
    const int x1 = min(x0 + 1, src_w - 1);
    const float dx = ix - static_cast<float>(x0);

    const uint8_t* row0 = src + y0 * src_pitch;
    const uint8_t* row1 = src + y1 * src_pitch;

    const float w00 = (1.0f - dy) * (1.0f - dx);
    const float w01 = (1.0f - dy) * dx;
    const float w10 = dy * (1.0f - dx);
    const float w11 = dy * dx;

    const int plane = dst_h * dst_w;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = w00 * row0[x0 * 4 + c]
                + w01 * row0[x1 * 4 + c]
                + w10 * row1[x0 * 4 + c]
                + w11 * row1[x1 * 4 + c];
        dst[c * plane + oh * dst_w + ow] = (v / 255.0f - c_mean[c]) / c_std[c];
    }
}

void preprocess_image_rgba_cuda(
    const uint8_t* d_rgba,
    int src_h, int src_w,
    size_t src_pitch,
    float* d_out,
    int dst_h, int dst_w,
    float crop_ratio,
    cudaStream_t stream)
{
    const int crop_h = static_cast<int>(src_h - src_h * crop_ratio);
    const float scale_y = static_cast<float>(crop_h) / dst_h;
    const float scale_x = static_cast<float>(src_w) / dst_w;

    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_rgba, src_h, src_w, src_pitch,
        d_out, dst_h, dst_w,
        crop_h, scale_y, scale_x);
}
