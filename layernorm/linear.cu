#include <cuda.h>
#include <cuda_runtime.h>

__global__ void RMSnorm(
    const float* __restrict__ weights, // [size]
    int size,
    const float* __restrict__ input,   // [num_vecs, size]
    float* __restrict__ output         // [num_vecs, size]
) {
    int tid = threadIdx.x;
    int vec = blockIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float x = input[vec * size + i];
        sum += x * x;
    }


    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }


    __shared__ float warp_sum[32]; // max 32 warps
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (lane == 0) {
        warp_sum[warp_id] = sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (warp_id == 0) {
        total_sum = (tid < (blockDim.x + 31) / 32)
                        ? warp_sum[lane]
                        : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
        }
    }


    __shared__ float inv_rms;
    if (tid == 0) {
        float mean_sq = total_sum * (1.0f / size);
        inv_rms = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < size; i += blockDim.x) {
        float x = input[vec * size + i];
        output[vec * size + i] = x * inv_rms * weights[i];
    }
}
