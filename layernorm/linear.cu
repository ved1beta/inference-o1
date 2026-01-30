#include <cuda.h>

__global__ void RMSnorm(
    const float* weights,   // [size]
    int size,               // hidden dimension
    const float* input,     // [num_vecs, size]
    float* output            // [num_vecs, size]
) {
extern __shared__ float sdata[];

int tid = threadIdx.x;
int vec = blockIdx.x;
float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float x = input[vec * size + i];
    sum += x * x;
}

sdata[tid] = sum;
__syncthreads();

for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }


    float total_sum = sdata[tid];
    if (tid < 32) {
        // add the other half first
        total_sum += sdata[tid + 32];

        // warp shuffle reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
        }
    }


    __shared__ float inv_rms;
    if (tid == 0) {
        float mean_sq = total_sum / size;
        inv_rms = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();


    for (int i = tid; i < size; i += blockDim.x) {
        float x = input[vec * size + i];
        output[vec * size + i] = x * inv_rms * weights[i];
}
}
