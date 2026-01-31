#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void softmax(
    const float* __restrict__ scores, // [num_vecs, size]
    int size,
    float* __restrict__ output        // [num_vecs, size]
) {
    int tid = threadIdx.x;
    int vec_idx = blockIdx.x;
    
    __shared__ float s_data[BLOCK_SIZE];

    float local_max = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        float score = scores[vec_idx * size + i];
        local_max = fmaxf(local_max, score);
    }
    
    s_data[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    
    float global_max = s_data[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float score = scores[vec_idx * size + i];
        float exp_score = expf(score - global_max);
        output[vec_idx * size + i] = exp_score;
        local_sum += exp_score;
    }
    
    s_data[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    float global_sum = s_data[0];
    __syncthreads();

    for (int i = tid; i < size; i += blockDim.x) {
        output[vec_idx * size + i] /= global_sum;
    }
}