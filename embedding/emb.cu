#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ void pool_l2(
    const float* data,   // [num_vecs, size]
    int size,
    float* output        // [num_vecs]
) {

    extern __shared__ float sdata[]; 
    int vec = blockIdx.x;
    int tid = threadIdx.x;


    float sum = 0.0f;

    for (int i = tid; i < size; i += blockDim.x) {
        float val = data[vec * size + i];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) { // this is some witch craft !!!
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        

            
        __syncthreads();
    }

    if (tid < 32) {
        sum = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0)
        output[vec] = sum;
}

void pool_l2_cpu(const float* data, int num_vecs, int size, float* out) {
    for (int v = 0; v < num_vecs; v++) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float x = data[v * size + i];
            sum += x * x;
        }
        out[v] = sum;
    }
}

int main() {
    int num_vecs = 128;
    int size = 1024;
    int block = 256;

    size_t bytes = num_vecs * size * sizeof(float);

    std::vector<float> h_data(num_vecs * size);
    std::vector<float> h_out(num_vecs), h_ref(num_vecs);

    for (auto& x : h_data) x = rand() / float(RAND_MAX);

    float *d_data, *d_out;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, num_vecs * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    // warmup
    pool_l2<<<num_vecs, block, block * sizeof(float)>>>(d_data, size, d_out);
    cudaDeviceSynchronize();

    // benchmark
    int iters = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        pool_l2<<<num_vecs, block, block * sizeof(float)>>>(d_data, size, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("pool_l2: %.3f us avg over %d iters\n", ms * 1000.0f / iters, iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // correctness check
    cudaMemcpy(h_out.data(), d_out, num_vecs * sizeof(float),
               cudaMemcpyDeviceToHost);

    pool_l2_cpu(h_data.data(), num_vecs, size, h_ref.data());

    for (int i = 0; i < num_vecs; i++) {
        if (fabs(h_out[i] - h_ref[i]) > 1e-3) {
            printf("Mismatch at %d: %f vs %f\n", i, h_out[i], h_ref[i]);
            break;
        }
    }

    cudaFree(d_data);
    cudaFree(d_out);
}


