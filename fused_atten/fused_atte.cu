#define HEAD_SIZE 64
#define SEQ_LEN 512
#define BLOCK_SIZE 256

__global__ void fused_attention_v1(
    const float* __restrict__ query,    // [num_heads, head_size]
    const float* __restrict__ key,      // [num_heads, seq_len, head_size]
    const float* __restrict__ value,    // [num_heads, seq_len, head_size]
    float* __restrict__ output          // [num_heads, head_size]
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float s_query[HEAD_SIZE];
    __shared__ float s_scores[SEQ_LEN];
    

    if (tid < HEAD_SIZE) {
        s_query[tid] = query[head_idx * HEAD_SIZE + tid];
    }
    __syncthreads();

    for (int s = tid; s < SEQ_LEN; s += BLOCK_SIZE) {
        float score = 0.0f;
        int key_offset = head_idx * SEQ_LEN * HEAD_SIZE + s * HEAD_SIZE;
        
        for (int d = 0; d < HEAD_SIZE; d++) {
            score += s_query[d] * key[key_offset + d];
        }
        
        s_scores[s] = score * rsqrtf((float)HEAD_SIZE);
    }
    __syncthreads();
    

    __shared__ float s_max;
    

    if (tid == 0) {
        float max_val = -INFINITY;
        for (int i = 0; i < SEQ_LEN; i++) {
            max_val = fmaxf(max_val, s_scores[i]);
        }
        s_max = max_val;
    }
    __syncthreads();
    

    __shared__ float s_sum;
    

    for (int s = tid; s < SEQ_LEN; s += BLOCK_SIZE) {
        s_scores[s] = expf(s_scores[s] - s_max);
    }
    __syncthreads();
    

    if (tid == 0) {
        float sum_val = 0.0f;
        for (int i = 0; i < SEQ_LEN; i++) {
            sum_val += s_scores[i];
        }
        s_sum = sum_val;
    }
    __syncthreads();

    for (int s = tid; s < SEQ_LEN; s += BLOCK_SIZE) {
        s_scores[s] = s_scores[s] / s_sum;
    }
    __syncthreads();

    for (int d = tid; d < HEAD_SIZE; d += BLOCK_SIZE) {
        float out_val = 0.0f;
        
        for (int s = 0; s < SEQ_LEN; s++) {
            int val_idx = head_idx * SEQ_LEN * HEAD_SIZE + s * HEAD_SIZE + d;
            out_val += s_scores[s] * value[val_idx];
        }
        
        output[head_idx * HEAD_SIZE + d] = out_val;
    }
}