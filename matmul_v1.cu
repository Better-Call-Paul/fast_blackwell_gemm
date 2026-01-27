#include "common.h"
#include "profiler.h"

template<int BN, int BM, int BK>
__global__ void basic_tiled_matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float a_mem[BM * BK];
    __shared__ float b_mem[BK * BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int a_local_row = tid / BK;
    int a_local_col = tid % BK;

    int b_local_row = tid / BN;
    int b_local_col = tid % BN;

    float accum = 0.0f;

    for (int i = 0; i < K; i += BK)
    {
        int a_global_row = blockIdx.y * BM + a_local_row;
        int a_global_col = a_local_col + i;

        int b_global_row = b_local_row + i;
        int b_global_col = blockIdx.x * BN + b_local_col;

        a_mem[a_local_row * BK + a_local_col] = A[a_global_row * K + a_global_col];
        b_mem[b_local_row * BN + b_local_col] = B[b_global_row * N + b_global_col];

        __syncthreads();

        for (int j = 0; j < BK; ++j)
        {
            accum += a_mem[threadIdx.y * BK + j] * b_mem[j * BN + threadIdx.x];
        }

        __syncthreads();
    }
    C[(blockIdx.y * BM + threadIdx.y) * N + (blockIdx.x * BN + threadIdx.x)] = accum;
}

void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int BM = 16, BN = 16, BK = 16;
    dim3 block(BN, BM);
    dim3 grid(cdiv(N, BN), cdiv(M, BM));
    basic_tiled_matmul<BN, BM, BK><<<grid, block>>>(A, B, C, M, N, K);
}

template<int BN, int BM, int BK>
__global__ void basic_tiled_matmul_profiled(const float *A, const float *B, float *C, int M, int N, int K,
                                             int64_t *profiler_data, int num_entries)
{
    __shared__ float a_mem[BM * BK];
    __shared__ float b_mem[BK * BN];

    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    Profiler profiler;
    if (tid == 0)
    {
        profiler.init(num_entries, profiler_data, block_id);
        profiler.start(ProfilerTag::Setup);
    }

    int a_local_row = tid / BK;
    int a_local_col = tid % BK;

    int b_local_row = tid / BN;
    int b_local_col = tid % BN;

    float accum = 0.0f;

    if (tid == 0) profiler.stop();

    for (int i = 0; i < K; i += BK)
    {
        if (tid == 0) profiler.start(ProfilerTag::IssueTMA);

        int a_global_row = blockIdx.y * BM + a_local_row;
        int a_global_col = a_local_col + i;

        int b_global_row = b_local_row + i;
        int b_global_col = blockIdx.x * BN + b_local_col;

        a_mem[a_local_row * BK + a_local_col] = A[a_global_row * K + a_global_col];
        b_mem[b_local_row * BN + b_local_col] = B[b_global_row * N + b_global_col];

        if (tid == 0) profiler.stop();

        __syncthreads();

        if (tid == 0) profiler.start(ProfilerTag::IssueMMA);

        for (int j = 0; j < BK; ++j)
        {
            accum += a_mem[threadIdx.y * BK + j] * b_mem[j * BN + threadIdx.x];
        }

        if (tid == 0) profiler.stop();

        __syncthreads();
    }

    if (tid == 0) profiler.start(ProfilerTag::Epilogue);

    C[(blockIdx.y * BM + threadIdx.y) * N + (blockIdx.x * BN + threadIdx.x)] = accum;

    if (tid == 0)
    {
        profiler.stop();
        profiler.flush();
    }
}

void profile_matmul_v1(const float *A, const float *B, float *C, int M, int N, int K,
                       int64_t *profiler_data, int num_entries)
{
    constexpr int BM = 16, BN = 16, BK = 16;
    dim3 block(BN, BM);
    dim3 grid(cdiv(N, BN), cdiv(M, BM));
    basic_tiled_matmul_profiled<BN, BM, BK><<<grid, block>>>(A, B, C, M, N, K, profiler_data, num_entries);
}