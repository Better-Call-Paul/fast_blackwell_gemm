#include "common.h"
#include "profiler.h"

__global__ void basic_matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < M)
    {
        float temp = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            temp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = temp;
    }
}

void matmul_v0(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid(cdiv(N, 16), cdiv(M, 16));
    basic_matmul<<<grid, block>>>(A, B, C, M, N, K);
}

__global__ void basic_matmul_profiled(const float *A, const float *B, float *C, int M, int N, int K,
                                       int64_t *profiler_data, int num_entries)
{
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    Profiler profiler;
    if (tid == 0)
    {
        profiler.init(num_entries, profiler_data, block_id);
        profiler.start(ProfilerTag::Setup);
    }

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid == 0)
    {
        profiler.stop();
        profiler.start(ProfilerTag::IssueMMA);
    }

    if (col < N && row < M)
    {
        float temp = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            temp += A[row * K + i] * B[i * N + col];
        }

        if (tid == 0)
        {
            profiler.stop();
            profiler.start(ProfilerTag::Epilogue);
        }

        C[row * N + col] = temp;
    }

    if (tid == 0)
    {
        profiler.stop();
        profiler.flush();
    }
}

void profile_matmul_v0(const float *A, const float *B, float *C, int M, int N, int K,
                       int64_t *profiler_data, int num_entries)
{
    dim3 block(16, 16);
    dim3 grid(cdiv(N, 16), cdiv(M, 16));
    basic_matmul_profiled<<<grid, block>>>(A, B, C, M, N, K, profiler_data, num_entries);
}


