#include "common.h"
#include "profiler.h"

template<int BN, int BM, int BK>
__global__ void basic_tiled_matmul(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
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

        a_mem[a_local_row * BK + a_local_col] = __bfloat162float(A[a_global_row * K + a_global_col]);
        b_mem[b_local_row * BN + b_local_col] = __bfloat162float(B[b_global_col * K + b_global_row]);

        __syncthreads();

        for (int j = 0; j < BK; ++j)
        {
            accum += a_mem[threadIdx.y * BK + j] * b_mem[j * BN + threadIdx.x];
        }

        __syncthreads();
    }
    C[(blockIdx.y * BM + threadIdx.y) * N + (blockIdx.x * BN + threadIdx.x)] = __float2bfloat16(accum);
}

void matmul_v1(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    constexpr int BM = 16, BN = 16, BK = 16;
    dim3 block(BN, BM);
    dim3 grid(cdiv(N, BN), cdiv(M, BM));
    basic_tiled_matmul<BN, BM, BK><<<grid, block>>>(A, B, C, M, N, K);
}
