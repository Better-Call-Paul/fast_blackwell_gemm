#include "common.h"
#include "profiler.h"

__global__ void basic_matmul(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < M)
    {
        float temp = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            temp += __bfloat162float(A[row * K + i]) * __bfloat162float(B[col * K + i]);
        }
        C[row * N + col] = __float2bfloat16(temp);
    }
}

void matmul_v0(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid(cdiv(N, 16), cdiv(M, 16));
    basic_matmul<<<grid, block>>>(A, B, C, M, N, K);
}
