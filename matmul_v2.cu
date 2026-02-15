#include "common.h"

#include <cuda_bf16.h>

constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int BLOCK_M = 128;
constexpr int MMA_K = 16;

template <int BLOCK_N, int BLOCK_K, bool TMAP_3D>
__global__
__launch_bounds__(TB_SIZE)
void matmul_v2_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16 *C_ptr,
    int M, int N, int K
)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int grid_m = M / BLOCK_M;
    const int grid_n = N / BLOCK_N;
    const int bid_m = bid / grid_n;
    const int bid_n = bid % grid_n;

    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    extern __shared__ __align__(1024) char smem[];
    const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[1];
    __shared__ int tmem_addr[1];
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));

    if (warp_id == 0 && elect_sync())
    {
        mbarrier_init(mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1)
    {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N));
    }

    __syncthreads();
    const int taddr = tmem_addr[0];

    int phase = 0;

    constexpr uint32_t i_desc = (1U << 4U)
                              | (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BLOCK_N >> 3U << 17U)
                              | ((uint32_t)BLOCK_M >> 4U << 24U)
                              ;

    const int num_iters = K / BLOCK_K;
    for (int iter_k = 0; iter_k < num_iters; iter_k++)
    {
        if (warp_id == 0 && elect_sync())
        {
            if constexpr (TMAP_3D)
            {
                const int off_k = iter_k * BLOCK_K / SWIZZLE_WIDTH;
                tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k, mbar_addr);
                tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k, mbar_addr);
            }
            else
            {
                for (int k = 0; k < BLOCK_K / 64; k++)
                {
                    const int off_k = iter_k * BLOCK_K + k * 64;
                    tma_2d_gmem2smem(A_smem + k * BLOCK_M * 128, &A_tmap, off_k, off_m, mbar_addr);
                    tma_2d_gmem2smem(B_smem + k * BLOCK_N * 128, &B_tmap, off_k, off_n, mbar_addr);
                }
            }

            constexpr int cp_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                        :: "r"(mbar_addr), "r"(cp_size) : "memory");
        }

        mbarrier_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;

        if (warp_id == 0 && elect_sync())
        {
            auto make_desc = [](int addr) -> uint64_t
            {
                const int SBO = 8 * 128;
                return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
            };

            {
                tcgen05_mma_bf16(taddr, make_desc(A_smem), make_desc(B_smem), i_desc, iter_k);
                for (int k2 = 1; k2 < 64 / MMA_K; k2++)
                {
                    uint64_t a_desc = make_desc(A_smem + k2 * 32);
                    uint64_t b_desc = make_desc(B_smem + k2 * 32);
                    tcgen05_mma_bf16(taddr, a_desc, b_desc, i_desc, 1);
                }
            }

            for (int k1 = 1; k1 < BLOCK_K / 64; k1++)
            {
                for (int k2 = 0; k2 < 64 / MMA_K; k2++)
                {
                    uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                    uint64_t b_desc = make_desc(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
                    tcgen05_mma_bf16(taddr, a_desc, b_desc, i_desc, 1);
                }
            }

            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mbar_addr) : "memory");
        }

        mbarrier_wait(mbar_addr, phase);
        phase ^= 1;
    }

    asm volatile("tcgen05.fence::after_thread_sync;");

    for (int n = 0; n < BLOCK_N / 8; n++)
    {
        float tmp[8];
        const int addr = taddr + ((warp_id * 32) << 16) + (n * 8);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                      "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                    : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        nv_bfloat162 out[4];
        for (int i = 0; i < 4; i++)
        {
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
        reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
    }

    __syncthreads();
    if (warp_id == 0)
    {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(BLOCK_N));
    }
}

template <int BLOCK_N, int BLOCK_K, bool TMAP_3D>
void matmul_v2_launch(
    const nv_bfloat16 *A_ptr,
    const nv_bfloat16 *B_ptr,
          nv_bfloat16 *C_ptr,
    int M, int N, int K
)
{
    CUtensorMap A_tmap, B_tmap;

    if constexpr (TMAP_3D)
    {
        auto init_tmap_AB = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr, uint64_t global_height, uint32_t shared_height)
        {
            constexpr uint32_t rank = 3;
            uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
            uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
            uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
            uint32_t elementStrides[rank]  = {1, 1, 1};

            auto err = cuTensorMapEncodeTiled(
                tmap,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                rank,
                (void *)ptr,
                globalDim,
                globalStrides,
                boxDim,
                elementStrides,
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
            check_cu(err);
        };
        init_tmap_AB(&A_tmap, A_ptr, M, BLOCK_M);
        init_tmap_AB(&B_tmap, B_ptr, N, BLOCK_N);
    }
    else
    {
        init_tmap_2d_simple(&A_tmap, A_ptr, M, K, BLOCK_M, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
        init_tmap_2d_simple(&B_tmap, B_ptr, N, K, BLOCK_N, 64, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    }

    int grid = (M / BLOCK_M) * (N / BLOCK_N);
    int size_AB = (BLOCK_M + BLOCK_N) * BLOCK_K;
    int smem_size = size_AB * sizeof(nv_bfloat16);

    auto this_kernel = matmul_v2_kernel<BLOCK_N, BLOCK_K, TMAP_3D>;
    if (smem_size > 48'000)
    {
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
}

void matmul_v2(
    const nv_bfloat16 *A_ptr,
    const nv_bfloat16 *B_ptr,
          nv_bfloat16 *C_ptr,
    int M, int N, int K
)
{
    matmul_v2_launch<256, 256, false>(A_ptr, B_ptr, C_ptr, M, N, K);
}

void matmul_v3(
    const nv_bfloat16 *A_ptr,
    const nv_bfloat16 *B_ptr,
          nv_bfloat16 *C_ptr,
    int M, int N, int K
)
{
    matmul_v2_launch<256, 256, true>(A_ptr, B_ptr, C_ptr, M, N, K);
}
