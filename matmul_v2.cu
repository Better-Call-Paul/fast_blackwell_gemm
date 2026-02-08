#include "common.h"

template<int BM, int BN, int BK>
__global__ __launch_bounds__(THREAD_BLOCK_SIZE) void basic_tcgen_matmul(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K, const __grid_constant__ CUtensorMap a_tensor_map, const __grid_constant__ CUtensorMap b_tensor_map)
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int grid_n = N / BN;
    int block_id_n = block_id % grid_n;
    int block_id_m = block_id / grid_n;

    int offset_n = BN * block_id_n;
    int offset_m = BM * block_id_m;

    // alloc tmem
    // alloc shmem for a and b 

    extern __shared__ __align__(128) char smem[];
    const int a_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int b_smem = a_smem + BM * BK * sizeof(__nv_bfloat16); // offset by a block (BM * BK)

    __shared__ uint64_t mbars[1];
    const int mbarrier_address = static_cast<int>(__cvta_generic_to_shared(mbars));
    __shared__ int tmem_addr[1];

    if (warp_id == 0 && elect_sync())
    {
        mbarrier_init(mbarrier_address, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1)
    {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BN));
    }

    __syncthreads();

    const int tensor_memory_address = tmem_addr[0];
    int phase = 0;

    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BN >> 3U << 17U)  // MMA_N
                            | ((uint32_t)BM >> 4U << 24U)  // MMA_M
                            ;

    const int iterations = K / BK;
    for (int i = 0; i < iterations; i++)
    {
        // tma load 
        if (warp_id == 0 && elect_sync())
        {
            for (int k = 0; k < BK / TMA_LOAD_WIDTH; ++k)
            {
                int offset_k = i * BK + k * TMA_LOAD_WIDTH;
                tma_2d_gmem2smem(a_smem + k * BM * TMA_LOAD_WIDTH * sizeof(__nv_bfloat16), a_tensor_map, offset_k, offset_m, mbarrier_address);
                tma_2d_gmem2smem(b_smem + k * BN * TMA_LOAD_WIDTH * sizeof(__nv_bfloat16), b_tensor_map, offset_k, offset_n, mbarrier_address);
            }

            constexpr int expected_bytes_to_be_recieved = (BM + BN) * BK * sizeof(__nv_bfloat16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbarrier_address), "r"(expected_bytes_to_be_recieved) : "memory");
        }

        // wait TMA
        mbarrier_wait(mbarrier_address, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");  // from DeepGEMM
        phase ^= 1;

        // mma
        if (warp_id == 0 && elect_sync())
        {
            for ()
            {
                tcgen05_mma_bf16(, 0);
            }


            for ()
            {
                tcgen05_mma_bf16(, 0);
            }
        }

        // load from tmem into registers

    }

    // each thread will de-alloc tmem
}

template<int BM, int BN, int BK>
void matmul_v2_launch(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    CUtensorMap a_tensor_map, b_tensor_map;

    // make maps on host
    init_tmap_2d_simple(&a_tensor_map, A, M, K, BM, TMA_LOAD_WIDTH, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_tmap_2d_simple(&b_tensor_map, B, N, K, BN, TMA_LOAD_WIDTH, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    int grid = (M * N) / (BM * BN);
    int TB_size = THREAD_BLOCK_SIZE;
    int smem_size = ((BM * BK) + (BK * BN)) * sizeof(__nv_bfloat16);

    auto base_kernel = basic_tcgen_matmul<BM, BN, BK>;

    if (smem_size > 48'000)
    {
        cudaFuncSetAttribute(base_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    base_kernel<<<grid, TB_size, smem_size>>>(A, B, C, M, N, K, a_tensor_map, b_tensor_map);
}

void matmul_v2(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    matmul_v2_launch<128, 256, 128>(A, B, C, M, N, K);
}
