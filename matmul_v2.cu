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
    for (int iter = 0; iter < iterations; iter++)
    {
        // tma load
        if (warp_id == 0 && elect_sync())
        {
            for (int k = 0; k < BK / SWIZZLE_WIDTH; ++k)
            {
                int offset_k = iter * BK + k * SWIZZLE_WIDTH;
                tma_2d_gmem2smem(a_smem + k * BM * SWIZZLE_WIDTH * sizeof(__nv_bfloat16), &a_tensor_map, offset_k, offset_m, mbarrier_address);
                tma_2d_gmem2smem(b_smem + k * BN * SWIZZLE_WIDTH * sizeof(__nv_bfloat16), &b_tensor_map, offset_k, offset_n, mbarrier_address);
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
            {
                // first iteration
                tcgen05_mma_bf16(tensor_memory_address, make_smem_desc(a_smem), make_smem_desc(b_smem), i_desc, iter);
                for (int k2 = 1; k2 < SWIZZLE_WIDTH / 16; ++k2)
                {
                    const uint64_t a_descr = make_smem_desc(a_smem + k2 * 16 * sizeof(__nv_bfloat16));
                    const uint64_t b_descr = make_smem_desc(b_smem + k2 * 16 * sizeof(__nv_bfloat16));
                    tcgen05_mma_bf16(tensor_memory_address, a_descr, b_descr, i_desc, 1);
                }
            }

            for (int k1 = 1; k1 < BK / SWIZZLE_WIDTH; ++k1)
            {
                for (int k2 = 0; k2 < SWIZZLE_WIDTH / 16; ++k2)
                {
                    const uint64_t a_descr = make_smem_desc(a_smem + (k1 * BM * SWIZZLE_WIDTH * sizeof(__nv_bfloat16)) + (k2 * 16 * sizeof(__nv_bfloat16)));
                    const uint64_t b_descr = make_smem_desc(b_smem + (k1 * BN * SWIZZLE_WIDTH * sizeof(__nv_bfloat16)) + (k2 * 16 * sizeof(__nv_bfloat16)));
                    tcgen05_mma_bf16(tensor_memory_address, a_descr, b_descr, i_desc, 1);
                }
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mbarrier_address) : "memory");
        }
        
        mbarrier_wait(mbarrier_address, phase);
        phase ^= 1;
    }

    asm volatile("tcgen05.fence::after_thread_sync;");

    // load from tmem into smem
    for (int n = 0; n < BN / 8; ++n)
    {
        float tmp[8];
        const int addr = tensor_memory_address + ((warp_id * 32) << 16) + (n * 8);

        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" 
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), 
        "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr)
        );
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        __nv_bfloat162 output[4];

        for (int i = 0; i < 4; ++i)
        {
            output[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        }

        __nv_bfloat16 *c_output_ptr = C + (offset_m + thread_id) * N + (offset_n + n * 8);
        reinterpret_cast<int4 *>(c_output_ptr)[0] = reinterpret_cast<int4 *>(output)[0];
    }

    // de-alloc tmem
    __syncthreads();
    if (warp_id == 0)
    {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tensor_memory_address), "r"(BN));
    }
}

template<int BM, int BN, int BK>
void matmul_v2_launch(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K)
{
    CUtensorMap a_tensor_map, b_tensor_map;

    // make maps on host
    init_tmap_2d_simple(&a_tensor_map, A, M, K, BM, SWIZZLE_WIDTH, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);
    init_tmap_2d_simple(&b_tensor_map, B, N, K, BN, SWIZZLE_WIDTH, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

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
