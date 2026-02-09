// Swizzle

template<int BM, int BN, int BK>
__global__ __launch_bounds__(THREAD_BLOCK_SIZE)
void swizzled_matmul(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K, const __grid_constant__ CUtensorMap a_tensor_map, const __grid_constant__ CUtensorMap b_tensor_map)
{
    
}