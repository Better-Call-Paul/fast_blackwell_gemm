#include <torch/library.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <cuda_bf16.h>

void matmul_v0(const float *A, const float *B, float *C, int M, int N, int K);
void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K);
void profile_matmul_v0(const float *A, const float *B, float *C, int M, int N, int K,
                       int64_t *profiler_data, int num_entries);
void profile_matmul_v1(const float *A, const float *B, float *C, int M, int N, int K,
                       int64_t *profiler_data, int num_entries);

template<int BM, int BN, int BK>
void matmul_v2(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);
template<int BM, int BN, int BK>
void matmul_v3(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);
template<int BM, int BN, int BK>
void matmul_v4(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);
template<int BM, int BN, int BK>
void matmul_v5(const __nv_bfloat16 *A, const __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K);

at::Tensor matmul_v0_wrapper(const at::Tensor& A, const at::Tensor& B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    matmul_v0(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

at::Tensor matmul_v1_wrapper(const at::Tensor& A, const at::Tensor& B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    matmul_v1(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

template<int BM, int BN, int BK>
at::Tensor matmul_v2_wrapper(const at::Tensor& A, const at::Tensor& B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    matmul_v2<BM, BN, BK>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K
    );

    return C;
}

at::Tensor matmul_v2_wrapper_default(const at::Tensor& A, const at::Tensor& B) {
    return matmul_v2_wrapper<256, 256, 128>(A, B);
}

at::Tensor profile_matmul_v0_wrapper(const at::Tensor& A, const at::Tensor& B,
                                      at::Tensor profiler, int64_t num_entries) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    profile_matmul_v0(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        static_cast<int>(num_entries)
    );

    return C;
}

at::Tensor profile_matmul_v1_wrapper(const at::Tensor& A, const at::Tensor& B,
                                      at::Tensor profiler, int64_t num_entries) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = at::empty({M, N}, A.options());

    profile_matmul_v1(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        profiler.data_ptr<int64_t>(),
        static_cast<int>(num_entries)
    );

    return C;
}

TORCH_LIBRARY(my_matmul, m) {

    m.def("matmul_v0(Tensor A, Tensor B) -> Tensor");
    m.impl("matmul_v0", &matmul_v0_wrapper);
    m.def("matmul_v1(Tensor A, Tensor B) -> Tensor");
    m.impl("matmul_v1", &matmul_v1_wrapper);
    m.def("profile_matmul_v0(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor");
    m.impl("profile_matmul_v0", &profile_matmul_v0_wrapper);
    m.def("profile_matmul_v1(Tensor A, Tensor B, Tensor(a!) profiler, int num_entries) -> Tensor");
    m.impl("profile_matmul_v1", &profile_matmul_v1_wrapper);

    m.def("matmul_v2(Tensor A, Tensor B) -> Tensor");
    m.impl("matmul_v2", &matmul_v2_wrapper_default);
}
