#include <torch/library.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>

typedef void MatmulFn(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K);

MatmulFn matmul_v0;
MatmulFn matmul_v1;
MatmulFn matmul_v2;

template <MatmulFn matmul_fn>
at::Tensor matmul(const at::Tensor& A, const at::Tensor& B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  auto C = at::empty({M, N}, A.options());
  matmul_fn(
    reinterpret_cast<nv_bfloat16 *>(A.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(B.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(C.data_ptr()),
    M, N, K
  );
  return C;
}

TORCH_LIBRARY(my_matmul, m) {
  m.def("matmul_v0(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_v0", &matmul<matmul_v0>);
  m.def("matmul_v1(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_v1", &matmul<matmul_v1>);
  m.def("matmul_v2(Tensor A, Tensor B) -> Tensor"); m.impl("matmul_v2", &matmul<matmul_v2>);
}
