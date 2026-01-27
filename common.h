#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

constexpr int WARP_SIZE = 32;

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Basic CUDA error checking
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            printf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                   cudaGetErrorString(err_)); \
        } \
    } while (0)





