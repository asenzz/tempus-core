//
// Created by zarko on 5/11/24.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include "matmul.cuh"
#include "common/cuda_util.cuh"
#include "common/constants.hpp"

namespace svr {
/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C)
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + common::C_cu_tile_width - 1) / common::C_cu_tile_width, (m + common::C_cu_tile_width - 1) / common::C_cu_tile_width);
        dim3 dimBlock(common::C_cu_tile_width, common::C_cu_tile_width);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void
gpu_matrix_mult(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, const unsigned m, const unsigned n, const unsigned k)
{
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (col >= k || row >= m) return;
    for (unsigned i = 0; i < n; i++)
        sum += a[row * n + i] * b[i * k + col];
    c[row * k + col] = sum;
}

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / common::C_cu_tile_width + 1, (n - 1) / common::C_cu_tile_width + 1, 1);
        dim3 dim_block(common::C_cu_tile_width, common::C_cu_tile_width, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(const double *__restrict__ d_a, const double *__restrict__ d_b, double *__restrict__ d_result, const unsigned n)
{
    __shared__ double tile_a[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tile_b[common::C_cu_tile_width][common::C_cu_tile_width];

    const auto row = blockIdx.y * common::C_cu_tile_width + threadIdx.y;
    const auto col = blockIdx.x * common::C_cu_tile_width + threadIdx.x;
    double tmp = 0;
    unsigned idx;
UNROLL()
    for (unsigned sub = 0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * common::C_cu_tile_width + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = idx >= n * n ? 0 : d_a[idx];

        idx = (sub * common::C_cu_tile_width + threadIdx.y) * n + col;
        tile_b[threadIdx.y][threadIdx.x] = idx >= n * n ? 0 : d_b[idx];

        __syncthreads();
UNROLL()
        for (unsigned k = 0; k < common::C_cu_tile_width; ++k)
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];

        __syncthreads();
    }
    if (row < n && col < n)
        d_result[row * n + col] = tmp;
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters:
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / common::C_cu_tile_width + 1, (n - 1) / common::C_cu_tile_width + 1, 1);
        dim3 dim_block(common::C_cu_tile_width, common::C_cu_tile_width, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(const double *__restrict__ mat_in, double *__restrict__ mat_out, const unsigned rows, const unsigned cols)
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= cols || idy >= rows) return;
    mat_out[idx * rows + idy] = mat_in[idy * cols + idx];
}

/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU,
             for validating GPU results

parameters:
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C)
            to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void matmul(const double *d_a, const double *d_b, double *d_c, const unsigned m, const unsigned n, const unsigned k, const cudaStream_t &strm)
{
    const auto grid_rows = (m + common::C_cu_tile_width - 1) / common::C_cu_tile_width;
    const auto grid_cols = (k + common::C_cu_tile_width - 1) / common::C_cu_tile_width;
    const dim3 dimGrid(grid_cols, grid_rows);
    constexpr dim3 dimBlock(common::C_cu_tile_width, common::C_cu_tile_width);
    if (m == n && n == k)
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    else
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
}

}