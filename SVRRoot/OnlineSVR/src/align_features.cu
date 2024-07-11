//
// Created by zarko on 7/4/24.
//

#include "align_features.cuh"
#include "common/cuda_util.cuh"

__device__ __host__ inline double vec_dist(const double *__restrict__ mean_L, const double *__restrict__ features, const unsigned n_rows, const double st)
{
    double res = 0;
    for (unsigned r = 0; r < n_rows; ++r) res += abs(mean_L[r] - svr::stretch_ix(features, r, n_rows, st));
    return res;
}

constexpr double C_stretch_multiplier = 1. + 1e-2;
constexpr double C_max_stretch_factor = 1e1;

__global__ void cu_align_features(const double *const features, const double *const mean_L, double *const scores, double *const stretches, const unsigned n_rows,
                                  const unsigned n_cols_superset)
{
    CUDA_STRIDED_FOR_i(n_cols_superset) {
        scores[i] = std::numeric_limits<double>::infinity();
#pragma unroll
        for (double st = 1; st < C_max_stretch_factor; st *= C_stretch_multiplier) {
            const auto dist = vec_dist(mean_L, features + n_rows * i, n_rows, st);
            if (dist < scores[i]) {
                scores[i] = dist;
                stretches[i] = st;
            }
        }
    }
}

namespace svr {

void align_features(const double *const p_double_features, const double *const p_mean_L, double *const p_scores, double *const p_stretches, const unsigned n_rows,
                    const unsigned n_cols_superset)
{
    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreateWithFlags(&custream, cudaStreamNonBlocking));
    const auto double_features_size = n_rows * n_cols_superset * sizeof(double);
    const auto L_size = n_rows * sizeof(double);
    const auto cols_size = n_cols_superset * sizeof(double);
    double *d_double_features, *d_scores, *d_stretches, *d_mean_L;
    //cu_errchk(cudaHostRegister((void *) p_double_features, double_features_size, cudaHostRegisterReadOnly)); // This synchronizes calls across GPUs
    cu_errchk(cudaMallocAsync((void **) &d_double_features, double_features_size, custream));
    cu_errchk(cudaMemcpyAsync(d_double_features, p_double_features, double_features_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_mean_L, L_size, custream));
    cu_errchk(cudaMemcpyAsync(d_mean_L, p_mean_L, L_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_scores, cols_size, custream));
    cu_errchk(cudaMallocAsync((void **) &d_stretches, cols_size, custream));
    cu_align_features<<<CUDA_THREADS_BLOCKS(n_cols_superset), 0, custream>>>(d_double_features, d_mean_L, d_scores, d_stretches, n_rows, n_cols_superset);
    cu_errchk(cudaFreeAsync(d_mean_L, custream));
    cu_errchk(cudaFreeAsync(d_double_features, custream));
    cu_errchk(cudaMemcpyAsync(p_scores, d_scores, cols_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_scores, custream));
    cu_errchk(cudaMemcpyAsync(p_stretches, d_stretches, cols_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_stretches, custream));
    // cu_errchk(cudaHostUnregister((void *) p_double_features));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}

}
