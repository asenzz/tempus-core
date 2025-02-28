//
// Created by zarko on 12/02/2025.
//

#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "common/cuda_util.cuh"
#include "cuqrsolve.cuh"
#include "ScalingFactorService.hpp"

namespace svr {
namespace business {

void ScalingFactorService::cu_scale_calc_I(RPTR(double) v, const size_t n, double &sf, double &dc, const cudaStream_t custream, const cublasHandle_t cublas_H)
{
    dc = solvers::mean(v, n, custream);
    thrust::transform(thrust::cuda::par.on(custream), v, v + n, v,[dc] __device__(const double x) { return x - dc; });
    cu_errchk(cudaStreamSynchronize(custream));
#if 0
    sf = solvers::medianabs(v, n, custream);
#else
    cb_errchk(cublasDasum(cublas_H, n, v, 1, &sf));
    sf /= n;
    sf /= common::C_input_obseg_labels;
#endif
    thrust::transform(thrust::cuda::par.on(custream), v, v + n, v,[sf] __device__(const double x) { return x / sf; });
    cu_errchk(cudaStreamSynchronize(custream));
}

}
}