//
// Created by zarko on 12/02/2025.
//

#include <thrust/reduce.h>
#include "ScalingFactorService.hpp"
#include "common/cuda_util.cuh"
#include "cuqrsolve.cuh"

namespace svr {
namespace business {

void ScalingFactorService::cu_scale_calc_I(RPTR(double) v, const size_t n, double &sf, double &dc, const cudaStream_t custream)
{
    dc = solvers::mean(v, n, custream);
    thrust::transform(thrust::cuda::par.on(custream), v, v + n, v,[dc] __device__(const double x) { return x - dc; });
    sf = solvers::meanabs(v, n, custream) / common::C_input_obseg_labels;
    thrust::transform(thrust::cuda::par.on(custream), v, v + n, v,[sf] __device__(const double x) { return x / sf; });
    // cu_errchk(cudaStreamSynchronize(custream));
}

}
}