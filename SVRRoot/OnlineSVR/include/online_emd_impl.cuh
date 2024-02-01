//
// Created by zarko on 2/16/22.
//

#ifndef SVR_ONLINE_EMD_IMPL_CUH
#define SVR_ONLINE_EMD_IMPL_CUH

// #define CUDA_OEMD_MULTIGPU // TODO Buggy, test!
// #define CUDA_OUTPUT_FIR
#include <vector>
#include "model/DataRow.hpp"

// Algo old FIROEMD or new FFTOEMD
#define USE_OLD
//#define USE_FFT
#ifdef USE_FFT
#define CUFFT_INPUT_LIMIT 64e6
#endif
//#define USE_VAR_STRETCH
#ifdef USE_VAR_STRETCH
#define MAX_STRETCH_MULT 42
#define MAX_STRETCH_EXP 3
#define MAX_SKEW_STRETCH_COEF 1.
#endif

#ifdef CUDA_OEMD_MULTIGPU
#include "common/gpu_handler.hpp"
#endif

namespace svr {
namespace cuoemd {

void
transform(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<size_t> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef,
        const size_t levels);

}
}

#endif //SVR_ONLINE_EMD_IMPL_CUH
