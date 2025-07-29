//
// Created by zarko on 19/03/2025.
//

#ifndef SVR_KERNEL_BASE_CUH
#define SVR_KERNEL_BASE_CUH

#include <cmath>
#include <cublas_v2.h>
#include <magma_types.h>
#include "common/gpu_handler.hpp"
#include "common/cuda_util.cuh"
#include "util/math_utils.hpp"

namespace svr {
namespace kernel {

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const float degree)
{
    return copysign(pow(abs(z), degree), z);
}

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const T divisor, const T mean)
{
    return common::scale(z, divisor, mean);
}

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const T divisor, const T mean, const float degree)
{
    return common::scale(K_from_Z(z, degree), divisor, mean);
}

class cutuner
{
    static constexpr uint16_t streams_per_gpu = 1;
    const uint16_t n_gpus;
    datamodel::SVRParameters template_parameters;
    const bool weighted;
    const uint32_t n, train_len, calc_start, calc_len, train_F_rows; // calc len of 3000 seems to work best
    const uint64_t K_train_len, K_train_size, K_calc_len, K_off, train_len_n, train_n_size;
    const arma::mat ref_K, train_F;
    const double ref_K_mean, ref_K_meanabs;

public:
    struct dev_ctx
    {
        struct stream_ctx
        {
            cudaStream_t custream;
            cublasHandle_t cublas_H;
            magma_queue_t ma_queue;
            double *d_K_train, *K_train_off;
        };

        double *d_train_F, *d_train_W, *d_ref_K, *d_D_paths;
        std::deque<stream_ctx> sx;
    };

    std::deque<dev_ctx> dx;

    cutuner(const arma::mat &train_F, const arma::mat &train_label_chunk, const arma::mat &train_W, const datamodel::SVRParameters &parameters);

    ~cutuner();

    std::tuple<double, double, double> normalize_result(const dev_ctx &dx_, const dev_ctx::stream_ctx &dxsx, const datamodel::SVRParameters &parameters) const;

    void prepare_second_phase(const datamodel::SVRParameters &first_phase_parameters);

    std::tuple<double, double, double> phase1(double tau, double H, double D, double V) const;

    std::tuple<double, double, double> phase2(double lambda) const;
};

}
}

#endif //SVR_KERNEL_BASE_CUH
