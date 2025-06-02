//
// Created by zarko on 4/29/24.
//
#include <armadillo>
#include <cublas_v2.h>
#include <driver_types.h>
#include <limits>
#include <thread>
#include <magma_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <tuple>
#include <thrust/async/reduce.h>
#include "common/compatibility.hpp"
#include "common/logging.hpp"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "cuqrsolve.cuh"
#include "onlinesvr.hpp"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include "kernel_base.cuh"
#include "kernel_factory.hpp"
#include "common/cuda_util.cuh"

namespace svr {
namespace datamodel {
constexpr double C_min_gamma = 1e-12;
constexpr double C_max_gamma = 1e12;

SVRParameters OnlineMIMOSVR::make_tuning_template(const SVRParameters &example)
{
#if 0
    SVRParameters template_p;
    template_p.set_kernel_type(example.get_kernel_type());
    template_p.set_svr_kernel_param2(example.get_svr_kernel_param2());
    template_p.set_kernel_param3(example.get_kernel_param3());
    template_p.set_lag_count(example.get_lag_count());
    template_p.set_H_feedback(example.get_H_feedback());
    template_p.set_D_feedback(example.get_V_feedback());
    template_p.set_V_feedback(example.get_V_feedback());
    template_p.set_min_Z(example.get_min_Z());
    template_p.set_max_Z(example.get_max_Z());
    return template_p;
#else
    return example;
#endif
}

cutuner::cutuner(
    const arma::mat &train_F, const arma::mat &train_label_chunk, const arma::mat &train_W, const SVRParameters &parameters) : train_len(train_F.n_cols), K_train_len(train_len * train_len),
    K_train_size(K_train_len * sizeof(double)), n(train_label_chunk.n_cols), train_len_n(train_len * n),
    train_n_size(train_len_n * sizeof(double)), calc_start(PROPS.get_tune_skip()), calc_len(train_len - calc_start), K_calc_len(calc_len * calc_len),
    weighted(train_W.n_elem), train_F_rows(train_F.n_rows), K_off(calc_start * train_len + calc_start),
    ref_K(kernel::get_reference_Z(train_label_chunk)), ref_K_mean(common::mean(ref_K)), ref_K_meanabs(common::meanabs(ref_K)),
    template_parameters(OnlineMIMOSVR::make_tuning_template(parameters)), train_F(train_F), n_gpus(common::gpu_handler<cutuner::streams_per_gpu>::get().get_gpu_devices_count())
{
    LOG4_TRACE("Train len " << train_len << ", streams per GPU " << streams_per_gpu << ", calc len " << calc_len);
    dx.resize(n_gpus);
    const auto K_train_l = common::extrude_rows(train_label_chunk.t().eval(), train_len);
    const double as_err = common::sumabs<double>((K_train_l + ref_K) * arma::ones<arma::vec>(train_len) - train_len * train_label_chunk);
    if (as_err > 0)
        LOG4_WARN("Tune ftor parameters " << parameters << "(K_train_l + ref_K) * ones are not equal to labels " << as_err / train_len << ", K_train_l "
        << common::present(K_train_l) + ", ref_K " << common::present(ref_K) << ", train_label_chunk " << common::present(train_label_chunk) <<
        ", train features " << common::present(train_F));
    OMP_FOR_i(n_gpus) {
        {
            // Read-only buffers per device
            DEV_CUSTREAM(i);
            dx[i].d_ref_K = cumallocopy(ref_K, custream);
            if (weighted) dx[i].d_train_W = cumallocopy(train_W, custream);
            dx[i].d_train_F = cumallocopy(train_F, custream);
            cusyndestroy(custream);
        }
        dx[i].sx.resize(streams_per_gpu);
        UNROLL(streams_per_gpu)
        for (DTYPE(streams_per_gpu) j = 0; j < streams_per_gpu; ++j) {
            // Read-write buffers, per stream
            auto &dxsx = dx[i].sx[j];
            magma_queue_create(i, &dxsx.ma_queue);
            dxsx.custream = magma_queue_get_cuda_stream(dxsx.ma_queue);
            dxsx.cublas_H = magma_queue_get_cublas_handle(dxsx.ma_queue);
            cu_errchk(cudaMallocAsync((void **) &dxsx.d_K_train, K_train_size, dxsx.custream));
            dxsx.K_train_off = dxsx.d_K_train + calc_start * train_len + calc_start;
        }
    }
}

cutuner::~cutuner()
{
    OMP_FOR_i(n_gpus) {
        {
            DEV_CUSTREAM(i);
            cu_errchk(cudaFreeAsync(dx[i].d_train_F, custream));
            cu_errchk(cudaFreeAsync(dx[i].d_ref_K, custream));
            if (weighted) cu_errchk(cudaFreeAsync(dx[i].d_train_W, custream));
            if (template_parameters.get_kernel_type() == e_kernel_type::PATH) cu_errchk(cudaFreeAsync(dx[i].d_D_paths, custream));
            cusyndestroy(custream);
        }

        UNROLL(streams_per_gpu)
        for (DTYPE(streams_per_gpu) j = 0; j < streams_per_gpu; ++j) {
            auto &dxsx = dx[i].sx[j];
            cu_errchk(cudaFreeAsync(dxsx.d_K_train, dxsx.custream));
            magma_queue_destroy(dxsx.ma_queue);
        }
    }
}

std::tuple<double, double, double> cutuner::normalize_result(const dev_ctx &dx_, const dev_ctx::stream_ctx &dxsx, const SVRParameters &parameters) const
{
    const auto mean = solvers::mean(dxsx.K_train_off, K_train_len, dxsx.custream) - ref_K_mean;
    if (mean != 0)
        thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.K_train_off, dxsx.K_train_off + K_train_len, dxsx.K_train_off,
                          [mean] __device__(const double x) { return x - mean; });
    const auto gamma = solvers::meanabs(dxsx.K_train_off, K_train_len, dxsx.custream) / ref_K_meanabs;
    if (gamma < C_min_gamma || gamma > C_max_gamma) {
        LOG4_ERROR("Gamma not sane " << gamma << ", ref meanabs " << ref_K_meanabs << ", ref mean " << ref_K_mean << ", Z mean " << mean << ", parameters " << parameters);
        return {std::numeric_limits<double>::max(), 0, 0};
    }
    if (gamma != 1)
        thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.K_train_off, dxsx.K_train_off + K_train_len, dxsx.K_train_off,
                          [gamma] __device__(const double x) { return x / gamma; });
    if (weighted)
        thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.d_K_train, dxsx.d_K_train + K_train_len, dx_.d_train_W, dxsx.d_K_train,
                          thrust::multiplies<double>());
    const auto score = solvers::cu_mae(dxsx.K_train_off, dx_.d_ref_K + K_off, K_calc_len, dxsx.custream);
    LOG4_TRACE("Mean " << mean << ", gamma " << gamma << ", score " << score << ", parameters " << parameters);
    return {score, gamma, mean};
}

std::tuple<double, double, double> cutuner::phase1(const double tau, const double H, const double D, const double V) const
{
    const common::gpu_context_<streams_per_gpu> ctx;
    const auto gpu_id = ctx.phy_id();
    const auto stream_id = ctx.stream_id();
    const auto &dx_ = dx[gpu_id];
    const auto &dxsx = dx_.sx[stream_id];
    auto svr_parameters_ = template_parameters;
    svr_parameters_.set_kernel_param3(tau);
    svr_parameters_.set_H_feedback(H);
    svr_parameters_.set_D_feedback(D);
    svr_parameters_.set_V_feedback(V);
    cu_errchk(cudaSetDevice(gpu_id));
    kernel::IKernel<double>::get<kernel::kernel_path<double> >(svr_parameters_)->kernel_base::d_distances(dx_.d_train_F, train_F_rows, train_len, dxsx.d_K_train, dxsx.custream);
    return normalize_result(dx_, dxsx, svr_parameters_);
}

void cutuner::prepare_second_phase(const SVRParameters &first_phase_parameters)
{
    template_parameters = first_phase_parameters;

    arma::mat Z;
    if (template_parameters.get_kernel_type() == e_kernel_type::PATH)
        Z = kernel::IKernel<double>::get<kernel::kernel_path<double> >(template_parameters)->kernel_base::distances(train_F);
    OMP_FOR_i(n_gpus) {
        // Read-only buffers per device
        DEV_CUSTREAM(i);
        dx[i].d_train_F = cumallocopy(train_F, custream);
        if (template_parameters.get_kernel_type() == e_kernel_type::PATH) dx[i].d_D_paths = cumallocopy(Z);
        cusyndestroy(custream);
    }
}

// TODO Fix Bug in debug build getting zero valued kernel matrices inside this function
std::tuple<double, double, double> cutuner::phase2(const double lambda) const
{
    const common::gpu_context_<streams_per_gpu> ctx;
    const auto gpu_id = ctx.phy_id();
    cu_errchk(cudaSetDevice(gpu_id));
    const auto stream_id = ctx.stream_id();
    const auto &dx_ = dx[gpu_id];
    const auto &dxsx = dx_.sx[stream_id];
    auto svr_parameters_ = template_parameters;
    svr_parameters_.set_svr_kernel_param2(lambda);
    thrust::transform(thrust::cuda::par.on(dxsx.custream), dx_.d_D_paths, dx_.d_D_paths + train_len, dxsx.d_K_train,
                      [lambda] __device__ (const double z) { return kernel::K_from_Z(z, lambda); });
    return normalize_result(dx_, dxsx, svr_parameters_);
}
}
}
