//
// Created by zarko on 4/29/24.
//
#include <armadillo>
#include <cublas_v2.h>
#include <driver_types.h>
#include <limits>
#include <magma_d.h>
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
#include "cuda_path.hpp"
#include "onlinesvr.hpp"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include "new_path_kernel.cuh"
#include "kernel_factory.hpp"
#include "common/cuda_util.cuh"

namespace svr {
namespace datamodel {

const uint16_t cusys::n_gpus = common::gpu_handler<cusys::streams_per_gpu>::get().get_gpu_devices_count();

SVRParameters cusys::make_tuning_template(const SVRParameters &example)
{
    SVRParameters template_p;
    template_p.set_kernel_type(example.get_kernel_type());
    template_p.set_svr_kernel_param2(example.get_svr_kernel_param2());
    template_p.set_kernel_param3(example.get_kernel_param3());
    template_p.set_lag_count(example.get_lag_count());
    return template_p;
}

cusys::cusys(
        const arma::mat &train_cuml, const arma::mat &train_label_chunk, const arma::mat &train_W, const SVRParameters &parameters) :
        train_len(train_cuml.n_cols), K_train_len(train_len * train_len), K_train_size(K_train_len * sizeof(double)), n(train_label_chunk.n_cols), train_len_n(train_len * n),
        train_n_size(train_len_n * sizeof(double)), calc_start(PROPS.get_tune_skip()), calc_len(train_len - calc_start), K_calc_len(calc_len * calc_len),
        weighted(train_W.n_elem), train_F_rows(train_cuml.n_rows), train_F_cols(train_cuml.n_cols), K_off(calc_start * train_len + calc_start),
        ref_K(kernel::get_reference_Z(train_label_chunk)), ref_K_mean(common::mean(ref_K)), ref_K_meanabs(common::meanabs(ref_K)),
        template_parameters(make_tuning_template(parameters))
{
    LOG4_TRACE("Train len " << train_len << ", streams per GPU " << unsigned(streams_per_gpu) << ", calc len " << calc_len);
    dx.resize(n_gpus);
    const auto K_train_l = common::extrude_cols<double>(train_label_chunk.t(), train_len);
    const double as_err = common::sumabs<double>((K_train_l + ref_K) * arma::ones<arma::vec>(train_len) - train_len * train_label_chunk);
    if (as_err > 0)
        LOG4_WARN("Tune ftor parameters " << parameters << "(K_train_l + ref_K) * ones are not equal to labels " << as_err / train_len << ", K_train_l "
                                          << common::present(K_train_l) + ", ref_K " << common::present(ref_K) << ", train_label_chunk " << common::present(train_label_chunk) <<
                                          ", train features " << common::present(train_cuml));

    OMP_FOR_i(n_gpus) {
        { // Read-only buffers per device
            DEV_CUSTREAM(i);
            dx[i].d_ref_K = cumallocopy(ref_K, custream);
            if (weighted) dx[i].d_train_W = cumallocopy(train_W, custream);
            dx[i].d_train_cuml = cumallocopy(train_cuml, custream);
            cusyndestroy(custream);
        }
        dx[i].sx.resize(streams_per_gpu);
        UNROLL(streams_per_gpu)
        for (DTYPE(streams_per_gpu) j = 0; j < streams_per_gpu; ++j) { // Read-write buffers, per stream
            auto &dxsx = dx[i].sx[j];
            magma_queue_create(i, &dxsx.ma_queue);
            dxsx.custream = magma_queue_get_cuda_stream(dxsx.ma_queue);
            dxsx.cublas_H = magma_queue_get_cublas_handle(dxsx.ma_queue);
            cu_errchk(cudaMallocAsync((void **) &dxsx.d_K_train, K_train_size, dxsx.custream));
            dxsx.K_train_off = dxsx.d_K_train + calc_start * train_len + calc_start;
        }
    }
}

cusys::~cusys()
{
    OMP_FOR_i(n_gpus) {
        {
            DEV_CUSTREAM(i);
            cu_errchk(cudaFreeAsync(dx[i].d_train_cuml, custream));
            cu_errchk(cudaFreeAsync(dx[i].d_ref_K, custream));
            if (weighted) cu_errchk(cudaFreeAsync(dx[i].d_train_W, custream));
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

// TODO Fix Bug in debug build getting zero valued kernel matrices inside this function
std::tuple<double, double, double> cusys::operator()(const double lambda, const double tau) const
{
    const common::gpu_context_<streams_per_gpu> ctx;
    const auto gpu_id = ctx.phy_id();
    cu_errchk(cudaSetDevice(gpu_id));
    const auto stream_id = ctx.stream_id();
    const auto &dx_ = dx[gpu_id];
    const auto &dxsx = dx_.sx[stream_id];
    auto svr_parameters_ = template_parameters;
    svr_parameters_.set_svr_kernel_param2(lambda);
    svr_parameters_.set_kernel_param3(tau);
    kernel::IKernel<double>::get(svr_parameters_)->d_distances(dx_.d_train_cuml, train_F_rows, train_F_cols, dxsx.d_K_train, dxsx.custream);
    const auto mean = solvers::mean(dxsx.K_train_off, K_train_len, dxsx.custream) - ref_K_mean;
    if (mean != 0) thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.K_train_off, dxsx.K_train_off + K_train_len, dxsx.K_train_off,
        [mean] __device__(const double x) { return x - mean; });
    const auto gamma = solvers::meanabs(dxsx.K_train_off, K_train_len, dxsx.custream) / ref_K_meanabs;
    if (gamma != 1) thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.d_K_train, dxsx.d_K_train + K_train_len, dxsx.d_K_train,
        [gamma] __device__(const double x) { return x / gamma; });
    if (weighted) thrust::transform(thrust::cuda::par.on(dxsx.custream), dxsx.d_K_train, dxsx.d_K_train + K_train_len, dx_.d_train_W, dxsx.d_K_train, thrust::multiplies<double>());
    const auto score = solvers::cu_mae(dxsx.K_train_off, dx_.d_ref_K + K_off, K_calc_len, dxsx.custream);
    return {score, gamma, mean};
}

}
}
