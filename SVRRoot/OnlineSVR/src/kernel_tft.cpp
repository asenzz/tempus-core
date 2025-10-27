//
// Created by zarko on 16/06/2025.
//

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <iostream>
#include <vector>
#include "common/compatibility.hpp"
#ifdef USE_TORCH
GCC_PUSH_DIAGNOSTIC_DISABLE_DANGLING_REF
#include <torch/torch.h>
GCC_DIAGNOSTIC_POP
#endif
#include "kernel_tft.hpp"
#include "ScalingFactorService.hpp"
#include "appcontext.hpp"
#ifdef USE_TORCH
#include "tft.hpp"
#endif

namespace svr {
namespace kernel {

#define T double

#ifdef USE_TORCH

torch::Device get_cuda_device()
{
    torch::Device device(torch::kCUDA); // fallback to CPU if needed: torch::kCPU
    if (!torch::cuda::is_available()) {
        LOG4_WARN("CUDA not available. Falling back to CPU.");
        device = torch::Device(torch::kCPU);
    }
    // return device;
    return device = torch::Device(torch::kCPU);
}

#endif

template<> void kernel_tft<T>::load()
{
#ifdef USE_TORCH

    LOG4_BEGIN();

    const auto device = get_cuda_device();
    const auto n_manifold_features = parameters.get_tft_n_classes();
    auto model = std::make_shared<TemporalFusionTransformer>(
            n_manifold_features, PROPS.get_nn_hide_coef() * n_manifold_features, PROPS.get_outputs(), PROPS.get_nn_head_coef() * n_manifold_features, device);
    std::stringstream s(parameters.get_model_blob());

#ifdef COMPRESS_MODEL
    std::stringstream decompressed_stream;
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::bzip2_decompressor());
    in.push(s);
    decompressed_stream << in.rdbuf();
    torch::load(*model, decompressed_stream);
#else
    torch::load(*model, s);
#endif

    parameters.set_manifold(model);
	
#endif //	#ifdef USE_TORCH
}

template<> void kernel_tft<T>::save()
{
#ifdef USE_TORCH

    const auto tftmod = *std::get<TemporalFusionTransformer_ptr>(parameters.get_manifold());
    std::ostringstream raw_model_stream;
    torch::save(tftmod, raw_model_stream);
#ifdef COMPRESS_MODEL
    std::ostringstream compressed_stream;
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::bzip2_compressor());
    out.push(compressed_stream);
    out << raw_model_stream.str();
    boost::iostreams::close(out);
    parameters.set_model_blob(compressed_stream.str());
#else
    parameters.set_model_blob(raw_model_stream.str());
#endif

#endif // #ifdef USE_TORCH
}

template<> kernel_tft<T>::kernel_tft(datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

template<> arma::Mat<T> kernel_tft<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
#ifdef USE_TORCH

    LOG4_BEGIN();
    const int64_t n_samples = X.n_cols * Xy.n_cols;
    const int64_t n_manifold_features = X.n_rows + Xy.n_rows;
    const auto device = get_cuda_device();
    auto p_model = *std::get<TemporalFusionTransformer_ptr>(parameters.get_manifold());
    auto manifold_features = torch::empty({1, n_samples, n_manifold_features});
    auto manifold_features_acc = manifold_features.accessor<float, 3>();
    OMP_FOR_(n_samples, collapse(2) SSIMD)
    for (uint32_t i = 0; i < X.n_cols; ++i)
        for (uint32_t j = 0; j < Xy.n_cols; ++j) {
            const arma::vec F = arma::join_cols(X.col(i), Xy.col(j));
            for (uint32_t k = 0; k < F.n_elem; ++k) manifold_features_acc[0][i + X.n_cols * j][k] = F[k];
        }
    if (!device.is_cpu()) {
        manifold_features = manifold_features.to(device);
        p_model->to(device);
    }
    p_model->eval();
    torch::NoGradGuard no_grad;
    const auto preds = p_model->forward(manifold_features);
    const auto preds_acc = preds.accessor<float, 3>();
    arma::Mat<T> res(X.n_cols, Xy.n_cols, ARMA_DEFAULT_FILL);
    OMP_FOR_(res.n_elem, SSIMD collapse(2))
    for (uint32_t i = 0; i < X.n_cols; ++i)
        for (uint32_t j = 0; j < Xy.n_cols; ++j)
            res(i, j) = preds_acc[0][i + X.n_cols * j][0];
    LOG4_TRACE("Predicted labels " << common::present(res));
    common::unscale_I(res, parameters.get_svr_kernel_param(), parameters.get_min_Z());
    return res;

#else
    return {};
#endif
}

template<> void kernel_tft<T>::init(datamodel::OnlineSVR &svrmod, const uint32_t chunk_ix)
{
#ifdef USE_TORCH
    LOG4_BEGIN();
    const auto &X = svrmod.get_X(chunk_ix);
    const auto &Y = svrmod.get_Y(chunk_ix);
    assert(Y.n_cols == 1);
    const int64_t n_samples = X.n_cols;
    const int64_t n_samples_2 = n_samples * n_samples;
    const int64_t n_manifold_features = X.n_rows * 2;
    // Torch uses row-major matrices
    auto manifold_features = torch::empty({1, n_samples_2, n_manifold_features});
    auto manifold_labels = torch::empty({1, n_samples_2, int64_t(Y.n_cols)});
    auto manifold_features_acc = manifold_features.accessor<float, 3>();
    auto manifold_labels_acc = manifold_labels.accessor<float, 3>();
    OMP_FOR_(n_samples_2, collapse(2) firstprivate(n_samples_2) SSIMD)
    for (DTYPE(n_samples) i = 0; i < n_samples; ++i)
        for (DTYPE(n_samples) j = 0; j < n_samples; ++j) {
            const auto row = i + n_samples * j;
            const arma::rowvec label_row = Y.row(i) - Y.row(j);
            for (uint32_t k = 0; k < label_row.n_elem; ++k) manifold_labels_acc[0][row][k] = label_row[k];
            const arma::vec F = arma::join_cols(X.col(i), X.col(j));
            for (uint32_t k = 0; k < F.n_elem; ++k) manifold_features_acc[0][row][k] = F[k];
        }

    const auto dc = manifold_labels.mean().item<float>();
    manifold_labels -= dc;
    const auto sf = manifold_labels.abs().mean().item<float>();
    manifold_labels /= sf;
    LOG4_TRACE("Scaled manifold labels, scaling factor " << sf << ", dc offset " << dc);
    parameters.set_svr_kernel_param(sf);
    parameters.set_min_Z(dc);
    parameters.set_tft_n_classes(n_manifold_features);
    const auto device = get_cuda_device();
    auto p_tftmod = ptr<TemporalFusionTransformer>(
            n_manifold_features, PROPS.get_nn_hide_coef() * n_manifold_features, PROPS.get_outputs(), PROPS.get_nn_head_coef() * n_manifold_features, device);
    auto tftmod = *p_tftmod;
    if (!device.is_cpu()) {
        tftmod->to(device);
        manifold_features = manifold_features.to(device);
        manifold_labels = manifold_labels.to(device);
    }
    tftmod->train();
    auto optimizer = torch::optim::Adam(tftmod->parameters(), torch::optim::AdamOptions(PROPS.get_k_learn_rate()));
    // torch::nn::BCEWithLogitsLoss criterion;
    for (uint16_t epoch = 0; epoch < PROPS.get_k_epochs(); ++epoch) {
        optimizer.zero_grad();
        const auto output = tftmod->forward(manifold_features);
        // auto loss = torch::nn::functional::cross_entropy(output, labels);
        const auto loss = torch::mse_loss(output, manifold_labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0) LOG4_TRACE("Epoch " << epoch << ", loss " << loss.item<float>());
    }
    parameters.set_manifold(p_tftmod);
    kernel_base::wrapup(svrmod, chunk_ix);
    LOG4_END();
#endif // #ifdef USE_TORCH
}

template<> arma::Mat<T> kernel_tft<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_THROW("Not implemented.");
    return {};
}

template<> void kernel_tft<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    LOG4_THROW("Not implemented.");
}

template<> void kernel_tft<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    LOG4_THROW("Not implemented.");
}

template<> void kernel_tft<T>::update(datamodel::OnlineSVR &svrmod, const uint32_t chunk_ix, const arma::Mat<T> &new_x, const arma::Mat<T> &new_y)
{
#ifdef USE_TORCH

    LOG4_BEGIN();

    const auto &X_t = svrmod.get_X(chunk_ix);
    const auto &Y = svrmod.get_Y(chunk_ix);
    const uint32_t n_new_samples = new_x.n_rows;
    const uint32_t n_samples = X_t.n_cols;
    const uint32_t n_samples_2 = n_new_samples * n_samples;
    const uint32_t n_manifold_features = X_t.n_rows * 2;
    // LightGBM uses row-major matrices
    auto manifold_features = torch::empty({1, n_samples_2, n_manifold_features});
    auto manifold_labels = torch::empty({1, n_samples_2, int64_t(Y.n_cols)});
    auto manifold_features_acc = manifold_features.accessor<float, 3>();
    auto manifold_labels_acc = manifold_labels.accessor<float, 3>();
    OMP_FOR_(n_samples_2, collapse(2) firstprivate(n_samples_2) SSIMD)
    for (DTYPE(n_samples) i = 0; i < n_new_samples; ++i)
        for (DTYPE(n_samples) j = 0; j < n_samples; ++j) {
            const auto row = i + n_samples * j;
            const arma::rowvec label_row = new_y.row(i) - Y.row(j);
            for (uint32_t k = 0; k < label_row.n_elem; ++k) manifold_labels_acc[0][row][k] = label_row[k];
            const arma::vec F = arma::join_cols(new_x.row(i).t(), X_t.col(j));
            for (uint32_t k = 0; k < F.n_elem; ++k) manifold_features_acc[0][row][k] = F[k];
        }
    business::ScalingFactorService::scale_I(manifold_labels, parameters.get_svr_kernel_param(), parameters.get_min_Z());

    const auto device = get_cuda_device();
    auto p_tftmod = std::get<TemporalFusionTransformer_ptr>(parameters.get_manifold());
    auto tftmod = *p_tftmod;
    if (!device.is_cpu()) {
        tftmod->to(device);
        manifold_features = manifold_features.to(device);
        manifold_labels = manifold_labels.to(device);
    }
    tftmod->train();
    auto optimizer = torch::optim::Adam(tftmod->parameters(), torch::optim::AdamOptions(PROPS.get_k_learn_rate()));
    // torch::nn::BCEWithLogitsLoss criterion;
    for (uint16_t epoch = 0; epoch < PROPS.get_k_epochs(); ++epoch) {
        optimizer.zero_grad();
        const auto output = tftmod->forward(manifold_features);
        // auto loss = torch::nn::functional::cross_entropy(output, labels);
        const auto loss = torch::mse_loss(output, manifold_labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0) LOG4_TRACE("Epoch " << epoch << ", loss " << loss.item<float>());
    }

    if (!device.is_cpu()) tftmod->to(torch::kCPU);
    parameters.set_manifold(p_tftmod);
    kernel_base::wrapup(svrmod, chunk_ix);
    LOG4_END();

#endif // #ifdef USE_TORCH
}

} // kernel
} // svr
