//
// Created by zarko on 16/06/2025.
//

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "common/compatibility.hpp"
GCC_PUSH_DIAGNOSTIC_DISABLE_DANGLING_REF
#include <torch/torch.h>
GCC_DIAGNOSTIC_POP
#include "kernel_tft.hpp"

#include "appcontext.hpp"
#include "tft.hpp"

namespace svr {
namespace kernel {

#define T double

template<> kernel_tft<T>::kernel_tft(datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

torch::Device get_cuda_device()
{
    torch::Device device(torch::kCUDA); // fallback to CPU if needed: torch::kCPU
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA not available. Falling back to CPU.\n";
        device = torch::Device(torch::kCPU);
    }
    // return device;
    return device = torch::Device(torch::kCPU);
}

template<> arma::Mat<T> kernel_tft<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_BEGIN();
    const int64_t n_samples = X.n_cols * Xy.n_cols;
    const int64_t n_manifold_features = X.n_rows + Xy.n_rows;
    const auto device = get_cuda_device();
    TemporalFusionTransformer model(n_manifold_features, PROPS.get_nn_hide_coef() * n_manifold_features, PROPS.get_multiout(), PROPS.get_nn_head_coef() * n_manifold_features, device);
    {
        std::stringstream s(parameters.get_tft_model());
#ifdef COMPRESS_MODEL
        std::stringstream decompressed_stream;
        boost::iostreams::filtering_istream in;
        in.push(boost::iostreams::bzip2_decompressor());
        in.push(s);
        decompressed_stream << in.rdbuf();
        torch::load(model, decompressed_stream);
#else
        torch::load(model, s);
#endif
    }
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
        model->to(device);
    }
    model->eval();
    torch::NoGradGuard no_grad;
    const auto preds = model->forward(manifold_features);
    const auto preds_acc = preds.accessor<float, 3>();
    arma::Mat<T> res(X.n_cols, Xy.n_cols, ARMA_DEFAULT_FILL);
    OMP_FOR_(res.n_elem, SSIMD collapse(2))
    for (uint32_t i = 0; i < X.n_cols; ++i)
        for (uint32_t j = 0; j < Xy.n_cols; ++j)
            res(i, j) = preds_acc[0][i + X.n_cols * j][0];
    LOG4_TRACE("Predicted labels " << common::present(res));
    return res;
}

template<> void kernel_tft<T>::init(const arma::Mat<T> &X, const arma::Mat<T> &Y)
{
    LOG4_BEGIN();
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

    const auto device = get_cuda_device();
    TemporalFusionTransformer model(n_manifold_features, PROPS.get_nn_hide_coef() * n_manifold_features, PROPS.get_multiout(), PROPS.get_nn_head_coef() * n_manifold_features, device);
    if (!device.is_cpu()) {
        model->to(device);
        manifold_features = manifold_features.to(device);
        manifold_labels = manifold_labels.to(device);
    }
    model->train();
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(PROPS.get_k_learn_rate()));
    // torch::nn::BCEWithLogitsLoss criterion;
    for (uint16_t epoch = 0; epoch < PROPS.get_k_epochs(); ++epoch) {
        optimizer.zero_grad();
        const auto output = model->forward(manifold_features);
        // auto loss = torch::nn::functional::cross_entropy(output, labels);
        const auto loss = torch::mse_loss(output, manifold_labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0) LOG4_TRACE("Epoch " << epoch << ", loss " << loss.item<float>());
    }
    if (!device.is_cpu()) model->to(torch::kCPU);
    std::ostringstream raw_model_stream;
    torch::save(model, raw_model_stream);
#ifdef COMPRESS_MODEL
    std::ostringstream compressed_stream;
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::bzip2_compressor());
    out.push(compressed_stream);
    out << raw_model_stream.str();
    boost::iostreams::close(out);
    parameters.set_tft_model(compressed_stream.str());
#else
    parameters.set_tft_model(raw_model_stream.str());
#endif
    LOG4_END();
}

template<> arma::Mat<T> kernel_tft<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_THROW("Not implemeneted.");
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
} // kernel
} // svr
