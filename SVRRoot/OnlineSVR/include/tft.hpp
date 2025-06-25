//
// Created by zarko on 17/06/2025.
//

#ifndef TFT_HPP
#define TFT_HPP

#include "common/compatibility.hpp"
GCC_PUSH_DIAGNOSTIC_DISABLE_DANGLING_REF
#include <torch/nn.h>
GCC_DIAGNOSTIC_POP

namespace svr {
namespace kernel {
// GatedResidualNetwork module used in TFT (simplified)
struct GatedResidualNetworkImpl final : torch::nn::Module
{
    torch::nn::Linear linear1 = nullptr, linear2 = nullptr, linear_skip = nullptr;
    torch::nn::Dropout dropout = nullptr;
    const uint32_t input_size, output_size;
    const bool use_skip;
    const torch::Device device;

    GatedResidualNetworkImpl(const uint32_t input_size_, const uint32_t output_size_, const double dropout_prob, const torch::Device &device_);

    torch::Tensor forward(const torch::Tensor &x);
};

TORCH_MODULE(GatedResidualNetwork);

// Temporal Fusion Transformer main module (simplified)
struct TemporalFusionTransformerImpl final : torch::nn::Module
{
    const uint32_t input_size, hidden_size, output_size, num_heads;
    const torch::Device device;

    // Components
    torch::nn::Linear input_embedding = nullptr;
    torch::nn::LSTM lstm_encoder = nullptr;
    torch::nn::MultiheadAttention multihead_attn = nullptr;
    GatedResidualNetwork grn1 = nullptr;
    GatedResidualNetwork grn2 = nullptr;
    torch::nn::Linear output_layer = nullptr;
    torch::nn::Dropout dropout = nullptr;

    TemporalFusionTransformerImpl(const uint32_t input_size_, const uint32_t hidden_size_, const uint32_t output_size_, const uint32_t num_heads_, const torch::Device &device_);

    torch::Tensor forward(const torch::Tensor &x);
};

TORCH_MODULE(TemporalFusionTransformer);
}
}

#endif //TFT_HPP
