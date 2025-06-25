//
// Created by zarko on 17/06/2025.
//
#include "common/compatibility.hpp"
GCC_PUSH_DIAGNOSTIC_DISABLE_DANGLING_REF
#include <torch/optim.h>
GCC_DIAGNOSTIC_POP
#include <torch/torch.h>
#include "tft.hpp"

namespace svr {
namespace kernel {

GatedResidualNetworkImpl::GatedResidualNetworkImpl(const uint32_t input_size_, const uint32_t output_size_, const double dropout_prob, const torch::Device &device_)
    : input_size(input_size_), output_size(output_size_), use_skip(input_size == output_size), device(device_)
{
    linear1 = register_module("linear1", torch::nn::Linear(input_size, output_size));
    linear2 = register_module("linear2", torch::nn::Linear(output_size, output_size));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_prob));

    if (use_skip == false) linear_skip = register_module("linear_skip", torch::nn::Linear(input_size, output_size));
    if (!device.is_cpu()) {
        linear1->to(device);
        linear2->to(device);
        dropout->to(device);
    }
}

torch::Tensor GatedResidualNetworkImpl::forward(const torch::Tensor &x)
{
    const auto residual = use_skip ? x : linear_skip->forward(x);
    // Gating mechanism: output = (activation(x1)) * sigmoid(x1)
    // Here using gated linear unit (GLU) style gate simplified as multiplication
    auto x1 = linear2->forward(dropout->forward(torch::relu(linear1->forward(x))));
    auto x2 = torch::sigmoid(x1);
    // Gating mechanism: output = (activation(x1)) * sigmoid(x1)
    // Here using gated linear unit (GLU) style gate simplified as multiplication
    return x1 * x2 + residual;
}


TemporalFusionTransformerImpl::TemporalFusionTransformerImpl(
    const uint32_t input_size_, const uint32_t hidden_size_, const uint32_t output_size_, const uint32_t num_heads_, const torch::Device &device_)
    : input_size(input_size_), hidden_size(hidden_size_), output_size(output_size_), num_heads(num_heads_), device(device_)
{
    // Embedding layer projects input features to hidden dimension
    input_embedding = register_module("input_embedding", torch::nn::Linear(input_size, hidden_size));
    // LSTM Encoder(s)
    lstm_encoder = register_module("lstm_encoder", torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size, hidden_size)));
    // lstm_encoder->options.batch_first(true); // Ensure LSTM expects (batch, seq_len, hidden_size)
    // Multihead Attention for temporal relationships
    multihead_attn = register_module("multihead_attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(hidden_size, num_heads)));
    // Gated Residual Network layers for representation refinement
    grn1 = register_module("grn1", GatedResidualNetwork(hidden_size, hidden_size, .1, device));
    grn2 = register_module("grn2", GatedResidualNetwork(hidden_size, hidden_size, .1, device));
    // Output layer mapping to target
    output_layer = register_module("output_layer", torch::nn::Linear(hidden_size, output_size));
    dropout = register_module("dropout", torch::nn::Dropout(.1));
    if (!device.is_cpu()) {
        input_embedding->to(device);
        lstm_encoder->to(device);
        multihead_attn->to(device);
        grn1->to(device);
        grn2->to(device);
        output_layer->to(device);
        dropout->to(device);
    }
}

torch::Tensor TemporalFusionTransformerImpl::forward(const torch::Tensor &x)
{
    // x shape: (batch_size, seq_len, input_size)

    // Input embedding
    // LSTM encoding
    const auto lstm_output = std::get<0>(lstm_encoder->forward(input_embedding->forward(x))); // (batch, seq_len, hidden_size)

    // Prepare inputs for multihead attention — needs (seq_len, batch, hidden_size)
    const auto lstm_out_trans = lstm_output.transpose(0, 1); // (seq_len, batch, hidden_size)

    // Multihead attention: Self-attention on lstm output
    // Q, K, V are all lstm output for self-attention
    // Transpose back to (batch, seq_len, hidden_size)
    // Residual connection with grn1
    // Another representation refinement with grn2
    // Apply dropout
    // Output layer to produce prediction
    auto xx = std::get<0>(multihead_attn->forward(lstm_out_trans, lstm_out_trans, lstm_out_trans));
    xx = grn1->forward(xx.transpose(0, 1));
    xx = grn2->forward(xx + lstm_output);
    xx = dropout->forward(xx);
    auto res = output_layer->forward(xx);
    if (!res.device().is_cpu()) res = res.to(torch::kCPU);
    return res.contiguous();
}

}
}
