//
// Created by zarko on 3/10/21.
//

#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "kernel_basic_integration_test.hpp"
#include <cassert>
#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif

#if 0
#include <torch/torch.h>
#include <iostream>

class LSTM
{
    const size_t hidden_layer_size;
    torch::Device device;
    torch::nn::LSTM lstm;
    std::tuple<torch::Tensor, torch::Tensor> hidden_cell;
    torch::nn::Linear linear;
public:
    LSTM(const size_t input_size = 1,
         const size_t hidden_layer_size = 400,
         const size_t output_size = 1,
         const size_t num_layers = 1,
         const double dropout_threshold = 0.,
         const bool bidirectional = false) :
            hidden_layer_size(hidden_layer_size),
            device(torch::Device(torch::kCPU)),
            lstm(torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_layer_size)
                        .dropout(dropout_threshold)
                        .num_layers(num_layers)
                        .bidirectional(bidirectional))),
            linear(torch::nn::Linear(hidden_layer_size, output_size))
    {
        // Use GPU when present, CPU otherwise.
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            LOG4_DEBUG("CUDA is available! Training on GPU.");
        }
        lstm->to(device);
        linear->to(device);
        this->reset_hidden();
    }

    torch::nn::LSTM &get_model() { return lstm; }

    torch::Tensor forward(const torch::Tensor &input_seq)
    {
        input_seq.data().to(device);
        const auto ret = this->lstm->forward(input_seq.view({
            input_seq.size(0), 1, -1}).to(device),this->hidden_cell);
        const auto lstm_output = std::get<0>(ret);
        this->hidden_cell = std::get<1>(ret);
        const auto p_predictions = this->linear->forward(lstm_output);//.view({lstm_output.size(0), -1}));
        LOG4_DEBUG("Predictions: " << p_predictions);
        return p_predictions.slice(1, -1);
    }

    void reset_hidden()
    {
        auto state1 = torch::zeros({1, 1, (long) this->hidden_layer_size}).to(device);
        auto state2 = torch::zeros({1, 1, (long) this->hidden_layer_size}).to(device);
        this->hidden_cell = std::tuple<torch::Tensor, torch::Tensor>(state1, state2);
    }

    const torch::Device &get_device() { return this->device; }
};

typedef std::vector<std::pair<torch::Tensor/* label */, torch::Tensor /* features */>> label_feat_vector_t;

label_feat_vector_t
prepare_input_data(
        const svr::datamodel::vmatrix<double> &labels_mx,
        const svr::datamodel::vmatrix<double> &features_mx)
{
    label_feat_vector_t result;
    for (ssize_t row_ix = 0; row_ix < labels_mx.get_length_rows() && row_ix < features_mx.get_length_rows(); ++row_ix) {
        result.push_back(
                {torch::tensor(labels_mx.get_row_ref(row_ix).to_vector()),
                 torch::tensor(features_mx.get_row_ref(row_ix).to_vector())});
    }
    return result;
}

TEST(lstm_train_predict, basic_integration)
{
    const svr::datamodel::vmatrix<double> *p_labels_mx = svr::datamodel::vmatrix<double>::load("/mnt/faststore/labels_dataset_100_q_svrwave_eurusd_avg_14400_bid_level_0_adjacent_levels_1_lag_400_call_7.out");
    const svr::datamodel::vmatrix<double> *p_features_mx = svr::datamodel::vmatrix<double>::load("/mnt/faststore/features_dataset_100_q_svrwave_eurusd_avg_14400_bid_level_0_adjacent_levels_1_lag_400_call_0.out");
    const auto input_data = prepare_input_data(
            p_labels_mx->extract_rows(0, 1000),
            p_features_mx->extract_rows(0, 1000));

    LSTM model(1, p_features_mx->get_length_cols(), 1);
    auto loss_function = torch::nn::L1Loss();
    loss_function->to(model.get_device());
    auto optimizer = torch::optim::AdamW(
            model.get_model()->parameters(),
            torch::optim::AdamWOptions(0.001));
    const size_t epochs = 1;
    for (size_t i = 0; i < epochs; ++i) {
        for (size_t row = 0; row < input_data.size(); ++row) {
            model.reset_hidden();

            const auto y_pred = model.forward(input_data[row].second);

            optimizer.zero_grad();
            const auto single_loss = loss_function->forward(y_pred, input_data[row].first.to(model.get_device()));
            single_loss.backward();
            optimizer.step();
            LOG4_DEBUG("Epoch " << i << " loss: " << single_loss.item().to<double>());
        }
    }

    const auto test_data = prepare_input_data(
            p_labels_mx->extract_rows(1000, 1100),
            p_features_mx->extract_rows(1000, 1100));
    auto p_predictions = std::vector<double>();
    auto orig_labels = std::vector<double>();
    model.get_model()->eval();
    for (const auto &row: test_data) {
        p_predictions.push_back(model.forward(row.second).item().to<double>());
        orig_labels.push_back(row.first.item().to<double>());
    }

    matplotlibcpp::plot(p_predictions);
    matplotlibcpp::plot(orig_labels);
    matplotlibcpp::show();
#if 0
    // Old code
    const size_t kSequenceLen = 1;
    const size_t kInputDim = 1;
    const size_t kHiddenDim = 5;
    //const size_t kOuputDim = 1;

    torch::Tensor input = torch::empty({kSequenceLen, kInputDim});
    torch::Tensor state1 = torch::zeros({kSequenceLen, kHiddenDim});
    torch::Tensor state2 = torch::zeros({kSequenceLen, kHiddenDim});

    auto input_acc = input.accessor<float, 2>();
    size_t count = 0;
    for (float i = 0.1; i < 0.4; i += 0.1) {
        input_acc[count][0] = i;
        count++;
    }
    input = input.toBackend(c10::Backend::CUDA);

    std::cout << "input = " << input << std::endl;

    auto i_tmp = input.view({input.size(0), 1, -1});
    i_tmp;
    auto s_tmp1 = state1.view({2, state1.size(0) / 2, 1, -1});
    auto s_tmp2 = state2.view({2, state2.size(0) / 2, 1, -1});

    auto rnn_output = time_serie_detector->forward(i_tmp), std::tuple{s_tmp1, s_tmp2});
    std::cout << "rnn_output/output: ";
    auto first = std::get<0>(rnn_output);
    auto second = std::get<1>(rnn_output);
    //pretty_print("rnn_output/state: ", rnn_output.state);
    std::cout << first;//.view({input.size(0), -1});
#endif
}
#endif