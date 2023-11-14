//
// Created by zarko on 3/10/21.
//

#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "kernel_basic_integration_test.hpp"
#include <cassert>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#if 0

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/rnn_ops_internal.h"
#include "tensorflow/cc/ops/rkz_rnn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/training/queue_runner.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#pragma GCC diagnostic pop

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace tensorflow::ops::internal;
using namespace std;


const char test_content[] =
        "hello world hello world hello world hello world hello world hello world "
        "hello world hello world hello world hello world hello world hello world "
        "hello world hello world hello world hello world hello world hello world "
        "hello world hello world hello world hello world hello world hello world "
        "hello world hello world hello world hello world hello world hello world "
        "hello world hello world hello world";

// Adjustable parameters
#define NUM_UNIT 16   // HIDDEN_SIZE
#define TIME_LEN 20   // NUM_STEPS
#define BATCH_SIZE 4  //
#define TRAINING_STEPS 20001

// Don't change
#define INPUT_SIZE 8  // "helo wrd"
#define SEQ_LENGTH TIME_LEN * BATCH_SIZE
#define TEST_TIME_LEN 1
#define TEST_BATCH_SIZE 1  //
#define TEST_SEQ_LENGTH (TEST_TIME_LEN * TEST_BATCH_SIZE)

namespace tensorflow {
// Helpers for loading a TensorFlow plugin (a .so file).
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len);


QueueRunnerDef BuildQueueRunnerDef(
        const std::string& queue_name, const std::vector<std::string>& enqueue_ops,
        const std::string& close_op, const std::string& cancel_op,
        const std::vector<tensorflow::error::Code>& queue_closed_error_codes) {
    QueueRunnerDef queue_runner_def;
    *queue_runner_def.mutable_queue_name() = queue_name;
    for (const std::string& enqueue_op : enqueue_ops) {
        *queue_runner_def.mutable_enqueue_op_name()->Add() = enqueue_op;
    }
    *queue_runner_def.mutable_close_op_name() = close_op;
    *queue_runner_def.mutable_cancel_op_name() = cancel_op;
    for (const auto& error_code : queue_closed_error_codes) {
        *queue_runner_def.mutable_queue_closed_exception_types()->Add() =
                error_code;
    }
    return queue_runner_def;
}

string DetailedDebugString(const Tensor& tensor) {
    return strings::StrCat("Tensor<type: ", DataTypeString(tensor.dtype()),
                           " shape: ", tensor.shape().DebugString(),
                           " values: ", tensor.SummarizeValue(-1, true), ">");
}
}  // namespace tensorflow

#define FEATURES_DATA ("../SVRRoot/OnlineSVR/test/test_data/level_1_eurusd_avg_5m_features.vmatrix")
#define LABELS_DATA ("../SVRRoot/OnlineSVR/test/test_data/level_1_eurusd_avg_5m_labels.vmatrix")
#define DUMP_FILE ("path_train_predict_dump.txt")
#define SAVED_MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/online_svr_final_pathk_test.txt")

#define SVR_COST (90000)
#define SVR_EPSILON (0.00001)
#define SVR_KERNEL_PARAM1 (3)
#define SVR_KERNEL_PARAM2 (1)
#define SVR_DECREMENTAL_DISTANCE (30000)
#define SVR_ADJACENT_LEVELS_RATIO (0.33)
#define SVR_KERNEL_TYPE (kernel_type_e::PATH)
#define SVR_LAG_COUNT (50)

TEST(lstm_train_predict_tf, basic_integration)
{
    const SVRParameters model_svr_parameters(
            0, 0, "", "", 1, SVR_COST, SVR_EPSILON, SVR_KERNEL_PARAM1,
            SVR_KERNEL_PARAM2, SVR_DECREMENTAL_DISTANCE,
            SVR_ADJACENT_LEVELS_RATIO, SVR_KERNEL_TYPE, SVR_LAG_COUNT);

    // Scope
    Scope root = Scope::NewRootScope();

    //
    // Data
    //
    // vocabulary vs index
    const std::map<int, char> index_vocab_dic = {{0, ' '}, {1, 'e'}, {2, 'd'},
                                                 {3, 'h'}, {4, 'l'}, {5, 'o'},
                                                 {6, 'r'}, {7, 'w'}};
    std::map<char, int> vocab_index_dic;
    for (auto iter = index_vocab_dic.begin(); iter != index_vocab_dic.end();
         iter++) {
        // LOG(INFO) << "----------------index_vocab_dic: " << iter->first <<' ' <<
        // iter->second;

        vocab_index_dic.insert(std::pair<char, int>(iter->second, iter->first));
    }

    //
    // Train
    //

    // Trainable parameters
    auto rate = Const(root, {0.01f});

    auto w_layer1 = Variable(
            root, {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4},
            DT_FLOAT);  // (input_size + cell_size, cell_size * 4) for {i, c, f, o}
    auto random_value =
            RandomNormal(root, {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
    auto assign_w_layer1 =
            Assign(root, w_layer1, Multiply(root, random_value, rate));

    auto w_layer2 = Variable(
            root, {NUM_UNIT + NUM_UNIT, NUM_UNIT * 4},
            DT_FLOAT);  // (cell_size + cell_size, cell_size * 4) for {i, c, f, o}
    auto random_value21 =
            RandomNormal(root, {NUM_UNIT + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
    auto assign_w_layer2 =
            Assign(root, w_layer2, Multiply(root, random_value21, rate));

    auto b_layer1 = Variable(root, {NUM_UNIT * 4}, DT_FLOAT);
    Tensor b_zero_tensor(DT_FLOAT, TensorShape({NUM_UNIT * 4}));
    b_zero_tensor.vec<float>().setZero();
    auto assign_b_layer1 = Assign(root, b_layer1, ZerosLike(root, b_zero_tensor));

    auto b_layer2 = Variable(root, {NUM_UNIT * 4}, DT_FLOAT);
    auto assign_b_layer2 = Assign(root, b_layer2, ZerosLike(root, b_zero_tensor));

    // used to compute dh (h_grad)
    auto w_y = Variable(root, {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
    auto random_value2 = RandomNormal(root, {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
    auto assign_w_y = Assign(root, w_y, Multiply(root, random_value2, rate));

    auto b_y = Variable(root, {INPUT_SIZE}, DT_FLOAT);
    Tensor b_y_zero_tensor(DT_FLOAT, TensorShape({INPUT_SIZE}));
    b_y_zero_tensor.vec<float>().setZero();
    auto assign_b_y = Assign(root, b_y, ZerosLike(root, b_y_zero_tensor));

    // Gradient accum parameters start here
    auto ada_w_layer1 =
            Variable(root, {INPUT_SIZE + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
    auto assign_ada_w_layer1 =
            Assign(root, ada_w_layer1, ZerosLike(root, w_layer1));

    auto ada_w_layer2 =
            Variable(root, {NUM_UNIT + NUM_UNIT, NUM_UNIT * 4}, DT_FLOAT);
    auto assign_ada_w_layer2 =
            Assign(root, ada_w_layer2, ZerosLike(root, w_layer2));

    auto ada_b_layer1 = Variable(root, {NUM_UNIT * 4}, DT_FLOAT);
    auto assign_ada_b_layer1 =
            Assign(root, ada_b_layer1, ZerosLike(root, b_layer1));

    auto ada_b_layer2 = Variable(root, {NUM_UNIT * 4}, DT_FLOAT);
    auto assign_ada_b_layer2 =
            Assign(root, ada_b_layer2, ZerosLike(root, b_layer2));

    auto ada_w_y = Variable(root, {INPUT_SIZE, NUM_UNIT}, DT_FLOAT);
    auto assign_ada_w_y = Assign(root, ada_w_y, ZerosLike(root, w_y));

    auto ada_b_y = Variable(root, {INPUT_SIZE}, DT_FLOAT);
    auto assign_ada_b_y = Assign(root, ada_b_y, ZerosLike(root, b_y));

    auto cs = Variable(root.WithOpName("cs"), {TIME_LEN, BATCH_SIZE, NUM_UNIT},
                       DT_FLOAT);
    Tensor cs_zero_tensor(DT_FLOAT,
                          TensorShape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));
    cs_zero_tensor.tensor<float, 3>().setZero();
    auto assign_cs = Assign(root, cs, ZerosLike(root, cs_zero_tensor));

    // Placeholders
    auto x = Placeholder(
            root, DT_FLOAT,
            Placeholder::Shape({TIME_LEN, BATCH_SIZE,
                                INPUT_SIZE}));  // (timelen, batch_size, num_inputs)
    auto y = Placeholder(
            root, DT_INT64,
            Placeholder::Shape({TIME_LEN, BATCH_SIZE}));  // (timelen, batch_size)

    // each layer has its states
    auto cs_prev_layer1 = Placeholder(
            root, DT_FLOAT,
            Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));  // (batch_size, cell_size)
    auto h_prev_layer1 =
            Placeholder(root, DT_FLOAT, Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));
    auto cs_grad_layer1 = Placeholder(
            root, DT_FLOAT, Placeholder::Shape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));

    auto cs_prev_layer2 = Placeholder(
            root, DT_FLOAT,
            Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));  // (batch_size, cell_size)
    auto h_prev_layer2 =
            Placeholder(root, DT_FLOAT, Placeholder::Shape({BATCH_SIZE, NUM_UNIT}));
    auto cs_grad_layer2 = Placeholder(
            root, DT_FLOAT, Placeholder::Shape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));

    // LSTM
    auto block_lstm_layer1 = BlockLSTM(
            root, Const<int64>(root, {TIME_LEN}),  // seq_len_max,
            x, cs_prev_layer1, h_prev_layer1, w_layer1,
            Const<float>(
                    root, 0,
                    TensorShape({NUM_UNIT})),  // wci, used when use_peephole is true
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wcf
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wco
            b_layer1);
    auto block_lstm_layer2 = BlockLSTM(
            root, Const<int64>(root, {TIME_LEN}),  // seq_len_max,
            block_lstm_layer1.h,  // the input is the output h of the previous layer
            cs_prev_layer2, h_prev_layer2, w_layer2,
            Const<float>(
                    root, 0,
                    TensorShape({NUM_UNIT})),  // wci, used when use_peephole is true
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wcf
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wco
            b_layer2);
    auto rnn_softmax_loss =
            RNNSoftmaxLoss(root, block_lstm_layer2.h, y, w_y, b_y, cs);
    LOG(INFO) << "Node building status: " << root.status();

    // Gradients
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(
            root, {rnn_softmax_loss.loss},
            {w_layer1, b_layer1, w_layer2, b_layer2, w_y, b_y}, &grad_outputs));
    LOG(INFO) << "Node building status: " << root.status();

    // Gradient
    auto lr = Cast(root, 0.1, DT_FLOAT);

    // alternative of ApplyAdagrad
    auto apply_w_layer1 =
            ApplyAdagrad(root, w_layer1, ada_w_layer1, lr, grad_outputs[0]);
    auto apply_b_layer1 =
            ApplyAdagrad(root, b_layer1, ada_b_layer1, lr, grad_outputs[1]);
    auto apply_w_layer2 =
            ApplyAdagrad(root, w_layer2, ada_w_layer2, lr, grad_outputs[2]);
    auto apply_b_layer2 =
            ApplyAdagrad(root, b_layer2, ada_b_layer2, lr, grad_outputs[3]);
    auto apply_w_y = ApplyAdagrad(
            root, w_y, ada_w_y, lr, grad_outputs[4]);  // only apply on output layer
    auto apply_b_y = ApplyAdagrad(
            root, b_y, ada_b_y, lr, grad_outputs[5]);  // only apply on output layer

    // BlockLSTM and RNNSoftmaxLossHGrad For Eval
    // Placeholders
    auto x_eval = Placeholder(
            root, DT_FLOAT,
            Placeholder::Shape({TEST_TIME_LEN, TEST_BATCH_SIZE, INPUT_SIZE}));
    auto y_eval = Placeholder(
            root, DT_INT64, Placeholder::Shape({TEST_TIME_LEN, TEST_BATCH_SIZE}));
    auto cs_prev_eval_layer1 =
            Placeholder(root, DT_FLOAT,
                        Placeholder::Shape(
                                {TEST_BATCH_SIZE, NUM_UNIT}));  // (batch_size, cell_size)
    auto h_prev_eval_layer1 = Placeholder(
            root, DT_FLOAT, Placeholder::Shape({TEST_BATCH_SIZE, NUM_UNIT}));
    auto cs_prev_eval_layer2 =
            Placeholder(root, DT_FLOAT,
                        Placeholder::Shape(
                                {TEST_BATCH_SIZE, NUM_UNIT}));  // (batch_size, cell_size)
    auto h_prev_eval_layer2 = Placeholder(
            root, DT_FLOAT, Placeholder::Shape({TEST_BATCH_SIZE, NUM_UNIT}));
    // BlockLSTM
    auto block_lstm_eval_layer1 = BlockLSTM(
            root, Const<int64>(root, {TEST_TIME_LEN}),  // seq_len_max,
            x_eval, cs_prev_eval_layer1, h_prev_eval_layer1, w_layer1,
            Const<float>(
                    root, 0,
                    TensorShape({NUM_UNIT})),  // wci, used when use_peephole is true
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wcf
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wco
            b_layer1);
    auto block_lstm_eval_layer2 = BlockLSTM(
            root, Const<int64>(root, {TEST_TIME_LEN}),  // seq_len_max,
            block_lstm_eval_layer1.h, cs_prev_eval_layer2, h_prev_eval_layer2,
            w_layer2,
            Const<float>(
                    root, 0,
                    TensorShape({NUM_UNIT})),  // wci, used when use_peephole is true
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wcf
            Const<float>(root, 0, TensorShape({NUM_UNIT})),  // wco
            b_layer2);
    auto rnn_softmax_loss_eval =
            RNNSoftmaxLoss(root, block_lstm_eval_layer2.h, y_eval, w_y, b_y, cs);

    // Top 1
    auto topk_input =
            Placeholder(root, DT_FLOAT, Placeholder::Shape({INPUT_SIZE}));
    auto topk = TopK(root, topk_input, Cast(root, 1, DT_INT32));

    ClientSession session(root);

    // Initialize variables
    TF_CHECK_OK(session.Run({assign_w_layer1, assign_b_layer1, assign_w_layer2,
                             assign_b_layer2, assign_w_y, assign_b_y, assign_cs},
                            nullptr));
    TF_CHECK_OK(session.Run(
            {assign_ada_w_layer1, assign_ada_b_layer1, assign_ada_w_layer2,
             assign_ada_b_layer2, assign_ada_w_y, assign_ada_b_y},
            nullptr));

    // loop
    int step = 0;
    while (step < TRAINING_STEPS) {
        int content_index = 0;

        // Evaluate
        if (step % 100 == 0) {
            // Batch input with batch size of TEST_SEQ_LENGTH
            tensorflow::Tensor x_tensor(
                    tensorflow::DT_FLOAT,
                    tensorflow::TensorShape(
                            {TEST_TIME_LEN, TEST_BATCH_SIZE, INPUT_SIZE}));
            auto e_2d = x_tensor.shaped<float, 2>({TEST_SEQ_LENGTH, INPUT_SIZE});
            char test_str[TEST_SEQ_LENGTH + 1];
            test_str[TEST_SEQ_LENGTH] = '\0';
            std::copy_n(&test_content[content_index], TEST_SEQ_LENGTH, test_str);

            // Prepare y, y value make no sense in the evaluation process
            tensorflow::Tensor y_tensor(
                    tensorflow::DT_INT64,
                    tensorflow::TensorShape({TEST_TIME_LEN, TEST_BATCH_SIZE}));
            typename TTypes<int64>::Matrix y_t = y_tensor.matrix<int64>();
            y_t.setZero();

            tensorflow::Tensor eval_h_prev_layer1_tensor(
                    tensorflow::DT_FLOAT,
                    tensorflow::TensorShape({TEST_BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix eval_h_prev_layer1_t =
                    eval_h_prev_layer1_tensor.matrix<float>();
            eval_h_prev_layer1_t.setZero();

            Tensor eval_cs_prev_layer1_tensor(
                    DT_FLOAT, TensorShape({TEST_BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix eval_cs_prev_layer1_t =
                    eval_cs_prev_layer1_tensor.matrix<float>();
            eval_cs_prev_layer1_t.setZero();

            tensorflow::Tensor eval_h_prev_layer2_tensor(
                    tensorflow::DT_FLOAT,
                    tensorflow::TensorShape({TEST_BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix eval_h_prev_layer2_t =
                    eval_h_prev_layer2_tensor.matrix<float>();
            eval_h_prev_layer2_t.setZero();

            Tensor eval_cs_prev_layer2_tensor(
                    DT_FLOAT, TensorShape({TEST_BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix eval_cs_prev_layer2_t =
                    eval_cs_prev_layer2_tensor.matrix<float>();
            eval_cs_prev_layer2_t.setZero();

            for (int i = 0; i < 20; i++) {
                LOG(INFO) << __FUNCTION__
                          << "----------------Evaluate test_str: " << test_str;

                {
                    // batch one-hot processing
                    for (int j = 0; j < TEST_SEQ_LENGTH; j++) {
                        // Ref: tensor_test.cc

                        // Assign a 1 x INPUT_SIZE * 1 matrix (really vector) to a slice of
                        // size
                        Eigen::Tensor<float, 2, Eigen::RowMajor> m(1, INPUT_SIZE);
                        m.setZero();

                        // one-hot processing for one character
                        char test_char = test_str[j];
                        auto search = vocab_index_dic.find(test_char);
                        int vocab_index = search->second;
                        m(0, vocab_index) = 1.0f;

                        // set e_2d
                        Eigen::DSizes<Eigen::DenseIndex, 2> indices(j, 0);
                        Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, INPUT_SIZE);
                        e_2d.slice(indices, sizes) = m;
                    }
                }
                std::vector<tensorflow::Tensor> outputs;

                // Run
                TF_CHECK_OK(
                        session.Run({{x_eval, x_tensor},
                                     {y_eval, y_tensor},
                                     {h_prev_eval_layer1, eval_h_prev_layer1_tensor},
                                     {cs_prev_eval_layer1, eval_cs_prev_layer1_tensor},
                                     {h_prev_eval_layer2, eval_h_prev_layer2_tensor},
                                     {cs_prev_eval_layer2, eval_cs_prev_layer2_tensor}},
                                    {rnn_softmax_loss_eval.loss, rnn_softmax_loss_eval.p,
                                     block_lstm_eval_layer1.h, block_lstm_eval_layer1.cs,
                                     block_lstm_eval_layer2.h, block_lstm_eval_layer2.cs},
                                    {}, &outputs));
#ifdef VERBOSE
                LOG(INFO) << "Print rnn_softmax_loss_hgrad, step: " << step
                  << ", p: " << std::endl
                  << DetailedDebugString(outputs[1]);
#endif

                // update test_str for x_tensor
                for (int i = 0; i < TEST_BATCH_SIZE; i++) {
                    vector<Tensor> outputs_topk;
                    Tensor topk_input_tensor(DT_FLOAT, TensorShape({INPUT_SIZE}));
                    topk_input_tensor.flat<float>() =
                            outputs[1].SubSlice(0).SubSlice(i).unaligned_flat<float>();
                    TF_CHECK_OK(session.Run({{topk_input, topk_input_tensor}},
                                            {topk.values, topk.indices}, {},
                                            &outputs_topk));
#ifdef VERBOSE
                    LOG(INFO) << "Print topk, step: " << step
                    << ", debug: " << outputs_topk[0].DebugString() << ", "
                    << outputs_topk[1].DebugString();
#endif
                    int index = outputs_topk[1].scalar<int>()();
                    auto search = index_vocab_dic.find(index);
                    test_str[i] = search->second;
                }

                // update eval_h_prev_tensor, eval_cs_prev_tensor
                CHECK(eval_h_prev_layer1_tensor.CopyFrom(
                        outputs[2].Slice(0, 1),
                        {outputs[2].dim_size(1), outputs[2].dim_size(2)}));
                CHECK(eval_cs_prev_layer1_tensor.CopyFrom(
                        outputs[3].Slice(0, 1),
                        {outputs[3].dim_size(1), outputs[3].dim_size(2)}));
                CHECK(eval_h_prev_layer2_tensor.CopyFrom(
                        outputs[4].Slice(0, 1),
                        {outputs[4].dim_size(1), outputs[4].dim_size(2)}));
                CHECK(eval_cs_prev_layer2_tensor.CopyFrom(
                        outputs[5].Slice(0, 1),
                        {outputs[5].dim_size(1), outputs[5].dim_size(2)}));
            }
        }  // Evaluate

        // Train
        {
            // zeroed out when batch 0
            Tensor h_prev_layer1_tensor(DT_FLOAT,
                                        TensorShape({BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix h_prev_layer1_t =
                    h_prev_layer1_tensor.matrix<float>();
            h_prev_layer1_t.setZero();

            Tensor cs_prev_layer1_tensor(DT_FLOAT,
                                         TensorShape({BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix cs_prev_layer1_t =
                    cs_prev_layer1_tensor.matrix<float>();
            cs_prev_layer1_t.setZero();

            // Always zero
            Tensor cs_grad_layer1_tensor(
                    DT_FLOAT, TensorShape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float, 3>::Tensor cs_grad_layer1_t =
                    cs_grad_layer1_tensor.tensor<float, 3>();
            cs_grad_layer1_t.setZero();

            Tensor h_prev_layer2_tensor(DT_FLOAT,
                                        TensorShape({BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix h_prev_layer2_t =
                    h_prev_layer2_tensor.matrix<float>();
            h_prev_layer2_t.setZero();

            Tensor cs_prev_layer2_tensor(DT_FLOAT,
                                         TensorShape({BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float>::Matrix cs_prev_layer2_t =
                    cs_prev_layer2_tensor.matrix<float>();
            cs_prev_layer2_t.setZero();

            // Always zero
            Tensor cs_grad_layer2_tensor(
                    DT_FLOAT, TensorShape({TIME_LEN, BATCH_SIZE, NUM_UNIT}));
            typename TTypes<float, 3>::Tensor cs_grad_layer2_t =
                    cs_grad_layer2_tensor.tensor<float, 3>();
            cs_grad_layer2_t.setZero();

            // Note that every input batch in BATCH_SIZE is from a different example
            Tensor x_tensor(DT_FLOAT,
                            TensorShape({TIME_LEN, BATCH_SIZE, INPUT_SIZE}));
            {
                // batch one-hot processing
                auto e_2d = x_tensor.shaped<float, 2>({SEQ_LENGTH, INPUT_SIZE});
                int x_index = content_index;
                for (int i = 0; i < TIME_LEN; i++) {
                    // Ref: tensor_test.cc

                    // Assign a 1 x INPUT_SIZE * 1 matrix (really vector) to a slice of
                    // size
                    Eigen::Tensor<float, 2, Eigen::RowMajor> m(1, INPUT_SIZE);
                    m.setZero();

                    // one-hot processing for one character
                    char test_char = test_content[x_index++];
                    auto search = vocab_index_dic.find(test_char);
                    int vocab_index = search->second;
                    m(0, vocab_index) = 1.0f;

                    for (int b = 0; b < BATCH_SIZE; b++) {
                        // set e_2d
                        Eigen::DSizes<Eigen::DenseIndex, 2> indices(i * BATCH_SIZE + b, 0);
                        Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, INPUT_SIZE);
                        e_2d.slice(indices, sizes) = m;
                    }
                }
            }
#ifdef VERBOSE
            // LOG(INFO) << "-------------------------------x_tensor: " << std::endl
      // << DetailedDebugString(x_tensor);
#endif

            // y
            Tensor y_tensor(DT_INT64, TensorShape({TIME_LEN, BATCH_SIZE}));
            {
                typename TTypes<int64>::Vec y_t =
                        y_tensor.shaped<int64, 1>({SEQ_LENGTH});

                // Prepare y
                int y_index = content_index + 1;
                for (int i = 0; i < TIME_LEN; i++) {
                    char test_char = test_content[y_index++];
                    auto search = vocab_index_dic.find(test_char);
                    int vocab_index = search->second;

                    for (int b = 0; b < BATCH_SIZE; b++) {
                        y_t(i * BATCH_SIZE + b) = vocab_index;
                    }
                }
            }
#ifdef VERBOSE
            // LOG(INFO) << "-------------------------------y_tensor: " << std::endl
      // << DetailedDebugString(y_tensor);
#endif

            // Run
            vector<Tensor> outputs;
            TF_CHECK_OK(
                    session.Run({{x, x_tensor},
                                 {y, y_tensor},
                                 {h_prev_layer1, h_prev_layer1_tensor},
                                 {cs_prev_layer1, cs_prev_layer1_tensor},
                                 {cs_grad_layer1, cs_grad_layer1_tensor},
                                 {h_prev_layer2, h_prev_layer2_tensor},
                                 {cs_prev_layer2, cs_prev_layer2_tensor},
                                 {cs_grad_layer2, cs_grad_layer2_tensor}},
                                {rnn_softmax_loss.loss, apply_w_layer1, apply_b_layer1,
                                 apply_w_layer2, apply_b_layer2, apply_w_y, apply_b_y},
                                {}, &outputs));

            if (step % 100 == 0) {
#ifdef VERBOSE
                LOG(INFO) << "Print step: " << step << ", loss: " << std::endl
                  << DetailedDebugString(outputs[0]);
#endif
                Eigen::Tensor<float, 0, Eigen::RowMajor> total_loss =
                        outputs[0].flat<float>().sum();
                LOG(INFO) << "Print step: " << step << ", total_loss: " << total_loss();
            }
        }

        // update h_prev, cs_prev when time sequence > 0

        step++;
    }
}
#endif
