//
// Created by zarko on 11/11/21.
//

#include "common/compatibility.hpp"
#include <armadillo>
#include <cmath>
#include <cstddef>
#include <random>
#include <random>
#include <vector>
#include <xoshiro.h>

#ifdef EXPERIMENTAL_FEATURES
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#pragma GCC diagnostic pop
#endif


#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "SVRParametersService.hpp"
#include "DQScalingFactorService.hpp"


namespace svr {
namespace datamodel {


#if 0 // Matlab code to generate ideal kernel matrix prototype
std::scoped_lock lg (mlab_mx);
    LOG4_BEGIN();

    const auto labels_sum = arma::sum(arma::vectorise(labels));
    const auto cached_labels_iter = cached_labels.find(labels_sum);
    if (cached_labels_iter != cached_labels.end()) {
        LOG4_DEBUG("Returning cached labels for sum " << labels_sum);
        manifold_labels = cached_labels_iter->second;
        return;
    }

    if (labels.n_cols != 1) LOG4_THROW("Labels dimensions are incorrect " << arma::size(labels));

    std::string cmd = "syms ";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < i; ++j) {
            cmd += "k" + std::to_string(i) + "_" + std::to_string(j) + " ";
        }
    }
    cmd += ";\n";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "l" + std::to_string(i) + "=" + common::to_string_with_precision(labels[i]) + ";";
    }
    cmd += "M=[";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < labels.n_elem; ++j) {
            if (i == j)
                cmd += "1";
            else if (i < j)
                cmd += "k" + std::to_string(j) +  "_" + std::to_string(i);
            else
                cmd += "k" + std::to_string(i) +  "_" + std::to_string(j);
            if (j < labels.n_elem - 1)
                cmd += ",";
        }
        if (i < labels.n_elem - 1)
            cmd += ";";
    }
    cmd += "];\nW = [";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "1";
        if (i < labels.n_elem - 1) cmd += ";";
    }
    cmd += "];\nL=[";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        cmd += "l" + std::to_string(i);
        if (i < labels.n_elem - 1) cmd += ";";
    }
    cmd += "];\nres=solve(M*W-L,";
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j < i; ++j) {
            cmd += "k" + std::to_string(i) + "_" + std::to_string(j);
            if (i == labels.n_elem - 1 and j == labels.n_elem - 2) continue;
            cmd += ",";
        }
    }
    cmd += ");";
    //LOG4_DEBUG("Matlab routine is:\n"+cmd);

    static std::unique_ptr<matlab::engine::MATLABEngine> p_matlab = matlab::engine::startMATLAB();
    static matlab::data::ArrayFactory factory;

    std::u16string u16cmd;
    common::from_utf8(cmd, u16cmd);
    LOG4_FILE("/tmp/manifold_cmd.txt", ""<<cmd);
    p_matlab->eval(u16cmd);
    matlab::data::StructArray res = p_matlab->getVariable(u"res");
    const matlab::data::ArrayDimensions dims = res.getDimensions();
    LOG4_DEBUG("Result dimensions " << dims[0] << "x" << dims[1]);

    // Get number of fields
    const size_t numFields = res.getNumberOfFields();
    LOG4_DEBUG("Result has " << numFields << " fields");
    if (numFields != (labels.n_elem * labels.n_elem) / 2 - labels.n_elem / 2)
        LOG4_THROW("Wrong number of fields returned " << numFields);
/*
    // Get the struct array fieldnames
    auto fields = res.getFieldNames();
    std::deque<matlab::data::MATLABFieldIdentifier> field_names;
    for (const auto& field_name : fields) { field_names.push_back(field_name);
        LOG4_DEBUG("Result got field " << std::string(name));
 */
    size_t output_ix = 0;
    manifold_labels.set_size(labels.n_elem * labels.n_elem / 2 + labels.n_elem / 2, 1);
    for (size_t i = 0; i < labels.n_elem; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                manifold_labels(output_ix++, 0) = 1;
                continue;
            }
            const std::string field_name = "k" + std::to_string(i) + "_" + std::to_string(j);
            std::u16string u16str;
            common::from_utf8(field_name, u16str);
            p_matlab->eval(u"tmp = double(res." + u16str + u");");
            matlab::data::TypedArray<double> field = p_matlab->getVariable(u"tmp");
            //LOG4_TRACE("Field " << field_name << " type " << int(field.getType()) << ", size " << field.getDimensions()[0] << "x" << field.getDimensions()[1] << ", elements " << field.getNumberOfElements());
            if (field.getDimensions()[0] < 1 or field.getDimensions()[1] < 1)
                LOG4_THROW("Empty symbol returned for " << field_name);
            manifold_labels(output_ix++, 0) = field[0][0];
            /*
                    matlab::data::ArrayDimensions data_dims = field.getDimensions();
                    for (size_t j=0; j<data_dims[0]; ++j){
                        for (size_t k=0; k<data_dims[1]; ++k){
                            std::cout << j << ", " << k << ": " << field[j][k] << " ";
                        }
                        std::cout << std::endl;
                    }
            */
        }
    }
    p_matlab->evalAsync(u"clear;");
    cached_labels[labels_sum] = manifold_labels;
    LOG4_END();
#endif /* EXPERIMENTAL_FEATURES */

void
OnlineMIMOSVR::init_manifold(const datamodel::SVRParameters_ptr &p, const bpt::ptime &last_value_time)
{
    if (p_manifold) {
        LOG4_ERROR("Manifold already initialized!");
        return;
    }

    auto p_manifold_parameters = otr(*p);
    switch (p->get_kernel_type()) {
        case datamodel::e_kernel_type::DEEP_PATH:
            p_manifold_parameters->set_kernel_type(datamodel::e_kernel_type::PATH);
            break;
        default:
            LOG4_ERROR("Kernel type " << int(p->get_kernel_type()) << " not handled.");
            return;
    }
    const uint32_t n_rows = p_features->n_rows;
    const auto n_learning_rows = std::min<uint32_t>(n_rows, p->get_svr_decremental_distance());
    const auto n_manifold_samples = n_rows * n_rows / C_manifold_interleave;
    p_manifold_parameters->set_svr_decremental_distance(n_learning_rows * n_learning_rows / C_manifold_interleave);
    p_manifold_parameters->set_chunk_index(0);
    {
        auto &fm = p_manifold_parameters->get_feature_mechanics();
        fm.stretches.insert_cols(fm.stretches.n_cols, fm.stretches);
        fm.shifts.insert_cols(fm.shifts.n_cols, fm.shifts);
        // fm.skips.insert_cols(fm.skips.n_cols, fm.skips);
    }
    p_manifold = otr<OnlineMIMOSVR>(0, model_id, t_param_set{p_manifold_parameters}, p_dataset);
    p_manifold->projection = projection + 1;

    auto p_manifold_features = ptr<arma::mat>(n_manifold_samples, p_features->n_cols * 2, arma::fill::none);
    auto p_manifold_labels = ptr<arma::mat>(p_manifold_features->n_rows, p_labels->n_cols, arma::fill::none);
    auto p_manifold_lastknowns = ptr<arma::vec>(p_manifold_features->n_rows, arma::fill::none);
    auto p_manifold_weights = ptr<arma::mat>(arma::size(*p_manifold_labels), arma::fill::none);

    arma::cube L_diff(n_rows, n_rows, p_labels->n_cols), L_weights(n_rows, n_rows, p_labels->n_cols);
    {
        /* L diff matrix:
         * L0 - L0, L0 - L1, L0 - L2, ..., L0 - Lm
         * L1 - L0, L1 - L1, L1 - L2, ..., L1 - Lm
         * ...
         * Ln - L0, Ln - L1, Ln - L2, ..., Ln - Lm
         */
        const arma::mat L_t = p_labels->t();
        const arma::mat weights_t = p_input_weights->t();
        OMP_FOR_(n_rows * p_labels->n_cols, SSIMD collapse(2))
        for (uint32_t r = 0; r < n_rows; ++r)
            for (uint32_t c = 0; c < p_labels->n_cols; ++c) {
                L_diff.slice(c).row(r) = p_labels->at(r, c) - L_t.row(c); // Asymmetric distances cube
                L_weights.slice(c).row(r) = p_input_weights->at(r, c) - weights_t.row(c); // Weights cube TODO Verify
            }
    }
    common::threadsafe_uniform_int_distribution<uint32_t> rand_int(0, C_manifold_interleave - 1);
    auto gen = common::reproducibly_seeded_64<xso::rng64>();
    t_omp_lock lk;
    OMP_FOR_(n_rows * n_rows * p_labels->n_cols, SSIMD collapse(3))
    for (uint32_t i = 0; i < n_rows; ++i)
        for (uint32_t j = 0; j < n_rows; ++j)
            for (uint32_t k = 0; k < p_labels->n_cols; ++k) {
                const auto r = i + j * n_rows; // Column by column ordering
                if (r % C_manifold_interleave == 0) {
                    const auto r_out = r / C_manifold_interleave;
                    if (r_out < n_manifold_samples) {
                        const auto rand_int_g = rand_int(gen);
                        const auto r_i = common::bounce<uint32_t>(i + rand_int_g, n_rows - 1);
                        const auto r_j = common::bounce<uint32_t>(j + rand_int_g, n_rows - 1);
                        p_manifold_labels->at(r_out, k) = L_diff(r_i, r_j, k);
                        p_manifold_features->row(r_out) = arma::join_rows(p_features->row(r_i), p_features->row(r_j));
                    }
                }
            }

    L_diff.clear();
    LOG4_DEBUG("Generated " << arma::size(*p_manifold_labels) << " manifold label matrix and " << arma::size(*p_manifold_features) <<
                " manifold feature matrix from " << arma::size(*p_features) << " feature matrix and " << arma::size(*p_labels) << " label matrix.");
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, p_manifold_lastknowns, p_manifold_weights, last_value_time);
}

datamodel::SVRParameters_ptr OnlineMIMOSVR::is_manifold() const
{
    return business::SVRParametersService::is_manifold(param_set);
}

arma::mat OnlineMIMOSVR::manifold_predict(const arma::mat &x_predict) const
{
    arma::mat result(x_predict.n_rows, p_labels->n_cols);
    arma::SizeMat predict_size(p_features->n_rows, x_predict.n_cols + p_features->n_cols);
#pragma omp parallel ADJ_THREADS(x_predict.n_rows * p_features->n_rows)
#pragma omp single
    {
        OMP_TASKLOOP_(x_predict.n_rows, SSIMD)
        for (uint32_t i_p = 0; i_p < x_predict.n_rows; ++i_p) {
            common::memory_manager::get().barrier();
            arma::mat predict_features(predict_size);
            OMP_TASKLOOP_(p_features->n_rows, SSIMD)
            for (uint32_t i_x = 0; i_x < p_features->n_rows; ++i_x)
                predict_features.row(i_x) = arma::join_rows(x_predict.row(i_p), p_features->row(i_x));
            result.row(i_p) = arma::mean(*p_labels + p_manifold->predict(predict_features));
        }
    }
    return result;
}

OnlineMIMOSVR_ptr OnlineMIMOSVR::get_manifold()
{
    return p_manifold;
}

} // datamodel
} // namespace svr
