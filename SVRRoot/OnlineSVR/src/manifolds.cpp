//
// Created by zarko on 11/11/21.
//

#include <cmath>
#include <cstddef>
#include <random>
#include <random>
#include <vector>
#include <xoshiro.h>
#include "common/compatibility.hpp"
#include "appcontext.hpp"

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

// #define RANDOMIZE_MANIFOLD_INDEXES // Worsens results

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
        LOG4_WARN("Manifold already initialized!");
        return;
    }
    assert(ixs.size() == 1 && train_feature_chunks_t.size() == 1 && train_label_chunks.size() == 1);
    const auto &features_t = train_feature_chunks_t.front();
    const auto &labels = train_label_chunks.front();
    const auto n_samples = features_t.n_cols;
    const auto n_samples_2 = n_samples * n_samples;
    const auto manifold_interleave = PROPS.get_interleave();
    const uint32_t n_manifold_samples = n_samples_2 / manifold_interleave;
    const auto n_manifold_features = features_t.n_rows * 2;

    auto p_manifold_parameters = otr(*p);
    switch (p->get_kernel_type()) {
        case datamodel::e_kernel_type::DEEP_PATH:
            p_manifold_parameters->set_kernel_type(datamodel::e_kernel_type::PATH);
            break;
        default:
            LOG4_THROW("Kernel type " << p->get_kernel_type() << " not handled.");
    }
    p_manifold_parameters->set_chunk_index(0);
    p_manifold_parameters->set_svr_kernel_param(0);
    p_manifold_parameters->set_svr_decremental_distance(n_manifold_samples);
    {
        auto &fm = p_manifold_parameters->get_feature_mechanics();
        fm.stretches = arma::fvec(n_manifold_features, arma::fill::ones);
        fm.shifts = arma::u32_vec(n_manifold_features, arma::fill::zeros);
        // fm.skips = arma::join_cols(fm.skips, fm.skips);
    }
    p_manifold = otr<OnlineMIMOSVR>(0, model_id, t_param_set{p_manifold_parameters}, p_dataset);
    p_manifold->projection += 1;
    auto p_manifold_features = ptr<arma::mat>(n_manifold_samples, n_manifold_features, arma::fill::none);
    auto p_manifold_labels = ptr<arma::mat>(n_manifold_samples, labels.n_cols, arma::fill::none);
    auto p_manifold_lastknowns = ptr<arma::vec>(n_manifold_samples, arma::fill::none);
    auto p_manifold_weights = ptr<arma::vec>(n_manifold_samples, arma::fill::none);
    /* L diff matrix format:
     * L0 - L0, L0 - L1, L0 - L2, ..., L0 - Lm
     * L1 - L0, L1 - L1, L1 - L2, ..., L1 - Lm
     * ...
     * Ln - L0, Ln - L1, Ln - L2, ..., Ln - Lm
     */
#pragma omp parallel ADJ_THREADS(n_samples_2 * labels.n_cols)
#pragma omp single
    {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
        std::uniform_int_distribution<uint32_t> rand_int(0, manifold_interleave - 1);
        // auto gen = common::reproducibly_seeded_64<xso::rng64>();
        auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(manifold_interleave));
        tbb::mutex r_lock;
#endif
        OMP_TASKLOOP_(n_samples, firstprivate(manifold_interleave, n_samples))
        for (uint32_t i = 0; i < n_samples; ++i) {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
            tbb::mutex::scoped_lock lk(r_lock);
            const auto rand_int_g = rand_int(gen); // For some odd reason this freezes on operator() if inside loop below
            lk.release();
            const auto r_i = common::bounce<uint32_t>(i + rand_int_g, n_samples - 1);
#endif
            for (uint32_t j = 0; j < n_samples; ++j) {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
                const auto r_j = common::bounce<uint32_t>(j + rand_int_g, n_samples - 1);
#else
                const auto r_i = i; const auto r_j = j;
#endif
                OMP_TASKLOOP(labels.n_cols)
                for (uint32_t k = 0; k < labels.n_cols; ++k) {
                    const auto r = i + j * n_samples; // Column by column ordering
                    if (r % manifold_interleave == 0) {
                        const auto r_out = r / manifold_interleave;
                        if (r_out < n_manifold_samples) {
                            p_manifold_labels->at(r_out, k) = labels(r_j, k) - labels(r_i, k);
                            p_manifold_features->row(r_out) = arma::join_rows(features_t.col(r_j).t(), features_t.col(r_i).t());
                        }
                    }
                }
            }
        }
    }

    LOG4_DEBUG("Generated " << arma::size(*p_manifold_labels) << " manifold label matrix and " << arma::size(*p_manifold_features) <<
                            " manifold feature matrix from " << arma::size(features_t) << " feature matrix and " << arma::size(labels) << " label matrix.");
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, p_manifold_lastknowns, p_manifold_weights, last_value_time);
}

datamodel::SVRParameters_ptr OnlineMIMOSVR::is_manifold() const
{
    return business::SVRParametersService::is_manifold(param_set);
}

arma::mat OnlineMIMOSVR::manifold_predict(const arma::mat &x_predict, const boost::posix_time::ptime &time) const
{
    assert(ixs.size() == 1 && train_feature_chunks_t.size() == 1 && train_label_chunks.size() == 1);

    auto x_predict_t = predict_chunk_t(x_predict);
    constexpr uint16_t chunk_ix = 0;
    const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
    business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, get_params(chunk_ix).get_lag_count(), chunk_sf, x_predict_t);
    arma::mat result(x_predict_t.n_cols, p_labels->n_cols);
    const arma::mat &features_t = train_feature_chunks_t.front();
    const auto manifold_interleave = PROPS.get_predict_ileave();
    const arma::SizeMat predict_size(x_predict_t.n_rows + features_t.n_rows, features_t.n_cols / manifold_interleave);
    arma::uvec predict_ixs(predict_size.n_cols);

#ifdef RANDOMIZE_MANIFOLD_INDEXES
    std::uniform_int_distribution<uint32_t> rand_int(0, manifold_interleave - 1);
    // auto gen = common::reproducibly_seeded_64<xso::rng64>(); // TODO Freezes on operator() fix
    auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(manifold_interleave));
    tbb::mutex r_lock;
#endif

#pragma omp parallel ADJ_THREADS(x_predict_t.n_cols * predict_size.n_cols)
#pragma omp single
    {
        OMP_TASKLOOP_(features_t.n_cols, firstprivate(manifold_interleave))
        for (uint32_t i_x = 0; i_x < features_t.n_cols; ++i_x)
            if (i_x % manifold_interleave) {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
                tbb::mutex::scoped_lock lk(r_lock);
                const auto rand_int_g = rand_int(gen);
                lk.release();
                predict_ixs[i_x / manifold_interleave] = common::bounce<uint32_t>(i_x + rand_int_g, features_t.n_cols - 1);
#else
                predict_ixs[i_x / manifold_interleave] = i_x;
#endif
            }
        const auto predict_labels = train_label_chunks.front().rows(predict_ixs);
        const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
        OMP_TASKLOOP_1()
        for (uint32_t i_p = 0; i_p < x_predict_t.n_cols; ++i_p) {
            common::memory_manager::get().barrier();
            arma::mat pred_features_t(predict_size, arma::fill::none);
            OMP_TASKLOOP_(pred_features_t.n_cols, firstprivate(i_p))
            for (uint32_t i_r = 0; i_r < pred_features_t.n_cols; ++i_r)
                pred_features_t.col(i_r) = arma::join_cols(x_predict_t.col(i_p), features_t.col(predict_ixs[i_r]));
            result[i_p] = arma::mean(arma::vectorise(predict_labels + p_manifold->predict(pred_features_t, bpt::not_a_date_time))); // TODO Figure out multioutput
            business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, result[i_p]);
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
