//
// Created by zarko on 11/11/21.
//

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <xoshiro.h>

#ifdef EXPERIMENTAL_FEATURES
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <MatlabEngine.hpp>
#include <MatlabDataArray.hpp>
#pragma GCC diagnostic pop
#endif

#include <oneapi/tbb/spin_mutex.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/blocked_range3d.h>
#include "common/compatibility.hpp"
#include "appcontext.hpp"
#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "SVRParametersService.hpp"
#include "DQScalingFactorService.hpp"

// #define RANDOMIZE_MANIFOLD_INDEXES // TODO Test, seems like it worsens results

namespace svr {
namespace datamodel {

/* L diff matrix format:
 * L0 - L0, L0 - L1, L0 - L2, ..., L0 - Lm
 * L1 - L0, L1 - L1, L1 - L2, ..., L1 - Lm
 * ...
 * Ln - L0, Ln - L1, Ln - L2, ..., Ln - Lm
 */

void
OnlineMIMOSVR::init_manifold(const datamodel::SVRParameters_ptr &p, const bpt::ptime &last_value_time)
{
    if (p_manifold) {
        LOG4_WARN("Manifold already initialized!");
        return;
    }

    assert(ixs.size() == 1);
    assert(train_feature_chunks_t.size() == 1);
    assert(train_label_chunks.size() == 1);

    const auto &features_t = train_feature_chunks_t.front();
    const auto &labels = train_label_chunks.front();
    const auto n_samples = features_t.n_cols;
    const auto n_samples_2 = n_samples * n_samples;
    const auto manifold_interleave = PROPS.get_interleave();
    const uint32_t n_manifold_samples = CDIVI(n_samples_2, manifold_interleave);
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
    p_manifold_parameters->set_svr_decremental_distance(n_manifold_samples); {
        auto &fm = p_manifold_parameters->get_feature_mechanics();
        fm.stretches = arma::fvec(n_manifold_features, arma::fill::ones);
        fm.shifts = arma::u32_vec(n_manifold_features, arma::fill::zeros);
        // fm.skips = arma::join_cols(fm.skips, fm.skips);
    }
    p_manifold = otr<OnlineMIMOSVR>(0, model_id, t_param_set{p_manifold_parameters}, p_dataset);
    p_manifold->projection += 1;
    const auto p_manifold_features = ptr<arma::mat>(n_manifold_samples, n_manifold_features, ARMA_DEFAULT_FILL);
    const auto p_manifold_labels = ptr<arma::mat>(n_manifold_samples, labels.n_cols, ARMA_DEFAULT_FILL);
    const auto p_manifold_lastknowns = ptr<arma::vec>(n_manifold_samples, arma::fill::zeros);
    const auto p_manifold_weights = ptr<arma::vec>(n_manifold_samples, arma::fill::ones); // TODO Implement
#pragma omp parallel ADJ_THREADS(n_samples_2 * labels.n_cols)
#pragma omp single
    {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
    common::threadsafe_uniform_int_distribution<uint32_t> rand_int(0, manifold_interleave - 1);
    // auto gen = common::reproducibly_seeded_64<xso::rng64>(); // TODO Freezes in multithreaded environment
    auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(manifold_interleave));
#endif
    tbb::mutex sm;
    OMP_TASKLOOP_(n_samples, firstprivate(manifold_interleave, n_samples) collapse(2))
    for (uint32_t r = 0; r < n_samples; ++r) {
        for (uint32_t c = 0; c < n_samples; ++c) {
#ifdef RANDOMIZE_MANIFOLD_INDEXES
                const auto rand_int_g = rand_int(gen); // For some odd reason this freezes on operator() if inside loop below
                const auto rr = common::bounce<uint32_t>(r + rand_int_g, n_samples - 1);
                const auto cc = common::bounce<uint32_t>(c + rand_int_g, n_samples - 1);
#else
#define rr r
#define cc c
#endif
                for (uint32_t k = 0; k < labels.n_cols; ++k) {
                    const auto row_out = r + c * n_samples; // Column by column ordering
                    if (row_out % manifold_interleave) continue;
                    const auto row_out_il = row_out / manifold_interleave;
                    if (row_out_il >= n_manifold_samples) continue;
                    const auto L_out = labels(cc, k) - labels(rr, k);
                    const arma::rowvec F_out = arma::join_rows(features_t.col(cc).t(), features_t.col(rr).t());
                    const tbb::mutex::scoped_lock lk(sm);
                    p_manifold_labels->at(row_out_il, k) = L_out;
                    p_manifold_features->row(row_out_il) = F_out;
                }
            }
        }
    }

    LOG4_DEBUG("Generated " << common::present(*p_manifold_labels) << " manifold label matrix and " << common::present(*p_manifold_features) <<
        " manifold feature matrix from " << common::present(features_t) << " feature matrix and " << common::present(labels) << " label matrix.");
    p_manifold->batch_train(p_manifold_features, p_manifold_labels, p_manifold_lastknowns, p_manifold_weights, last_value_time);

    assert(p_manifold_features->has_nonfinite() == false && p_manifold_labels->has_nonfinite() == false);
    assert(p_manifold_lastknowns->has_nonfinite() == false && p_manifold_weights->has_nonfinite() == false);
}

datamodel::SVRParameters_ptr OnlineMIMOSVR::is_manifold() const
{
    return business::SVRParametersService::is_manifold(param_set);
}

arma::mat OnlineMIMOSVR::manifold_predict(arma::mat x_predict_t, const boost::posix_time::ptime &time) const
{
    assert(ixs.size() == 1 && train_feature_chunks_t.size() == 1 && train_label_chunks.size() == 1);

    constexpr uint16_t chunk_ix = 0;
    const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
    business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, get_params(chunk_ix).get_lag_count(), chunk_sf, x_predict_t);
    arma::mat result(x_predict_t.n_cols, p_labels->n_cols);
    const arma::mat &features_t = train_feature_chunks_t.front();
    const auto manifold_interleave = PROPS.get_predict_ileave();
    const arma::SizeMat predict_size(x_predict_t.n_rows + features_t.n_rows, CDIVI(features_t.n_cols, manifold_interleave));
    arma::uvec predict_ixs(predict_size.n_cols);
#ifdef RANDOMIZE_MANIFOLD_INDEXES
    common::threadsafe_uniform_int_distribution<uint32_t> rand_int(0, manifold_interleave - 1);
    // auto gen = common::reproducibly_seeded_64<xso::rng64>(); // TODO Freezes on operator() fix
    auto gen = std::mt19937_64((uint_fast64_t) common::pseudo_random_dev::max(manifold_interleave));
    tbb_pfor_i__(0, predict_size.n_cols,
        const auto rand_int_g = rand_int(gen);
        predict_ixs[i] = common::bounce<uint32_t>(i * manifold_interleave + rand_int_g, features_t.n_cols - 1);
    )
#else
    for (uint32_t i = 0; i < predict_size.n_cols; ++i) predict_ixs[i] = i * manifold_interleave;
#endif
    const auto predict_labels = train_label_chunks.front().rows(predict_ixs);
    const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
#pragma omp parallel ADJ_THREADS(std::max<int32_t>(1, common::gpu_handler<4>::get().get_max_gpu_threads() - int32_t(p_manifold->get_num_chunks())))
#pragma omp single
    {
        OMP_TASKLOOP_(x_predict_t.n_cols,)
        for (uint32_t i_p = 0; i_p < x_predict_t.n_cols; ++i_p) {
            arma::mat pred_features_t(predict_size, ARMA_DEFAULT_FILL);
            OMP_TASKLOOP_(pred_features_t.n_cols, firstprivate(i_p))
            for (uint32_t i_r = 0; i_r < pred_features_t.n_cols; ++i_r)
                pred_features_t.col(i_r) = arma::join_cols(x_predict_t.col(i_p), features_t.col(predict_ixs[i_r]));
            result[i_p] = arma::mean(arma::vectorise(predict_labels + p_manifold->predict(pred_features_t, bpt::not_a_date_time)));
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
