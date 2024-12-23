//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <execution>
#include <mutex>
#include <cmath>
#include <vector>
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "cuqrsolve.cuh"
#include "DQScalingFactorService.hpp"
#include "align_features.cuh"

namespace svr {
namespace datamodel {

// Stretch, shift, skip and transpose rows with indexes
arma::mat sst(const arma::mat &m, const t_feature_mechanics &fm, const arma::uvec &ixs)
{
    arma::mat v(m.n_cols, ixs.size(), arma::fill::none);
    OMP_FOR_(m.n_elem, simd collapse(2))
    for (unsigned c = 0; c < m.n_cols; ++c)
        for (unsigned r = 0; r < ixs.size(); ++r) {
            const auto i = ixs[r] - fm.shifts[c];
            const auto st = fm.stretches[c];
            // const auto sk = fm.skips[c];
            const auto i_s = STRETCHSKIP_(i);
            v(c, r) = m(i_s, c);
        }
    LOG4_TRACE("Returning " << common::present(v) << " from " << common::present(m) << ", indexes " << common::present_chunk(ixs, .1) << " with features " << fm);
#if 0
    static unsigned file_ct = 0;
    v.save(common::formatter() << "sst_v.csv" << file_ct, arma::csv_ascii);
    m.save(common::formatter() << "sst_m.csv" << file_ct, arma::csv_ascii);
    ++file_ct;
#endif
    return v;
}

arma::mat OnlineMIMOSVR::feature_chunk_t(const arma::uvec &ixs_i)
{
    const auto &fm = front(param_set)->get_feature_mechanics();
    if (fm.needs_tuning()) {
        LOG4_WARN("Feature alignment parameters not present.");
        return p_features->rows(ixs_i).t();
    }
    return sst(*p_features, fm, ixs_i);
}

arma::mat OnlineMIMOSVR::predict_chunk_t(const arma::mat &x_predict)
{
    const auto &fm = front(param_set)->get_feature_mechanics();
    if (fm.needs_tuning()) {
        LOG4_WARN("Feature alignment parameters not present.");
        return x_predict.t();
    }
    return sst(arma::join_cols(*p_features, x_predict), fm, arma::regspace<arma::uvec>(p_features->n_rows, p_features->n_rows + x_predict.n_rows - 1));
}

arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict, const bpt::ptime &time)
{
    if (is_manifold()) return manifold_predict(x_predict);
    const auto x_predict_t = predict_chunk_t(x_predict);
    arma::mat prediction;
    t_omp_lock predict_l;
#pragma omp parallel ADJ_THREADS(ixs.size() * x_predict.n_cols * weight_chunks.front().n_cols)
#pragma omp single
    {
        OMP_TASKLOOP_1()
        for (size_t chunk_ix = 0; chunk_ix < ixs.size(); ++chunk_ix) {
            const auto p_params = get_params_ptr(chunk_ix);
            arma::mat scaled_x_predict_t = x_predict_t;
            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
            business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, p_params->get_lag_count(), chunk_sf, scaled_x_predict_t);
#ifdef SINGLE_CHUNK_LEVEL
            const auto chunk_predict_K = prepare_Ky(*p_params, train_feature_chunks_t[chunk_ix], scaled_x_predict_t, time, last_trained_time,
                                                    common::gpu_handler<1>::get().get_gpu_devices_count());
#else
            const auto chunk_predict_K = time == bpt::not_a_date_time ?
                                         prepare_Ky(*p_params, train_feature_chunks_t[chunk_ix], scaled_x_predict_t) :
                                         prepare_Ky(ccache(), *p_params, train_feature_chunks_t[chunk_ix], scaled_x_predict_t, time, last_trained_time);
#endif
            LOG4_TRACE(
                    "Predicting " << arma::size(x_predict) << ", indexes " << common::present_chunk(ixs[chunk_ix], C_chunk_header) << ", size " << arma::size(ixs[chunk_ix])
                                  << ", parameters " << *p_params << ", K " << common::present(*chunk_predict_K) << ", w " << common::present(weight_chunks[chunk_ix]));
            arma::mat multiplicated(chunk_predict_K->n_rows, weight_chunks[chunk_ix].n_cols);
            OMP_TASKLOOP_(chunk_predict_K->n_rows * weight_chunks[chunk_ix].n_cols, collapse(2))
            for (size_t r = 0; r < chunk_predict_K->n_rows; ++r)
                for (size_t c = 0; c < weight_chunks[chunk_ix].n_cols; ++c)
                    multiplicated(r, c) = arma::as_scalar(chunk_predict_K->row(r) * weight_chunks[chunk_ix].col(c));// - p_params->get_svr_epsilon();
            business::DQScalingFactorService::unscale_labels(*business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true),
                                                             multiplicated);
            predict_l.set();
            if (prediction.empty())
                prediction = multiplicated;
            else
                prediction += multiplicated;
            predict_l.unset();
        }
    }
    return prediction / ixs.size();
}

// TODO Review and test
t_gradient_data OnlineMIMOSVR::produce_residuals()
{
    if (ixs.size() < 2) LOG4_THROW("At least two chunks are needed to produce residuals.");

    arma::vec row_divisors(p_labels->n_rows);
    arma::mat residuals(arma::size(*p_labels));
    const arma::uvec all_rows = arma::regspace<arma::uvec>(0, p_labels->n_rows - 1);
    arma::uvec res_rows;
    t_omp_lock residuals_l;
    OMP_FOR_i(ixs.size()) {
        arma::uvec chunk_excluded_rows = all_rows;
        chunk_excluded_rows.shed_rows(ixs[i]);
        const auto p_params = get_params_ptr(i);
        arma::mat excluded_features_t = p_features->rows(chunk_excluded_rows).t();
        arma::mat excluded_labels = p_labels->rows(chunk_excluded_rows);
        const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, i, gradient, step);
        business::DQScalingFactorService::scale_features(i, gradient, step, p_params->get_lag_count(), chunk_sf, excluded_features_t);
        const auto p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, i, gradient, step, level, false, true);
        business::DQScalingFactorService::scale_labels(*p_sf, excluded_labels);
        arma::mat this_residuals = excluded_labels + p_params->get_svr_epsilon()
                                   - *prepare_Ky(*p_params, train_feature_chunks_t[i], excluded_features_t) * weight_chunks[i];
        business::DQScalingFactorService::unscale_labels(*p_sf, this_residuals);
        residuals_l.set();
        residuals.rows(chunk_excluded_rows) += this_residuals;
        row_divisors.rows(chunk_excluded_rows) += 1;
        res_rows.insert_rows(res_rows.n_rows, chunk_excluded_rows);
        residuals_l.unset();
    }
    row_divisors.rows(arma::find(row_divisors == 0)).ones(); //?
    res_rows = arma::sort(arma::unique(res_rows));
    LOG4_TRACE("Resultant rows " << common::present(res_rows) << ", residuals " << common::present(residuals));
    return {ptr<arma::mat>(p_features->rows(res_rows)),
            ptr<arma::mat>(residuals.rows(res_rows) / row_divisors.rows(res_rows)),
            ptr<arma::vec>(p_last_knowns->rows(res_rows))};
}

} // namespace datamodel
} // namespace svr