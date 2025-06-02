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
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/spin_mutex.h>
#include "model/SVRParameters.hpp"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "DQScalingFactorService.hpp"
#include "align_features.cuh"
#include "appcontext.hpp"
#include "kernel_factory.hpp"


namespace svr {
namespace datamodel {
// Stretch, shift, skip and transpose rows with indexes
arma::mat sst(const arma::mat &m, const t_feature_mechanics &fm, const arma::uvec &ixs)
{
    arma::mat v(m.n_cols, ixs.size(), ARMA_DEFAULT_FILL);
    tbb::parallel_for(tbb::blocked_range2d<uint32_t>(0, ixs.size(), 0, m.n_cols), [&](const auto &br) {
        for (auto r = br.rows().begin(); r < br.rows().end(); ++r)
            for (auto c = br.cols().begin(); c < br.cols().end(); ++c) {
                const auto i = ixs[r] - fm.shifts[c];
                const auto st = fm.stretches[c];
                // const auto sk = fm.skips[c];
                const auto i_s = STRETCHSKIP_(i);
                v(c, r) = m(i_s, c);
            }
    });

    LOG4_TRACE("Returning " << common::present(v) << " from " << common::present(m) << ", indexes " << common::present_chunk(ixs, .1) << " with features " << fm);
    return v;
}

arma::mat OnlineMIMOSVR::feature_chunk_t(const arma::uvec &ixs_i) const
{
    const auto &fm = front(param_set)->get_feature_mechanics();
    if (fm.needs_tuning())
        LOG4_THROW("Feature alignment parameters not present.");
    return sst(*p_features, fm, ixs_i);
}

arma::mat OnlineMIMOSVR::predict_chunk_t(const arma::mat &x_predict) const
{
    const auto &fm = front(param_set)->get_feature_mechanics();
    if (fm.needs_tuning())
        LOG4_WARN("Feature alignment parameters not present.");
    return sst(arma::join_cols(*p_features, x_predict), fm, arma::regspace<arma::uvec>(p_features->n_rows, p_features->n_rows + x_predict.n_rows - 1));
}

arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict, const bpt::ptime &time)
{
    // const auto dev_ct = 1; // common::gpu_handler<>::get().get_gpu_devices_count();
    CPTR(arma::mat) p_x_predict_t = projection ? &x_predict : new auto(predict_chunk_t(x_predict));
    if (is_manifold()) {
        const auto res = manifold_predict(*p_x_predict_t, time);
        if (!projection) delete p_x_predict_t;
        return res;
    }
    arma::mat prediction;
    tbb::mutex predict_l;
    const auto l_cols = p_labels->n_cols;
    const auto active_chunks = get_predict_chunks();
    const double chunk_divisor = 1. / active_chunks.size();
#pragma omp parallel ADJ_THREADS(ixs.size() * p_x_predict_t->n_cols * PROPS.get_weight_columns())
#pragma omp single
    {
        OMP_TASKLOOP_1()
        for (const auto chunk_ix: active_chunks) {
            const auto p_params = get_params_ptr(chunk_ix);
            if (!p_params) LOG4_THROW("No parameters for chunk " << chunk_ix);
            arma::mat scaled_x_predict_t = *p_x_predict_t;
            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
            business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, p_params->get_lag_count(), chunk_sf, scaled_x_predict_t);
            const arma::mat chunk_predict_K = kernel::IKernel<double>::get(*p_params)->kernel(ccache(), scaled_x_predict_t, train_feature_chunks_t[chunk_ix], time, last_trained_time);
            // TODO Fix order of scaled predict and train chunk after fixing the kernel output position
            assert(p_x_predict_t->n_cols == chunk_predict_K.n_rows);
            arma::mat multiplicated(chunk_predict_K.n_rows, l_cols, arma::fill::zeros);
            tbb::mutex add_l;
            const uint32_t w_cols = weight_chunks[chunk_ix].n_cols / l_cols;
            OMP_TASKLOOP_(chunk_predict_K.n_rows,)
            for (uint32_t predict_row = 0; predict_row < chunk_predict_K.n_rows; ++predict_row) {
                arma::mat K_row_t = chunk_predict_K.row(predict_row).t();
                if (l_cols > 1) K_row_t = common::extrude_rows<double>(K_row_t, l_cols);
                OMP_TASKLOOP(w_cols)
                for (uint32_t weight_col = 0; weight_col < w_cols; ++weight_col) {
                    const auto start_col = weight_col * l_cols;
                    const double pred = common::mean(weight_chunks[chunk_ix].cols(start_col, start_col + l_cols - 1) % (K_row_t + train_label_chunks[chunk_ix]));
                    const tbb::mutex::scoped_lock lk_add(add_l);
                    multiplicated.row(predict_row) += pred;
                }
            }
            const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
            business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, multiplicated);
            multiplicated *= chunk_divisor * chunks_score[chunk_ix];
            LOG4_TRACE("Chunk " << chunk_ix << " predicted " << common::present(multiplicated) << " from " << common::present(chunk_predict_K) << " with " <<
                common::present(weight_chunks[chunk_ix]) << ", scaling factor " << *p_labels_sf << ", labels " << common::present(train_label_chunks[chunk_ix]) << ", features " <<
                common::present(train_feature_chunks_t[chunk_ix]) << ", chunk divisor " << chunk_divisor << ", time " << time << ", column " << p_params->get_input_queue_column_name() <<
                ", predict features " << common::present(scaled_x_predict_t));
            const tbb::mutex::scoped_lock lk_pred(predict_l);
            if (prediction.empty()) prediction = multiplicated; else prediction += multiplicated;
        }
    }
    if (!projection) delete p_x_predict_t;
    LOG4_TRACE("For " << time << ", predicted " << common::present(prediction));
    return prediction;
}

// TODO Review and test
t_gradient_data OnlineMIMOSVR::produce_residuals()
{
    if (ixs.size() < 2)
        LOG4_THROW("At least two chunks are needed to produce residuals.");

    arma::vec row_divisors(p_labels->n_rows);
    arma::mat residuals(arma::size(*p_labels));
    const arma::uvec all_rows = arma::regspace<arma::uvec>(0, p_labels->n_rows - 1);
    arma::uvec res_rows;
    tbb::mutex residuals_l;
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
                                   - kernel::IKernel<double>::get(*p_params)->kernel(train_feature_chunks_t[i], excluded_features_t) * weight_chunks[i];
        business::DQScalingFactorService::unscale_labels_I(*p_sf, this_residuals);
        const tbb::mutex::scoped_lock lk(residuals_l);
        residuals.rows(chunk_excluded_rows) += this_residuals;
        row_divisors.rows(chunk_excluded_rows) += 1;
        res_rows.insert_rows(res_rows.n_rows, chunk_excluded_rows);
    }
    row_divisors.rows(arma::find(row_divisors == 0)).ones(); //?
    res_rows = arma::sort(arma::unique(res_rows));
    LOG4_TRACE("Resultant rows " << common::present(res_rows) << ", residuals " << common::present(residuals));
    return {
        ptr<arma::mat>(p_features->rows(res_rows)),
        ptr<arma::mat>(residuals.rows(res_rows) / row_divisors.rows(res_rows)),
        ptr<arma::vec>(p_last_knowns->rows(res_rows))
    };
}
} // namespace datamodel
} // namespace svr
