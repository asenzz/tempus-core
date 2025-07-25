//
// Created by zarko on 11/11/21.
//

#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <execution>
#include <mutex>
#include <cmath>
#include <vector>
#include <mpi.h>
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

// Stretch, shift and transpose and transpose rows with indexes
arma::mat OnlineSVR::sst(const arma::mat &m, const t_feature_mechanics &fm, const arma::uvec &ixs)
{
    LOG4_BEGIN();
    arma::mat v(m.n_cols, ixs.size(), ARMA_DEFAULT_FILL);
    OMP_FOR_(m.n_elem, SSIMD collapse(2))
    for (uint32_t c = 0; c < m.n_cols; ++c)
        for (uint32_t r = 0; r < ixs.size(); ++r) {
            const auto i = ixs[r] - fm.shifts[c];
            const auto st = fm.stretches[c];
            // const auto sk = fm.skips[c];
            const auto i_s = STRETCHSKIP_(i);
            v(c, r) = m(i_s, c);
        }

    LOG4_TRACE("Returning " << common::present(v) << " from " << common::present(m) << ", indexes " << common::present_chunk(ixs, .1) << " with features " << fm);
    return v;
}

arma::mat OnlineSVR::feature_chunk_t(const arma::uvec &ixs_i) const
{
    LOG4_BEGIN();
    const auto &fm = front(param_set)->get_feature_mechanics();
    assert(fm.needs_tuning() == false);
    return sst(*p_features, fm, ixs_i);
}

arma::mat OnlineSVR::predict_chunk_t(const arma::mat &x_predict) const
{
    LOG4_BEGIN();
    const auto &fm = front(param_set)->get_feature_mechanics();
    assert(fm.needs_tuning() == false);
    return sst(arma::join_cols(*p_features, x_predict), fm, arma::regspace<arma::uvec>(p_features->n_rows, p_features->n_rows + x_predict.n_rows - 1));
}

arma::mat OnlineSVR::predict(const arma::mat &x_predict, const bpt::ptime &time)
{
    LOG4_BEGIN();
    const auto x_predict_t = predict_chunk_t(x_predict);
    arma::mat prediction;
    t_omp_lock l2;
    const auto l_cols = p_labels->n_cols;
    const auto active_chunks = get_predict_chunks();
    const auto n_chunks = active_chunks.size();
    const auto chunk_divisor = 1. / active_chunks.size();
    // #pragma omp parallel ADJ_THREADS(ixs.size() * x_predict_t.n_cols * PROPS.get_weight_layers())
// #pragma omp single
    {
        const auto [start_chunk, end_chunk] = get_mpi_bounds(n_chunks);
        // OMP_TASKLOOP_1()
        for (auto ch = start_chunk; ch < end_chunk; ++ch) {
            const auto chunk_ix = active_chunks[ch];
            const auto p_params = get_params_ptr(chunk_ix);
            assert(p_params);
            arma::mat scaled_x_predict_t = x_predict_t;
            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
            business::DQScalingFactorService::scale_features_I(chunk_ix, gradient, step, p_params->get_lag_count(), chunk_sf, scaled_x_predict_t);
            arma::mat chunk_predict_K = kernel::IKernel<double>::get(*p_params)->kernel(
                ccache(), scaled_x_predict_t, train_feature_chunks_t[chunk_ix], time, last_trained_time);
            assert(x_predict_t.n_cols == chunk_predict_K.n_rows);
            assert(weight_chunks[chunk_ix].n_cols % l_cols == 0);
            arma::mat multiplicated(chunk_predict_K.n_rows, l_cols, arma::fill::zeros);
            // OMP_TASKLOOP_(chunk_predict_K.n_rows,)
            for (uint32_t predict_row = 0; predict_row < chunk_predict_K.n_rows; ++predict_row) {
                arma::mat K_row_t = chunk_predict_K.row(predict_row).t();
                if (l_cols > 1) K_row_t = common::extrude_cols(K_row_t, l_cols);
                t_omp_lock l1;
                // OMP_TASKLOOP(weight_chunks[chunk_ix].n_cols / l_cols)
                for (uint32_t start_col = 0; start_col < weight_chunks[chunk_ix].n_cols; start_col += l_cols) {
                    const double pred = common::mean<double>(weight_chunks[chunk_ix].cols(start_col, start_col + l_cols - 1) % (K_row_t + train_label_chunks[chunk_ix]));
                    l1.set();
                    multiplicated.row(predict_row) += pred;
                    l1.unset();
                }
            }
            const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
            business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, multiplicated);
            if (chunks_score.size() > 1) multiplicated *= chunk_divisor * chunks_score[chunk_ix];
            LOG4_TRACE("Chunk " << chunk_ix << " predicted " << common::present(multiplicated) << " from " << common::present(chunk_predict_K) << " with " <<
                common::present(weight_chunks[chunk_ix]) << ", scaling factor " << *p_labels_sf << ", labels " << common::present(train_label_chunks[chunk_ix]) << ", features " <<
                common::present(train_feature_chunks_t[chunk_ix]) << ", chunk divisor " << chunk_divisor << ", time " << time << ", column " << p_params->get_input_queue_column_name() <<
                ", predict features " << common::present(scaled_x_predict_t));
            l2.set();
            if (prediction.empty()) prediction = multiplicated;
            else prediction += multiplicated;
            l2.unset();
        }
    }

    if (const auto world_size = PROPS.get_mpi_size(); world_size > 1) {
        if (PROPS.get_mpi_rank()) {
            mpi_errchk(MPI_Gather(prediction.mem, prediction.n_elem, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, PROPS.get_mpi_comm()));
        } else {
            std::vector<double> mpi_all_predictions(prediction.n_elem * world_size);
            mpi_errchk(MPI_Gather(prediction.mem, prediction.n_elem, MPI_DOUBLE, mpi_all_predictions.data(), prediction.n_elem, MPI_DOUBLE, 0, PROPS.get_mpi_comm()));
            for (DTYPE(world_size) i = 0; i < world_size; ++i)
                prediction += arma::mat(mpi_all_predictions.data() + i * prediction.n_elem, prediction.n_rows, prediction.n_cols, false, true);
        }
    }

    LOG4_TRACE("For " << time << ", predicted " << common::present(prediction));
    return prediction;
}

#ifdef INTEGRATION_TEST

arma::mat OnlineSVR::predict(const arma::mat &x_predict, const arma::mat &y_reference, const bpt::ptime &time)
{
    LOG4_BEGIN();
    const auto x_predict_t = predict_chunk_t(x_predict);
    arma::mat prediction;
    t_omp_lock l2;
    const auto l_cols = p_labels->n_cols;
    const auto active_chunks = get_predict_chunks();
    const auto chunk_divisor = 1. / active_chunks.size();
// #pragma omp parallel ADJ_THREADS(ixs.size() * x_predict_t.n_cols * PROPS.get_weight_layers())
// #pragma omp single
    {
        // OMP_TASKLOOP_1()
        for (uint32_t ch = 0; ch < active_chunks.size(); ++ch) {
            const auto chunk_ix = active_chunks[ch];
            const auto p_params = get_params_ptr(chunk_ix);
            assert(p_params);
            arma::mat scaled_x_predict_t = x_predict_t;
            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
            const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, step, level, false, true);
            business::DQScalingFactorService::scale_features_I(chunk_ix, gradient, step, p_params->get_lag_count(), chunk_sf, scaled_x_predict_t);
            arma::mat chunk_predict_K = kernel::IKernel<double>::get(*p_params)->kernel(
                ccache(), scaled_x_predict_t, train_feature_chunks_t[chunk_ix], time, last_trained_time);
            arma::mat scaled_ref_labels = y_reference;
            business::DQScalingFactorService::scale_labels_I(*p_labels_sf, scaled_ref_labels);
            const auto ref_K = kernel::get_reference_Z(scaled_ref_labels, train_label_chunks[chunk_ix]);
            LOG4_TRACE("Difference between reference and predicted K " << common::present<double>(ref_K - chunk_predict_K) << ", reference " << common::present<double>(ref_K) <<
                ", predicted " << common::present<double>(chunk_predict_K));
            assert(x_predict_t.n_cols == chunk_predict_K.n_rows);
            assert(weight_chunks[chunk_ix].n_cols % l_cols == 0);
            arma::mat multiplicated(chunk_predict_K.n_rows, l_cols, arma::fill::zeros);
            // OMP_TASKLOOP_(chunk_predict_K.n_rows,)
            for (uint32_t predict_row = 0; predict_row < chunk_predict_K.n_rows; ++predict_row) {
                arma::mat K_row_t = arma::trans(chunk_predict_K.row(predict_row));
                if (l_cols > 1) K_row_t = common::extrude_cols(K_row_t, l_cols);
                // t_omp_lock l1;
                // OMP_TASKLOOP(weight_chunks[chunk_ix].n_cols / l_cols)
                for (uint32_t start_col = 0; start_col < weight_chunks[chunk_ix].n_cols; start_col += l_cols) {
                    const double pred = common::mean<double>(weight_chunks[chunk_ix].cols(start_col, start_col + l_cols - 1) % (K_row_t + train_label_chunks[chunk_ix]));
                    // l1.set();
                    multiplicated.row(predict_row) += pred;
                    // l1.unset();
                }
            }
            business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, multiplicated);
            if (chunks_score.size() > 1) multiplicated *= chunk_divisor * chunks_score[chunk_ix];
            LOG4_TRACE("Chunk " << chunk_ix << " predicted " << common::present(multiplicated) << " from " << common::present(chunk_predict_K) << " with " <<
                common::present(weight_chunks[chunk_ix]) << ", scaling factor " << *p_labels_sf << ", labels " << common::present(train_label_chunks[chunk_ix]) << ", features " <<
                common::present(train_feature_chunks_t[chunk_ix]) << ", chunk divisor " << chunk_divisor << ", time " << time << ", column " << p_params->get_input_queue_column_name() <<
                ", predict features " << common::present(scaled_x_predict_t));
            l2.set();
            if (prediction.empty()) prediction = multiplicated;
            else prediction += multiplicated;
            l2.unset();
        }
    }
    LOG4_TRACE("For " << time << ", predicted " << common::present(prediction));
    return prediction;
}

#endif

// TODO Review and test
t_gradient_data OnlineSVR::produce_residuals()
{
    if (ixs.size() < 2) LOG4_THROW("At least two chunks are needed to produce residuals.");

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
        business::DQScalingFactorService::scale_features_I(i, gradient, step, p_params->get_lag_count(), chunk_sf, excluded_features_t);
        const auto p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, i, gradient, step, level, false, true);
        business::DQScalingFactorService::scale_labels_I(*p_sf, excluded_labels);
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
        ptr<arma::mat>(residuals.rows(res_rows) / row_divisors.rows(res_rows))
    };
}
} // namespace datamodel
} // namespace svr
