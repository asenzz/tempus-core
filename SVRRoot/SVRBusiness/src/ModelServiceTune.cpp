//
// Created by zarko on 6/24/21.
//

#include <optimizer.hpp>

#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DAO/ModelDAO.hpp"
#include "common/thread_pool.hpp"
#include <iostream>
#include <fstream>
#include <armadillo>
#include <error.h>

using namespace svr::common;
using namespace svr::datamodel;

namespace svr {
namespace business {

static std::atomic<size_t> calls_ct = 0;

cycle_returns_t
train_predict_cycle(
        const int validate_start_pos,
        const matrix_ptr &p_all_labels_mx,
        const matrix_ptr &p_all_features_mx,
        const svr::datamodel::SVRParameters &svr_parameters,
        const std::vector<matrices_ptr> &kernel_matrices = {},
        const bool train_only = false)
{
    ++calls_ct;
    const auto pred_len = PROPS.get_future_predict_count() + PROPS.get_multistep_len();
    cycle_returns_t all_cycle_returns{
        0,
        0,
        std::vector<double>(pred_len, 0),
        std::vector<double>(pred_len, 0),
        0,
        std::vector<double>(pred_len, 0)};
    std::mutex slide_mx;
    LOG4_DEBUG("Train predict cycle, slides " << PROPS.get_slide_count() << ", multistep len " << PROPS.get_multistep_len() << ", future predict count " << PROPS.get_future_predict_count());
#pragma omp parallel for default(shared)
    for (size_t slide_ix = 0; slide_ix < PROPS.get_slide_count(); ++slide_ix) {
        svr::OnlineMIMOSVR slide_svr_model(std::make_shared<SVRParameters>(svr_parameters), MimoType::single, PROPS.get_multistep_len());
        const auto slide_offset = slide_ix * PROPS.get_future_predict_count() / PROPS.get_slide_count();
        const auto slide_start_pos = validate_start_pos + slide_offset;
        {
            const matrix_ptr p_slide_features = std::make_shared<arma::mat>(p_all_features_mx->rows(slide_start_pos - svr_parameters.get_svr_decremental_distance(), slide_start_pos - 1));
            const matrix_ptr p_slide_labels = std::make_shared<arma::mat>(p_all_labels_mx->rows(slide_start_pos - svr_parameters.get_svr_decremental_distance(), slide_start_pos - 1));
            PROFILE_EXEC_TIME(slide_svr_model.batch_train(p_slide_features, p_slide_labels, false, kernel_matrices.empty() ? matrices_ptr{} : kernel_matrices[slide_ix]), "Batch train");
        }
        /* No workie!
        const auto cycle_return = OnlineMIMOSVR::future_validate(slide_start_pos, slide_svr_model, *p_all_features_mx, *p_all_labels_mx, *p_all_labels_mx, times);
        std::scoped_lock lg(slide_mx);
        std::get<0>(all_cycle_returns) += std::get<0>(cycle_return);
        std::get<1>(all_cycle_returns) += std::get<1>(cycle_return);
        std::copy(std::get<2>(cycle_return).begin(), std::get<2>(cycle_return).end(), std::next(std::get<2>(all_cycle_returns).begin(), slide_offset));
        std::copy(std::get<3>(cycle_return).begin(), std::get<3>(cycle_return).end(), std::next(std::get<3>(all_cycle_returns).begin(), slide_offset));
        std::get<4>(all_cycle_returns) += std::get<4>(cycle_return);
        std::copy(std::get<5>(cycle_return).begin(), std::get<5>(cycle_return).end(), std::next(std::get<5>(all_cycle_returns).begin(), slide_offset));
         */
    }
    std::get<0>(all_cycle_returns) /= double(PROPS.get_slide_count());
    std::get<1>(all_cycle_returns) /= double(PROPS.get_slide_count());
    std::get<4>(all_cycle_returns) /= double(PROPS.get_slide_count());
    LOG4_DEBUG("Cycle done, parameters " << svr_parameters.to_sql_string() << " calls count " << calls_ct << " MAE " << std::get<0>(all_cycle_returns) << " MAPE " << std::get<1>(all_cycle_returns) << " Lin MAPE " << std::get<4>(all_cycle_returns));
    return train_only ? cycle_returns_t{} : all_cycle_returns;
}

cycle_returns_t
ModelService::train_predict_cycle_online(
        const int validate_start_pos,
        const matrix_ptr &p_all_labels_mx,
        const matrix_ptr &p_all_features_mx,
        svr::OnlineMIMOSVR &svr_model)
{
    ++calls_ct;
    const auto &features_batch = p_all_features_mx->rows(validate_start_pos - 1, validate_start_pos - 1);
    const auto &labels_batch = p_all_labels_mx->rows(validate_start_pos - 1, validate_start_pos - 1);
    LOG4_DEBUG("Online cycle parameters " << svr_model.get_svr_parameters().to_sql_string() << " calls count " << calls_ct);
    PROFILE_EXEC_TIME(svr_model.learn(features_batch, labels_batch, false, true), "Online train");
    return {};
//    return OnlineMIMOSVR::future_validate(validate_start_pos, svr_model, *p_all_features_mx, *p_all_labels_mx, *p_all_labels_mx);
}

cycle_returns_t
ModelService::final_cycle(
        const SVRParameters_ptr &p_best_parameters,
        const int validate_start_pos,
        const matrix_ptr &p_all_labels_mx,
        const matrix_ptr &p_all_features_mx)
{
#if 0 // Batch forecast (faster)
    return train_predict_cycle(validate_start_pos, p_all_labels_mx, p_all_features_mx, *p_best_parameters);
#else // Online forecast (slower, more precise)
    const auto num_cycles = p_all_labels_mx->n_rows - validate_start_pos;
    LOG4_DEBUG("Train predict cycle, num cycles " << num_cycles << ", multistep len " << PROPS.get_multistep_len() << ", future predict count " << PROPS.get_future_predict_count());
    svr::OnlineMIMOSVR p_svr_model(
        p_best_parameters,
        std::make_shared<arma::mat>(p_all_features_mx->rows(0, validate_start_pos)),
        std::make_shared<arma::mat>(p_all_labels_mx->rows(0, validate_start_pos)),
        false, matrices_ptr{}, false, MimoType::single, p_all_labels_mx->n_cols);
    std::vector<double> best_actual_values, best_linpred_values, best_predict_values;
    double all_mae = 0, all_mape = 0, all_lin_mape = 0;
    for (size_t ix = 1; ix < num_cycles; ++ix) {
#ifdef FAUX_FINAL_ONLINE
        best_actual_values.push_back(0);
        best_linpred_values.push_back(0);
        best_predict_values.push_back(0);
#else
        const auto cycle_returns = train_predict_cycle_online(validate_start_pos + ix, p_all_labels_mx, p_all_features_mx, p_svr_model);
        all_mae += std::get<0>(cycle_returns);
        all_mape += std::get<1>(cycle_returns);
        const auto predicted_values = std::get<2>(cycle_returns);
        const auto actual_values = std::get<3>(cycle_returns);
        all_lin_mape += std::get<4>(cycle_returns);
        const auto linpred_values = std::get<5>(cycle_returns);
        LOG4_DEBUG("Online train predict cycle " << ix << " MAE: " << std::get<0>(cycle_returns) << ", MAPE: " << std::get<1>(cycle_returns) << ", Lin MAPE: " << std::get<4>(cycle_returns) << ", Parameters: " << p_best_parameters->to_sql_string());
        best_actual_values.push_back(actual_values[0]);
        best_linpred_values.push_back(linpred_values[0]);
        best_predict_values.push_back(predicted_values[0]);
#endif
    }
    all_mae /= (num_cycles - 1);
    all_mape /= (num_cycles - 1);
    all_lin_mape /= (num_cycles - 1);
    LOG4_DEBUG("MAE " << all_mae << ", MAPE " << all_mape << " pct , Lin MAPE " << all_lin_mape << " pct");
    return {all_mae, all_mape, best_actual_values, best_linpred_values, all_lin_mape, best_predict_values};
#endif
}

} // namespace svr {
} //namespace business {
