#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <boost/date_time/posix_time/ptime.hpp>
#include "model/DataRow.hpp"
#include "model/SVRParameters.hpp"
#include "model/Ensemble.hpp"
#include "DeconQueueService.hpp"
#ifdef INTEGRATION_TEST
#include <LightGBM/c_api.h>
#include "kernel_gbm.hpp"
#endif
#include "ModelService.hpp"
#include "align_features.cuh"
#include "appcontext.hpp"
#include "DataRowService.hpp"
#include "DQScalingFactorService.hpp"
#include "firefly.hpp"
#include "onlinesvr.hpp"
#include "SVRParametersService.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"
#include "common/defines.h"
#include "common/logging.hpp"
#include "common/parallelism.hpp"
#include "DAO/ModelDAO.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "util/time_utils.hpp"
#include "util/validation_utils.hpp"
#include "common/exceptions.hpp"


namespace svr {
namespace business {


uint32_t ModelService::get_max_row_len()
{
    return PROPS.get_max_quant() * (1 + PROPS.get_lag_multiplier() * datamodel::C_default_svrparam_lag_count);
}


ModelService::ModelService(dao::ModelDAO &model_dao) : model_dao(model_dao)
{
}


uint16_t ModelService::to_level_ct(const uint16_t model_ct) noexcept
{
#ifdef VMD_ONLY
     model_ct * 2 / PROPS.get_steps();
#elif defined(EMD_ONLY)
    return model_ct / PROPS.get_steps();
#else
    return model_ct >= MIN_LEVEL_COUNT / 2 - 1 ? model_ct * 2 + 2 : 1;
#endif
}

uint16_t ModelService::to_model_ct(const uint16_t level_ct) noexcept
{
    return
#ifdef VMD_ONLY
    level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 : 1
#elif defined(EMD_ONLY)
    level_ct
#else
    level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 - 1 : 1
#endif
    * PROPS.get_steps();
}

#ifdef INTEGRATION_TEST

#define LGBM_MAXBIN "255"
// #define SCALE_REF

arma::mat aux_train_predict(const datamodel::SVRParameters &param, const arma::mat &features_, const arma::mat &labels_, const uint32_t start_ix_)
{
    const auto start_ix = start_ix_ - PROPS.get_shift_limit();
    const arma::uvec all_ixs = arma::regspace<arma::uvec>(PROPS.get_shift_limit(), labels_.n_rows - 1);
    const auto features_t = datamodel::OnlineSVR::sst(features_, param.get_feature_mechanics(), all_ixs);
    const arma::mat labels = labels_.rows(all_ixs);
    arma::uvec shifted_train_ixs = all_ixs.rows(0, start_ix - 1);
    if (PROPS.get_outlier_slack()) datamodel::OnlineSVR::score_indexes(features_t.cols(0, start_ix - 1), labels.rows(0, start_ix - 1), shifted_train_ixs);
    shifted_train_ixs -= PROPS.get_shift_limit();
    arma::fmat train_features_t = arma::conv_to<arma::fmat>::from(features_t.cols(shifted_train_ixs));
    arma::fmat train_labels = arma::conv_to<arma::fmat>::from(labels.rows(shifted_train_ixs));
    arma::fmat predict_features_t = arma::conv_to<arma::fmat>::from(features_t.cols(start_ix, features_t.n_cols - 1));
    arma::fmat predict_labels = arma::conv_to<arma::fmat>::from(labels.rows(start_ix, labels.n_rows - 1));
#ifdef SCALE_REF // scale>
    const auto features_sf = business::DQScalingFactorService::calculate(0, param, train_features_t, train_labels);
    const auto p_labels_sf = business::DQScalingFactorService::find(features_sf, 0, param.get_chunk_index(), param.get_grad_level(), param.get_step(), param.get_decon_level(), false, true);
    business::DQScalingFactorService::scale_features_I(param.get_chunk_index(), param.get_grad_level(), param.get_step(), param.get_lag_count(), features_sf, train_features_t);
    business::DQScalingFactorService::scale_labels_I(*p_labels_sf, train_labels);
    business::DQScalingFactorService::scale_features_I(param.get_chunk_index(), param.get_grad_level(), param.get_step(), param.get_lag_count(), features_sf, predict_features_t);
#endif
    LGBM_ERRCHK(LGBM_SetMaxThreads(C_n_cpu));
    DatasetHandle train_dataset;
    const auto lgbm_dataset_parameters = kernel::get_lgbm_dataset_parameters();
    LGBM_ERRCHK(LGBM_DatasetCreateFromMat(train_features_t.mem, C_API_DTYPE_FLOAT32, train_features_t.n_cols, train_features_t.n_rows, 1, // is_row_major = 1 (row-major order)
        lgbm_dataset_parameters.c_str(), nullptr, &train_dataset));

    LGBM_ERRCHK(LGBM_DatasetSetField(train_dataset, "label", train_labels.mem, train_labels.n_rows, C_API_DTYPE_FLOAT32));

    BoosterHandle booster;
    const std::string lgbm_core_parameters = common::formatter() << "objective=regression tree_learner=data seed=123 learning_rate=" << PROPS.get_k_learn_rate() << " num_iterations=" <<
                                             PROPS.get_k_epochs() << " early_stopping_round=200 metric=l2 force_col_wise=true num_threads=" << C_n_cpu << " device_type=gpu " <<
                                             lgbm_dataset_parameters;
    LGBM_ERRCHK(LGBM_BoosterCreate(train_dataset, lgbm_core_parameters.c_str(), &booster));
    int train_complete = 0;
    auto iter = PROPS.get_k_epochs() + 1;
    assert(iter);
    while (train_complete != 1 && --iter) LGBM_ERRCHK(LGBM_BoosterUpdateOneIter(booster, &train_complete));
    arma::vec res(predict_features_t.n_cols, ARMA_DEFAULT_FILL);
    int64_t out_len;
    LGBM_ERRCHK(
        LGBM_BoosterPredictForMat(booster, predict_features_t.mem, C_API_DTYPE_FLOAT32, predict_features_t.n_cols, predict_features_t.n_rows, 1, C_API_PREDICT_NORMAL, 0, 0,
            lgbm_core_parameters.c_str(), &out_len, res.memptr()));
    LGBM_ERRCHK(LGBM_BoosterFree(booster));
    LGBM_ERRCHK(LGBM_DatasetFree(train_dataset));
#ifdef SCALE_REF
    business::DQScalingFactorService::unscale_labels_I(*p_labels_sf, res);
#endif
    LOG4_DEBUG(
        "Predicted " << common::present(res) << ", difference " << common::present<double>(arma::vectorise(predict_labels) - res) << ", reference " << common::present<float>(predict_labels))
    ;
    return res;
}

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, arma::vec, arma::vec, arma::vec, double, arma::vec>
ModelService::validate(const uint32_t start_ix, const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model, const arma::mat &features,
                       const arma::mat &labels, const arma::vec &last_knowns, const arma::mat &weights, const datamodel::data_row_container &times, const bool online, const bool verbose)
{
    LOG4_BEGIN();
    if (labels.n_rows <= start_ix)
        LOG4_THROW("Calling future validate " << start_ix << " at the end of labels array " << labels.n_rows);

    const uint32_t ix_fini = labels.n_rows - 1;
    const uint32_t num_preds = labels.n_rows - start_ix;
    const auto head_param = model.get_head_param();
    const auto level = head_param.get_decon_level();

    datamodel::t_level_predict_features predict_features({times.cbegin() + start_ix, times.cend()}, otr<arma::mat>(features.rows(start_ix, ix_fini)));
    LOG4_TRACE("Predicting features " << common::present<double>(*predict_features.p));
    datamodel::data_row_container batch_predicted, cont_predicted_online;
    tbb::mutex mx;
    PROFILE_INFO(predict(ensemble, model, predict_features, dataset.get_input_queue()->get_resolution(), mx, labels.rows(start_ix, ix_fini), batch_predicted),
                 "Batch predict of " << num_preds << " rows, level " << level << ", step " << model.get_step());
    arma::mat predict_lgbm;
    PROFIL3(predict_lgbm = aux_train_predict(head_param, features, labels, start_ix));
    if (batch_predicted.size() != num_preds || predict_lgbm.n_rows != num_preds || predict_lgbm.n_cols != 1)
        LOG4_THROW("Predicted size " << batch_predicted.size() << " not sane " << arma::size(*predict_features.p) << ", LGBM predicted " << common::present(predict_lgbm));
    predict_lgbm += last_knowns.rows(start_ix, ix_fini);

    LOG4_DEBUG("Batch predicted " << batch_predicted.size() << " values, parameters " << head_param);
    const auto stepping = model.get_gradient()->get_dataset()->get_steps();
    arma::vec predicted_batch(num_preds), predicted_online(num_preds), actual = arma::mean(labels.rows(start_ix, ix_fini), 1), lastknown = last_knowns.rows(start_ix, ix_fini);
#ifdef EMO_DIFF
    OMP_FOR_i(actual.n_cols) actual.col(i) += lastknown; // common::sexp<double>(actual.col(i)) + lastknown;
#endif
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0, sum_absdiff_lgbm = 0;
    double batch_correct_directions = 0, lgbm_correct_directions = 0, lgbm_correct_predictions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (uint32_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted_batch[ix] = stepping * batch_predicted[ix]->at(level);
        const double cur_absdiff_lk = std::abs(lastknown[ix] - actual[ix]);
        const double cur_absdiff_batch = std::abs(predicted_batch[ix] - actual[ix]);
        const double cur_absdiff_lgbm = std::abs(predict_lgbm(ix, 0) - actual[ix]);
        const double cur_alpha_pct_batch = common::alpha(cur_absdiff_lk, cur_absdiff_batch);
        const double cur_alpha_pct_lgbm = common::alpha(cur_absdiff_lk, cur_absdiff_lgbm);
        sum_abs_labels += std::abs(actual[ix]);
        sum_absdiff_batch += cur_absdiff_batch;
        sum_absdiff_lgbm += cur_absdiff_lgbm;
        const auto actual_move = actual[ix] - lastknown[ix];
        sum_absdiff_lk += std::abs(actual_move);
        batch_correct_predictions += cur_absdiff_batch < cur_absdiff_lk;
        const auto actual_sign = std::signbit(actual_move);
        batch_correct_directions += std::signbit(predicted_batch[ix] - lastknown[ix]) == actual_sign;
        lgbm_correct_predictions += cur_absdiff_lgbm < cur_absdiff_lk;
        lgbm_correct_directions += std::signbit(predict_lgbm[ix] - lastknown[ix]) == actual_sign;

        const auto ix_div = ix + 1.;
        const bool print_line = verbose || ix_future == ix_fini || ix % 115 == 0;
        std::stringstream row_report;
        if (print_line)
            row_report << "Position " << ix << ", level " << level << ", step " << model.get_step() <<
                    ", actual " << actual[ix] << ", batch predicted " << predicted_batch[ix] << ", LGBM predicted " << predict_lgbm[ix] << ", last known " << lastknown[ix] <<
                    " batch MAE " << sum_absdiff_batch / ix_div << ", MAE last-known " << sum_absdiff_lk / ix_div << " LGBM MAE " << sum_absdiff_lgbm / ix_div <<
                    ", LGBM MAPE " << common::mape(sum_absdiff_lgbm, sum_abs_labels) <<
                    "pc, batch MAPE " << common::mape(sum_absdiff_batch, sum_abs_labels) <<
                    "pc, MAPE last-known " << common::mape(sum_absdiff_lk, sum_abs_labels) <<
                    "pc, batch alpha " << common::alpha(sum_absdiff_lk, sum_absdiff_batch) <<
                    "pc, LGBM alpha " << common::alpha(sum_absdiff_lk, sum_absdiff_lgbm) <<
                    "pc, current batch alpha " << cur_alpha_pct_batch << "pc, current LGBM alpha " << cur_alpha_pct_lgbm <<
                    "pc, batch correct predictions " << 100. * batch_correct_predictions / ix_div << "pc, batch correct directions " << 100. * batch_correct_directions / ix_div << "pc" <<
                    "pc, LGBM correct predictions " << 100. * lgbm_correct_predictions / ix_div << "pc, LGBM correct directions " << 100. * lgbm_correct_directions / ix_div << "pc";
        if (online) {
            PROFILE_INFO(
                ModelService::predict(
                    ensemble, model,
                    datamodel::t_level_predict_features{
                    {times[ix]},
                    ptr<arma::mat>(features.row(ix_future))},
                    dataset.get_input_queue()->get_resolution(), mx, cont_predicted_online),
                "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                ", step " << model.get_step() << " at " << times[ix_future]);
            PROFILE_INFO(
                ModelService::train_online(model, features.row(ix_future), labels.row(ix_future), weights.row(ix_future),
                    times[ix_future]->get_value_time()),
                "Online learn " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                ", step " << model.get_step() << " at " << times[ix_future]);
            predicted_online[ix] = stepping * cont_predicted_online.front()->at(level);
#ifdef EMO_DIFF
            predicted_online[ix] += lastknown[ix]; // TODO Test common::sexp(predicted_online[ix]) + lastknown[ix];
#endif
            const double cur_absdiff_online = std::abs(predicted_online[ix] - actual[ix]);
            const double cur_alpha_pct_online = common::alpha(cur_absdiff_lk, cur_absdiff_online);
            sum_absdiff_online += cur_absdiff_online;
            online_correct_predictions += cur_absdiff_online < cur_absdiff_lk;
            online_correct_directions += std::signbit(predicted_online[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);
            if (print_line)
                row_report << ", online predicted " << predicted_online[ix] << ", online MAE " << sum_absdiff_online / ix_div << ", online MAPE " <<
                        common::mape(sum_absdiff_online, sum_abs_labels) << "pc, online alpha " << common::alpha(sum_absdiff_lk, sum_absdiff_online)
                        << "pc, current online alpha " << cur_alpha_pct_online << "pc, online correct predictions " << 100. * online_correct_predictions / ix_div
                        << "pc, online correct directions " << 100. * online_correct_directions / ix_div << "pc";
        }
        if (row_report.tellp() != std::streampos(0)) LOG4_INFO(row_report.str());
    }
    const auto mape_lk = 100. * sum_absdiff_lk / sum_abs_labels;
    const auto &sum_absdiff = online ? sum_absdiff_online : sum_absdiff_batch;
    const auto &predicted = online ? predicted_online : predicted_batch;
    LOG4_INFO("Parameters " << head_param << ", predictions start " << start_ix << ", last index " << ix_fini << ", concession " << common::present<double>(actual - predicted));
    return {sum_absdiff / double(num_preds), common::mape(sum_absdiff, sum_abs_labels), predicted, predict_lgbm, actual, mape_lk, lastknown};
}

#endif

datamodel::Model_ptr ModelService::get_model_by_id(const bigint model_id) const
{
    return model_dao.get_by_id(model_id);
}


datamodel::Model_ptr
ModelService::find(const std::deque<datamodel::Model_ptr> &models, const uint16_t levix, const uint16_t stepix)
{
    const auto res = std::find_if(C_default_exec_policy, models.cbegin(), models.cend(),
                                  [levix, stepix](const auto &p_model) { return p_model->get_decon_level() == levix && p_model->get_step() == stepix; });
    if (res != models.cend()) return *res;
    LOG4_WARN("Model for level " << levix << ", step " << stepix << " not found among " << models.size() << " models.");
    return nullptr;
}

datamodel::SVRParameters_ptr ModelService::produce_parameters(
    const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, const std::deque<datamodel::SVRParameters_ptr> &paramset,
    const uint16_t chunk_ix, const uint16_t grad_ix)
{
    const auto level_grad_param_set = SVRParametersService::slice(paramset, chunk_ix, grad_ix);
    return level_grad_param_set.size()
               ? *level_grad_param_set.cbegin()
               : ptr<datamodel::SVRParameters>(0, dataset.get_id(), dataset.get_input_queue()->get_table_name(),
                                               ensemble.get_column_name(), dataset.get_spectral_levels(), model.get_decon_level(), model.get_step(), chunk_ix, grad_ix);
}

void ModelService::configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model) const
{
    if (!check(model.get_gradients(), model.get_gradient_count()) && model.get_id())
        model.set_gradients(model_dao.get_svr_by_model_id(model.get_id()), false);

    model.set_max_chunk_size(p_dataset->get_max_chunk_size());
    std::deque<datamodel::SVRParameters_ptr> paramset;
    if (p_dataset->get_id()) paramset = APP.svr_parameters_service.get_by_dataset_column_level(p_dataset->get_id(), ensemble.get_column_name(), model.get_decon_level(), model.get_step());
/* Avoid calculating the number of chunks until first training takes place
    const uint16_t default_model_num_chunks = datamodel::OnlineSVR::get_num_chunks(
          paramset.empty() ? datamodel::C_default_svrparam_decrement_distance : (**paramset.cbegin()).get_svr_decremental_distance(), model.get_max_chunk_size());
*/
    constexpr uint16_t default_model_num_chunks = 1;
    datamodel::dq_scaling_factor_container_t all_model_scaling_factors;
    if (model.get_id()) all_model_scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model.get_id());
    OMP_PAR(p_dataset->get_gradient_count() * default_model_num_chunks * p_dataset->get_spectral_levels())
    {
        tbb::mutex gradients_l;
        OMP_TASKLOOP_1(firstprivate(default_model_num_chunks))
        for (uint16_t gix = 0; gix < p_dataset->get_gradient_count(); ++gix) {
            tbb::mutex::scoped_lock l1(gradients_l);
            auto p_svr_model = model.get_gradient(gix);
            l1.release();

            bool set_params = false;
            uint32_t grad_num_chunks = 0;
            // Prepare this gradient parameters
            datamodel::t_param_set grad_params;
            if (p_svr_model) {
                grad_params = p_svr_model->get_param_set();
                grad_num_chunks = p_svr_model->get_num_chunks();
            } else {
                set_params = true;
                grad_num_chunks = default_model_num_chunks;
            }

            tbb::mutex grad_params_l;
            OMP_TASKLOOP_1(SSIMD firstprivate(gix))
            for (DTYPE(grad_num_chunks) chix = 0; chix < grad_num_chunks; ++chix)
                if (SVRParametersService::slice(grad_params, chix, gix).empty()) {
                    const auto p_params = produce_parameters(*p_dataset, ensemble, model, paramset, chix, gix);
                    const tbb::mutex::scoped_lock l2(grad_params_l);
                    grad_params.emplace(p_params);
                    set_params = true;
                }

            if (!p_svr_model) {
                const tbb::mutex::scoped_lock l2(gradients_l);
                p_svr_model = model.get_gradients().emplace_back(ptr<datamodel::OnlineSVR>(0, model.get_id(), grad_params, p_dataset));
            } else {
                if (set_params) p_svr_model->set_param_set(grad_params);
                if (!p_svr_model->get_dataset()) p_svr_model->set_dataset(p_dataset);
            }

            const auto adjacent_ct = (**grad_params.cbegin()).get_adjacent_levels().size();
            if (p_svr_model->get_scaling_factors().size() != p_svr_model->get_num_chunks()) {
                OMP_TASKLOOP_1(firstprivate(grad_num_chunks))
                for (DTYPE(grad_num_chunks) chix = 0; chix < grad_num_chunks; ++chix) {
                    datamodel::DQScalingFactor_ptr p_sf;
                    if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(), model.get_step(),
                                                      model.get_decon_level(), false, true) &&
                        (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(), p_svr_model->get_step(),
                                                             model.get_decon_level(), false, true))) {
                        const tbb::mutex::scoped_lock l2(gradients_l);
                        p_svr_model->set_scaling_factor(p_sf);
                    }
                    OMP_TASKLOOP_1(SSIMD firstprivate(adjacent_ct))
                    for (DTYPE(adjacent_ct) levix = 0; levix < adjacent_ct; ++levix) {
                        if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                          p_svr_model->get_step(), levix, true, false)
                            && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                                    p_svr_model->get_step(), levix, true, false))) {
                            const tbb::mutex::scoped_lock l2(gradients_l);
                            p_svr_model->set_scaling_factor(p_sf);
                        }
                    }
                }
            }
        }
    }
}


int ModelService::save(const datamodel::Model_ptr &p_model) const
{
    REJECT_NULLPTR(p_model);
    if (!p_model->get_id()) p_model->set_id(model_dao.get_next_id());
    return model_dao.save(p_model);
}

bool ModelService::exists(const datamodel::Model &model) const
{
    return model_dao.exists(model.get_id());
}

int ModelService::remove(const datamodel::Model_ptr &model) const
{
    REJECT_NULLPTR(model);
    return model_dao.remove(model);
}

int ModelService::remove_by_ensemble_id(const bigint ensemble_id) const
{
    return model_dao.remove_by_ensemble_id(ensemble_id);
}

std::deque<datamodel::Model_ptr> ModelService::get_all_models_by_ensemble_id(const bigint ensemble_id) const
{
    return model_dao.get_all_ensemble_models(ensemble_id);
}

datamodel::Model_ptr ModelService::get_model(const bigint ensemble_id, const uint16_t decon_level) const
{
    return model_dao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

bool ModelService::check(const std::deque<datamodel::Model_ptr> &models, const uint16_t model_ct)
{
    const auto level_ct = to_level_ct(model_ct);
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [level_ct](const auto &p_model) { return p_model->get_decon_level() < level_ct; });
}

bool ModelService::check(const std::deque<datamodel::OnlineSVR_ptr> &models, const uint16_t grad_ct)
{
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [grad_ct](const auto p_model) { return p_model->get_gradient_level() < grad_ct; });
}

arma::rowvec ModelService::prepare_special_features(const datamodel::data_row_container::const_iterator &last_known_it, const bpt::time_duration &resolution, const uint32_t len)
{
    const bpt::ptime value_time = (**last_known_it).get_value_time();
    LOG4_TRACE("Processing row with value time " << value_time);

    std::deque<double> spec_features;
    spec_features.emplace_back(double(value_time.time_of_day().hours()) / 24.); // Hour of the day
    spec_features.emplace_back(double(value_time.date().day_of_week()) / 7.);
    spec_features.emplace_back(double(value_time.date().day()) / 31.);
    spec_features.emplace_back(double(value_time.date().week_number()) / 52.);
    spec_features.emplace_back(double(value_time.date().month()) / 12.);
    const auto step = len / spec_features.size();

    arma::rowvec row(len, ARMA_DEFAULT_FILL);
    for (uint32_t i = 0; i < spec_features.size(); ++i) row.subvec(i * step, i == spec_features.size() - 1 ? row.n_cols - 1 : (i + 1) * step - 1).fill(spec_features[i]);
    return row;
}

void ModelService::prepare_weights(
        arma::mat &weights, const datamodel::data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const arma::fvec &steps,
        const bpt::time_duration &label_duration)
{
    LOG4_BEGIN();

    const auto num_rows = times.size();
    if (num_rows < 1) LOG4_THROW("No times to prepare weights for.");
    if (weights.n_rows != num_rows || weights.n_cols != steps.size()) weights.set_size(num_rows, steps.size());
    const arma::fvec cumsteps = arma::cumsum(steps);
    OMP_PAR(num_rows * aux_inputs.size() * steps.size())
    {
        OMP_TASKLOOP_(num_rows,)
        for (DTYPE(num_rows) i = 0; i < num_rows; ++i) {
            const auto &t = times[i];
            for (const auto &q: aux_inputs)
                OMP_TASKLOOP_1()
                for (uint16_t s = 0; s < steps.size(); ++s) {
                    const auto s_start = t->get_value_time() + (s ? label_duration * cumsteps[s - 1] : bpt::seconds(0));
                    for (auto it = lower_bound(std::as_const(q->get_data()), s_start); it != q->cend() && (*it)->get_value_time() < s_start + label_duration * cumsteps[s]; ++it)
                        weights(i, s) += (**it).get_tick_volume();
                }
        }
    }
    weights = (weights + PROPS.get_instance_inert()) / (PROPS.get_instance_inert() + 1);

    LOG4_END();
}

datamodel::t_model_train_data
ModelService::get_training_data(datamodel::Dataset &dataset, datamodel::Ensemble &ensemble, const uint16_t level, uint32_t dataset_rows)
{
    LOG4_BEGIN();

    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    const auto steps = dataset.get_steps();
    const auto do_dataset_rows = dataset_rows == 0;
    bpt::ptime last_modeled_time = bpt::max_date_time;
    std::deque<datamodel::SVRParameters_ptr> params(steps);
    for (DTYPE(steps) s = 0; s < steps; ++s) {
        auto model = ensemble.get_model(level, s);
        auto p = model->get_head_param_ptr();
        if (do_dataset_rows && dataset_rows < p->get_svr_decremental_distance()) dataset_rows = p->get_svr_decremental_distance();
        if (model->get_last_modeled_value_time() < last_modeled_time) last_modeled_time = model->get_last_modeled_value_time();
        params[s] = p;
    }
    const auto resolution = dataset.get_input_queue()->get_resolution();
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? resolution : dataset.get_aux_input_queue()->get_resolution();
    const datamodel::datarow_crange labels_range{
            DataRowService::get_start(label_decon.get_data().cbegin(), label_decon.get_data().cend(), dataset_rows, last_modeled_time, resolution),
            label_decon.get_data().cend(), label_decon};
    const auto [p_features, p_labels, p_last_knowns, p_label_times] = calc_cache::get_training_data(
            dataset.get_steps(), labels_range, labels_aux, dataset.get_max_lookback_time_gap(), last_modeled_time, resolution, aux_resolution,
            ensemble.get_aux_decon_queues(), params);
    OMP_FOR_i(steps) {
        auto model = ensemble.get_model(level, i);

        const auto param_set = model->get_gradient()->get_param_set();
        for (auto &p: param_set) // Set all chunks in the model'i root gradient to the same feature mechanics
            if (p->get_feature_mechanics().needs_tuning())
                p->set_feature_mechanics(front(param_set)->get_feature_mechanics());
    }
    const auto p_weights = calc_cache::get_weights(dataset.get_id(), *p_label_times, dataset.get_aux_input_queues(), ensemble.get_model(level)->get_gradient()->get_params().get_feature_mechanics().steps, resolution);
    assert(p_labels->n_rows == p_weights->n_rows);

    return {p_features, p_labels, p_last_knowns, p_weights, p_label_times};
}


void ModelService::coordinates_knowns(
        arma::vec &out_last_knowns, datamodel::data_row_container &times, std::vector<t_label_ix> &L_ixs, std::vector<uint32_t> &ix_F_end, const datamodel::datarow_crange &main_data,
        const datamodel::datarow_crange &aux_data, const bpt::time_duration &max_gap, const uint16_t level, const bpt::time_duration &resolution_aux, const bpt::ptime &last_time,
        const bpt::time_duration &resolution, const uint32_t lag)
{
    LOG4_BEGIN();
    assert(aux_data.distance() > 1);
    const auto req_rows = main_data.distance();
    assert(req_rows > 0);
    const uint32_t coef_lag = PROPS.get_lag_multiplier() * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
#define coef_lag_ coef_lag
#endif
    LOG4_TRACE("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.front()->get_value_time() << " until " << main_data.back()->get_value_time()
                                  << ", main resolution " << resolution << ", aux resolution " << resolution_aux);
    const auto &label_duration = resolution;
    const auto horizon_duration = resolution * PROPS.get_prediction_horizon();
    const auto valid_start_drift = std::min(label_duration * PROPS.get_label_drift(), max_gap);
    assert(main_data.get_container().size());
    const uint32_t label_len = resolution / resolution_aux;

    L_ixs.reserve(req_rows);
    ix_F_end.reserve(req_rows);
    std::vector<double> last_knowns;
    last_knowns.reserve(req_rows);
    const uint32_t horizon_len_2 = label_len * PROPS.get_prediction_horizon() * 2;
    const auto label_len_1 = label_len + 1;
    const auto stripe_period = resolution_aux * coef_lag_;
    const auto first_time = aux_data.front()->get_value_time();
    const auto max_row_duration = horizon_duration + stripe_period * get_max_quant();
#ifdef NDEBUG
    const uint32_t avail_rows = main_data.cend() - main_data.contcbegin();
    OMP_FOR_(avail_rows, ordered)
#endif
    for (auto it_main_time = main_data.contcbegin(); it_main_time != main_data.cend(); ++it_main_time) {
        const auto L_start_time = (**it_main_time).get_value_time();
        if (L_start_time - max_row_duration < first_time || L_start_time <= last_time) {
            LOG4_TRACE("Skipping time " << L_start_time << " as it is before last modeled time " << last_time << " or before max row duration " << max_row_duration);
            continue;
        }
        const auto L_start_it = lower_bound_or_before(aux_data.cbegin(), aux_data.cend(), L_start_time);
        if (L_start_it == aux_data.cend() || (**L_start_it).get_value_time() > L_start_time + valid_start_drift) {
            LOG4_TRACE("No aux data for time " << L_start_time);
            continue;
        }

        const auto L_end_it = lower_bound(L_start_it, aux_data.cend() - L_start_it > label_len_1 ? L_start_it + label_len_1 : aux_data.cend(), L_start_time + label_duration);
        if (L_end_it - L_start_it < 1) {
            LOG4_TRACE("No aux data for time " << L_start_time);
            continue;
        }
        const auto L_end_it_time = L_end_it == aux_data.cend() ? (**std::prev(L_end_it)).get_value_time() : (**L_end_it).get_value_time();
        bpt::time_duration label_gap;
        if ((label_gap = L_end_it_time - L_start_time) > max_gap) {
            LOG4_TRACE("Label gap " << label_gap << " is larger than max gap " << max_gap << " for time " << L_start_time);
            continue;
        }
        const auto F_end_it = lower_bound_before(L_start_it - aux_data.cbegin() > horizon_len_2 ? L_start_it - horizon_len_2 : aux_data.cbegin(), L_start_it,
                                                 L_start_time - horizon_duration);
        if (F_end_it == aux_data.cend() || F_end_it == aux_data.cbegin()) {
            LOG4_TRACE("No feature data for label at " << L_start_time);
            continue;
        }
        const uint32_t F_end_ix = F_end_it - aux_data.cbegin();
        t_label_ix this_label_ixs{.n_ixs = label_len};
        try {
            if constexpr (C_label_bias == 0)
                generate_twap_indexes(aux_data.cbegin(), L_start_it, L_end_it, L_start_time, label_duration, label_len, this_label_ixs.label_ixs);
            else
                this_label_ixs.special_x = generate_twap_bias(this_label_ixs.label_ixs, false /*askbid*/, aux_data.cbegin(), L_start_it, L_end_it, L_start_time, label_duration,
                                                              label_len, level);
        } catch (...) {
            LOG4_WARN("Failed to generate label indexes for time " << L_start_time << ", aux start iterator time " << (**L_start_it).get_value_time());
            continue;
        }
        LOG4_TRACE("Adding row at " << L_start_time << " label at " << *this_label_ixs.label_ixs << " with " << F_end_ix << " index, of length " << label_len);
#pragma omp ordered
        {
            ix_F_end.emplace_back(F_end_ix);
            times.emplace_back(*it_main_time);
            last_knowns.emplace_back((**F_end_it)[level]);
            L_ixs.emplace_back(this_label_ixs);
        };
    }

#ifdef LAST_KNOWN_LABEL // TODO Implement for online learn!
    const auto L_start_it = aux_data.cend() - 1;
    const auto L_end_it = aux_data.get_container().cend();
    const auto L_start_time = (**L_start_it).get_value_time();
    const auto F_end_it = lower_bound_before(L_start_it - aux_data.cbegin() > horizon_len_2 ? L_start_it - horizon_len_2 : aux_data.cbegin(), L_start_it,
                                             L_start_time - horizon_duration);
    if (F_end_it < aux_data.cend() && F_end_it > aux_data.cbegin()) {
        const uint32_t F_end_ix = F_end_it - aux_data.cbegin();
        t_label_ix this_label_ixs{.n_ixs = label_len};
        try {
            if constexpr (C_label_bias == 0)
                generate_twap_indexes(aux_data.cbegin(), L_start_it, L_end_it, L_start_time, label_duration, label_len, this_label_ixs.label_ixs);
            else
                this_label_ixs.special_x = generate_twap_bias(this_label_ixs.label_ixs, false /*askbid*/, aux_data.cbegin(), L_start_it, L_end_it, L_start_time, label_duration, label_len, level);
        } catch (...) {
            LOG4_WARN("Failed to generate label indexes for time " << L_start_time << ", label duration " << label_duration << ", aux start iterator time " << (**L_start_it).get_value_time());
        }
        ix_F_end.emplace_back(F_end_ix);
        times.emplace_back(*L_start_it);
        last_knowns.emplace_back((**F_end_it)[level]);
        L_ixs.emplace_back(this_label_ixs);
        LOG4_DEBUG("Temporary last known row, time " << L_start_time << " anchor time " << (**F_end_it).get_value_time());
    } else {
        LOG4_WARN("No feature data for label at " << L_start_time);
    }
#endif

    const auto labels_size = L_ixs.size();
    assert(labels_size == ix_F_end.size());
    assert(labels_size == times.size());
    assert(labels_size == last_knowns.size());
    if (CAST2(req_rows) labels_size > req_rows) {
        const auto offshoot = labels_size - req_rows;
        L_ixs.erase(L_ixs.begin(), L_ixs.begin() + offshoot);
        ix_F_end.erase(ix_F_end.begin(), ix_F_end.begin() + offshoot);
        times.erase(times.begin(), times.begin() + offshoot);
        last_knowns.erase(last_knowns.begin(), last_knowns.begin() + offshoot);
    } else if (CAST2(req_rows) labels_size < req_rows)
        LOG4_THROW("Label indexes size " << labels_size << " less than required " << req_rows);
    assert(CAST2(req_rows) L_ixs.size() == req_rows);

    out_last_knowns.set_size(req_rows);
    memcpy(out_last_knowns.memptr(), last_knowns.data(), req_rows * sizeof(double));
}

void ModelService::tune_data(
        std::deque<mat_ptr> &out_features, arma::mat &all_labels, std::deque<datamodel::SVRParameters_ptr> &params, const datamodel::data_row_container &label_times,
        const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &ix_F_end, const std::deque<datamodel::DeconQueue_ptr> &feat_queues, const bpt::time_duration &resolution,
        const bpt::time_duration &resolution_aux, const uint16_t steps, const datamodel::datarow_crange &aux_label_data)
{
    LOG4_BEGIN();

    assert(steps);
    assert(steps == params.size());

    if (label_times.empty()) {
        LOG4_WARN("No new data to prepare for training, labels " << arma::size(all_labels));
        return;
    }

    const auto level = params.front()->get_decon_level();
    LOG4_TRACE("Preparing level " << level << ", labels " << label_times.size());
    std::vector<double> labels_aux_in(aux_label_data.distance());
    const uint32_t label_len = resolution / resolution_aux;
    OMP_FOR_i(aux_label_data.distance()) labels_aux_in[i] = aux_label_data[i]->at(level);
    assert(steps);
    const uint32_t n_rows = label_times.size();
    const uint16_t n_queues = feat_queues.size();
    const auto horizon_duration = resolution * PROPS.get_prediction_horizon();
    const auto &gpu_handler_4 = common::gpu_handler_4::get();
    const auto max_gpu_chunk_size = gpu_handler_4.get_max_gpu_data_chunk_size();
    std::deque<t_step_params> sp(steps);
    OMP_FOR_i(steps) {
        auto &sps = sp[i];
        sps.lag = params[i]->get_lag_count();
        sps.coef_lag = PROPS.get_lag_multiplier() * sps.lag;
        sps.adjacent_levels = params[i]->get_adjacent_levels();
        sps.n_levels = sps.adjacent_levels.size();
        assert(sps.n_levels == params.front()->get_adjacent_levels().size());
        sps.coef_lag_ = sps.coef_lag
#ifdef EMO_DIFF
         + 1;
#else
        ;
#endif
        sps.n_levels_queues = sps.n_levels * n_queues;
        sps.align_features_size = n_rows * sps.coef_lag * sizeof(double) + n_rows * sizeof(double) + sps.coef_lag * sizeof(double) + n_rows * sizeof(uint32_t);
        sps.n_chunks_align = cdiv(sps.align_features_size, max_gpu_chunk_size);
        sps.chunk_len_align = cdiv(sps.coef_lag, sps.n_chunks_align);
        sps.stripe_period = resolution_aux * sps.coef_lag_;
        sps.coef_lag_max_q = sps.coef_lag_ * PROPS.get_max_quant();
        sps.chunk_len_quantise.resize(n_queues);
        sps.in_rows.resize(n_queues);
        sps.decon.resize(n_queues);
        sps.feat_params.resize(n_queues);
    }
    const auto earliest_label_horizon = label_times.front()->get_value_time() - horizon_duration;
    const auto latest_label_horizon = label_times.back()->get_value_time() - horizon_duration;
    LOG4_TRACE("Preparing level " << level << ", " << n_rows << " rows, main range from " << earliest_label_horizon << " until " << latest_label_horizon << ", " << n_queues <<
        " queues, max quant " << PROPS.get_max_quant());
    OMP_PAR(n_queues * n_rows * sp.front().n_levels)
    {
        OMP_TASKLOOP_1()
        for (DTYPE(steps) s = 0; s < steps; ++s) {
            auto &sps = sp[s];
            OMP_TASKLOOP_1()
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues may have different amount of samples, fix the assumption here that they are the same!
            const auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_horizon);
                const auto start_iter = lower_bound_before(std::as_const(*p_queue), earliest_label_horizon) - sps.coef_lag_max_q;
                assert(start_iter >= p_queue->cbegin() && start_iter < p_queue->cend());
            const uint32_t start_offset = start_iter - p_queue->cbegin();
                sps.in_rows[qix] = last_iter - start_iter;
                const size_t quantise_features_size = n_rows * sps.coef_lag_ * sizeof(double) + sps.in_rows[qix] * sizeof(double) + 2 * n_rows * sizeof(uint32_t) + sps.in_rows[qix] * sizeof(uint32_t);
                const uint16_t n_chunks_quantise = cdivi(quantise_features_size, max_gpu_chunk_size);
                sps.chunk_len_quantise[qix] = cdiv(n_rows, n_chunks_quantise);
                sps.decon[qix].set_size(sps.in_rows[qix], sps.n_levels);
                sps.feat_params[qix].resize(n_rows);
                OMP_TASKLOOP_(sps.in_rows[qix] * sps.n_levels, SSIMD firstprivate(start_offset, qix) collapse(2))
                for (DTYPE(sps.in_rows)::value_type r = 0; r < sps.in_rows[qix]; ++r)
                    for (DTYPE(sps.n_levels) l = 0; l < sps.n_levels; ++l)
                        sps.decon[qix](r, l) = p_queue->at(start_offset + r)->at(sps.adjacent_levels ^ l);
            OMP_TASKLOOP_(n_rows, SSIMD firstprivate(n_rows))
                for (DTYPE(n_rows) r = 0; r < n_rows; ++r)
                    sps.feat_params[qix][r].ix_end = (lower_bound_before(*p_queue, label_times[r]->get_value_time() - horizon_duration) - p_queue->cbegin()) - start_offset;
            }
        }
    }

    const auto tune_data_fun = [&, steps, label_len, n_rows, n_queues](CRPTRd x, RPTR(double) f) {
        arma::mat labels;
        *f = align_data(steps, n_rows, n_queues, x, sp, label_ixs, ix_F_end, labels_aux_in, label_len, labels, nullptr);
    };

    arma::mat bounds;
    if (steps > 1) {
        bounds.set_size(steps + sp.front().n_levels_queues, 2);
        bounds.submat(0, 0, steps - 1, 0).fill(PROPS.get_min_step());
        bounds.submat(0, 1, steps - 1, 1).fill(PROPS.get_max_step());
        bounds.submat(steps, 0, bounds.n_rows - 1, 0).fill(PROPS.get_min_quant());
        bounds.submat(steps, 1, bounds.n_rows - 1, 1).fill(PROPS.get_max_quant());
    } else {
        bounds.set_size(sp.front().n_levels_queues, 2);
        bounds.col(0).fill(PROPS.get_min_quant());
        bounds.col(1).fill(PROPS.get_max_quant());
    }
    const optimizer::t_pprune_res res = optimizer::pprune(
            optimizer::pprune::C_default_algo, PROPS.get_tune_data_pop(), bounds, tune_data_fun, PROPS.get_tune_data_iter(), 0, 0, {}, {}, common::iter_depth(PROPS.get_tune_data_iter()));

    std::vector<datamodel::t_feature_mechanics *> fm(steps);
    OMP_FOR_i(steps) {
        const auto &sps = sp[i];
        fm[i] = &params[i]->set_feature_mechanics(DTYPE(params[i]->get_feature_mechanics()){
                {sps.n_levels_queues, ARMA_DEFAULT_FILL},
                {sps.n_levels_queues * sps.lag, ARMA_DEFAULT_FILL},
                std::deque<arma::uvec>(sps.n_levels_queues),
                {sps.n_levels_queues * sps.lag, ARMA_DEFAULT_FILL},
                {steps, ARMA_DEFAULT_FILL}
        });
    }

    (void) align_data(steps, n_rows, n_queues, res.best_parameters.mem, sp, label_ixs, ix_F_end, labels_aux_in, label_len, all_labels, fm.data());
    OMP_FOR_i(steps) {
        const auto &sps = sp[i];
        if (!out_features[i]) out_features[i] = ptr<arma::mat>();
        do_features(*out_features[i], n_rows, sps.lag, sps.coef_lag, sps.coef_lag_, *fm[i], sps.stripe_period, sps.chunk_len_quantise, sps.in_rows, sps.decon, sps.feat_params, sps.adjacent_levels);
    }
    LOG4_END();
}

double ModelService::align_data(
        const uint16_t steps, const uint32_t n_rows, const uint16_t n_queues, CRPTRd x, const std::deque<t_step_params> &sp, const std::vector<t_label_ix> &label_ixs,
                    const std::vector<uint32_t> &ix_F_end, const std::vector<double> &labels_aux_in, const uint32_t label_len, arma::mat &labels, datamodel::t_feature_mechanics **p_fm)
{
    uint32_t *step_ixs;
    float *points;
    if (steps > 1) {
        step_ixs = CAST2(step_ixs) malloc(steps * sizeof(float));
        points = CAST2(points) malloc(steps * sizeof(float));
        for (DTYPE(steps) i = 0; i < steps; ++i) {
            assert(x[i] > 0);
            points[i] = x[i];
            if (i) points[i] += points[i - 1];
        }
        const auto sum = points[steps - 1];
        for (DTYPE(steps) i = 0; i < steps; ++i) {
            points[i] /= sum;
            const auto ns = x[i] / sum;
            if (p_fm) p_fm[i]->steps[i] = ns;
            step_ixs[i] = label_len * ns;
        }
    } else {
        step_ixs = nullptr;
        points = nullptr;
    }
    labels.set_size(label_ixs.size(), 1);
    PROFILE_TRACE(quantise_labels(label_len, labels_aux_in, label_ixs, ix_F_end, labels.memptr(), steps, points, step_ixs), "Quantise labels");
    if (steps > 1) {
        free(points);
        free(step_ixs);
    }
    assert(!labels.has_nonfinite());

    std::atomic<double> total_score = 0;
                tbb::mutex ins_l;
OMP_PAR(common::gpu_handler_4::get().get_max_gpu_threads())
    {
        OMP_TASKLOOP_1()
        for (DTYPE(steps) step = 0; step < steps; ++step) {
            const auto &sps = sp[step];
            OMP_TASKLOOP_1()
            for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
                OMP_TASKLOOP_1(firstprivate(qix, n_rows))
                for (DTYPE(sps.n_levels) lix = 0; lix < sps.n_levels; ++lix) {
                    const auto level = sps.adjacent_levels ^ lix;
                    const auto lqix = lix + qix * sps.n_levels;
                    const uint32_t quantise = steps > 1 ? x[steps + lqix] : x[lqix];
                    const auto coef_lag_q = sps.coef_lag_ * quantise;
                    auto feat_params_qix = sps.feat_params[qix];
                    OMP_FOR(n_rows)
                    for (auto &fp: feat_params_qix) { fp.ix_start = fp.ix_end - coef_lag_q + 1; }

                    arma::mat features(n_rows, sps.coef_lag, ARMA_DEFAULT_FILL);

                    for (DTYPE(n_rows) i = 0; i < n_rows; i += sps.chunk_len_quantise[qix]) PROFILE_INFO(quantise_features(
                            sps.decon[qix].mem, feat_params_qix.data(), i, std::min<uint32_t>(i + sps.chunk_len_quantise[qix], n_rows) - i, n_rows, sps.in_rows[qix], level,
                            sps.coef_lag_, sps.coef_lag, quantise, features.memptr()), "Quantise features " << sps.chunk_len_quantise[qix] << ", quantise " << quantise);
                    RELEASE_CONT(feat_params_qix);
                    double stub_sf, stub_dc;
                    ScalingFactorService::scale_calc_I(features, stub_sf, stub_dc);

                    arma::vec scores(sps.coef_lag, ARMA_DEFAULT_FILL);
                    if (p_fm) {
                        arma::fvec stretches(sps.coef_lag, ARMA_DEFAULT_FILL);
                        arma::u32_vec shifts(sps.coef_lag, ARMA_DEFAULT_FILL);
                        // OMP_TASKLOOP_1(firstprivate(n_rows, quantise))
                        for (DTYPE(sps.coef_lag) i = 0; i < sps.coef_lag; i += sps.chunk_len_align) PROFILE_INFO(align_features(
                                features.colptr(i), labels.colptr(step), scores.memptr() + i, stretches.memptr() + i, shifts.memptr() + i, n_rows,
                                std::min<uint32_t>(i + sps.chunk_len_align, sps.coef_lag) - i), "Align features " << n_rows << "x" << sps.chunk_len_align);
                        const arma::uvec trims = arma::uvec(arma::stable_sort_index(scores)).tail(sps.coef_lag - sps.lag);
                        const auto lqix_1 = lqix + 1;
                        LOG4_DEBUG("Quantise " << quantise << ", aux queue " << qix << ", level " << lix << ":" << level << ", lag " << sps.lag << ", coef lag " << sps.coef_lag);
                        p_fm[step]->quantization[lqix] = quantise;
                        stretches.shed_rows(trims);
                        shifts.shed_rows(trims);
                        const auto lqix_lag = lqix * sps.lag;
                        const auto lqix_1_lag = lqix_1 * sps.lag - 1;
                        p_fm[step]->stretches.rows(lqix_lag, lqix_1_lag) = stretches;
                        p_fm[step]->shifts.rows(lqix_lag, lqix_1_lag) = shifts;
                        p_fm[step]->trims[lqix] = trims;
                    } else {
                        // OMP_TASKLOOP_1(firstprivate(n_rows, quantise))
                        for (DTYPE(sps.coef_lag) i = 0; i < sps.coef_lag; i += sps.chunk_len_align) PROFILE_INFO(align_features(
                                features.colptr(i), labels.colptr(step), scores.memptr() + i, nullptr, nullptr, n_rows, std::min<uint32_t>(i + sps.chunk_len_align, sps.coef_lag) - i),
                                                                                                                 "Align features " << n_rows << "x" << sps.chunk_len_align);
                        scores.shed_rows(arma::uvec(arma::stable_sort_index(scores)).tail(sps.coef_lag - sps.lag));
                        const double score = arma::accu(scores);
                        LOG4_DEBUG("Score " << score << ", quantise " << quantise << ", aux queue " << qix << ", level " << lix << ":" << level << ", lag " << sps.lag << ", coef lag " << sps.coef_lag);
                        total_score += score;
                    }
                }
            }
        }
    }

    return total_score;
}


void ModelService::do_features(
        arma::mat &out_features, const uint32_t n_rows, const uint32_t lag, const uint32_t coef_lag, const uint32_t coef_lag_, const datamodel::t_feature_mechanics &fm,
        const boost::posix_time::time_duration &stripe_period, const std::deque<uint32_t> &chunk_len_quantise, const std::deque<uint32_t> &in_rows, const std::deque<arma::mat> &decon,
    const std::deque<std::vector<t_feat_params> > &feat_params, const std::set<uint16_t> &adjacent_levels)
{
    const auto n_levels = adjacent_levels.size();
    const auto n_queues = feat_params.size();
    const auto levels_lag = n_levels * lag;
    const auto feature_cols = levels_lag * n_queues;
    const auto &feat_params_f = feat_params.front();
    if (out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);
    LOG4_TRACE(
        "Preparing features " << n_rows << "x" << feature_cols << ", lag " << lag << ", coef lag " << coef_lag << ", levels " << n_levels << ", queues " << n_queues << ", decon queue "
                                  << common::present(decon.front()) << ", feat params " << feat_params_f.size() << ", feat params ix_end " << feat_params_f.front().ix_end
                                  << ", stripe period " << stripe_period << ", quantisation " << fm.quantization[0] << ", stretches " << common::present(fm.stretches) << ", shifts "
                                  << common::present(fm.shifts));
    OMP_PAR(n_levels * n_queues * n_rows)
    {
        OMP_TASKLOOP_1(firstprivate(levels_lag))
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            OMP_TASKLOOP_1(firstprivate(lag, coef_lag, qix, n_levels))
            for (DTYPE(n_levels) adj_ix = 0; adj_ix < n_levels; ++adj_ix) {
                const auto adj_level = adjacent_levels ^ adj_ix;
                const auto adj_ix_q = adj_ix + qix * n_levels;
                const auto adj_ix_q_lag = adj_ix_q * lag;
                auto feat_params_qix_qt = feat_params[qix];
                const auto quantise = fm.quantization[adj_ix_q];
                const auto coef_lag_q = coef_lag_ * quantise;
                OMP_TASKLOOP_(n_rows, firstprivate(coef_lag_q))
                for (auto &f: feat_params_qix_qt) f.ix_start = f.ix_end - coef_lag_q + 1;
                arma::mat level_features(n_rows, coef_lag, ARMA_DEFAULT_FILL);
                OMP_TASKLOOP_1(firstprivate(n_rows, adj_ix, coef_lag_, coef_lag))
                for (DTYPE(n_rows) i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_INFO(
                    quantise_features(decon[qix].mem, feat_params_qix_qt.data(), i, std::min<uint32_t>(i + chunk_len_quantise[qix], n_rows) - i,
                        n_rows, in_rows[qix], adj_level, coef_lag_, coef_lag, quantise, level_features.memptr()),
                    "Prepare quantised features " << chunk_len_quantise[qix]);
                level_features.shed_cols(fm.trims[adj_ix_q]);
                out_features.cols(adj_ix_q_lag, adj_ix_q_lag + lag - 1) = level_features;
                LOG4_TRACE("Level " << adj_ix << ", queue " << qix << ", features " << common::present(level_features) << ", quantise period " << stripe_period * quantise);
            }
        }
    }
    if (out_features.empty())
        LOG4_WARN("No new data to prepare for training, features " << arma::size(out_features));
    LOG4_TRACE("Out features " << common::present(out_features));
}

void ModelService::prepare_labels(
        arma::mat &out_labels, const datamodel::data_row_container &label_times, const datamodel::datarow_crange &aux_label_data, const datamodel::SVRParameters &param,
        const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &ix_F_end, const bpt::time_duration &resolution, const bpt::time_duration &resolution_aux)
{
    LOG4_BEGIN();

    const uint32_t label_len = resolution / resolution_aux;
    const auto steps = param.get_feature_mechanics().steps.size();
    const auto p_steps = param.get_feature_mechanics().steps.mem;
    float *points;
    uint32_t *step_ixs;
    if (steps > 1) {
        points = CAST2(points) malloc(steps * sizeof(float));
        step_ixs = CAST2(step_ixs) malloc(steps * sizeof(uint32_t));
        for (DTYPE(steps) i = 0; i < steps; ++i) {
            step_ixs[i] = p_steps[i] * label_len;
            points[i] = p_steps[i];
            if (i) points[i] += points[i - 1];
        }
    } else {
        step_ixs = const_cast<uint32_t *>(&label_len);
        points = nullptr;
    }

    const auto level = param.get_decon_level();
    LOG4_TRACE("Preparing level " << level << ", labels " << label_times.size());
    std::vector<double> labels_aux_in(aux_label_data.distance());
    OMP_FOR_i(aux_label_data.distance()) labels_aux_in[i] = aux_label_data[i]->at(level);

    PROFIL3(quantise_labels(label_len, labels_aux_in, label_ixs, ix_F_end, out_labels.memptr(), steps, points, step_ixs));
}

void
ModelService::prepare_features(
    arma::mat &out_features, const datamodel::data_row_container &label_times, const std::deque<datamodel::DeconQueue_ptr> &feat_queues, const datamodel::SVRParameters &param,
    const bpt::time_duration &resolution_aux, const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();

    const auto horizon_duration = main_queue_resolution * PROPS.get_prediction_horizon();
    const auto earliest_label_horizon = label_times.front()->get_value_time() - horizon_duration;
    const auto latest_label_horizon = label_times.back()->get_value_time() - horizon_duration;

    const uint32_t n_rows = label_times.size();
    const auto lag = param.get_lag_count();
    const auto &adjacent_levels = param.get_adjacent_levels();
    const uint32_t coef_lag = PROPS.get_lag_multiplier() * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
#define coef_lag_ coef_lag
#endif
    const uint16_t n_levels = adjacent_levels.size();
    const uint16_t n_queues = feat_queues.size();
    arma::vec best_score(n_levels, arma::fill::value(std::numeric_limits<double>::infinity()));
    const auto stripe_period = resolution_aux * coef_lag_;
    const auto coef_lag_max_q = coef_lag_ * PROPS.get_max_quant();
    LOG4_TRACE("Preparing level " << param.get_decon_level() << ", " << n_rows << " rows, main range from " << earliest_label_horizon << " until " << latest_label_horizon <<
        ", lag " << lag << ", " << n_queues << " queues, " << n_levels << " levels, stripe period " << stripe_period);

    std::deque<uint32_t> chunk_len_quantise(n_queues), in_rows(n_queues);
    std::deque<arma::mat> decon(n_queues);
    std::deque<std::vector<t_feat_params> > feat_params(n_queues);
    const auto &fm = param.get_feature_mechanics();
    const auto max_gpu_data_chunk_size = common::gpu_handler_4::get().get_max_gpu_data_chunk_size();
    OMP_PAR(std::min<uint32_t>(common::gpu_handler_4::get().get_max_gpu_threads(), n_queues * n_levels * n_rows))
    {
        OMP_TASKLOOP_1(firstprivate(n_rows, n_levels))
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues have different amount of samples, fix assumption they are same!
            const auto start_iter = lower_bound_before(std::as_const(*p_queue), earliest_label_horizon) - coef_lag_max_q;
            if (start_iter == p_queue->cend())
                LOG4_THROW("Start offset invalid");
            auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_horizon);
            if (last_iter != p_queue->cend()) ++last_iter;
            const uint32_t start_offset = start_iter - p_queue->cbegin();
            in_rows[qix] = last_iter - start_iter;
            const size_t quantise_features_size = n_rows * coef_lag_ * sizeof(double) + in_rows[qix] * sizeof(double) + (2 * n_rows + in_rows[qix]) * sizeof(uint32_t);
            const uint16_t n_chunks_quantise = cdivi(quantise_features_size, max_gpu_data_chunk_size);
            chunk_len_quantise[qix] = cdivi(n_rows, n_chunks_quantise);
            decon[qix].set_size(in_rows[qix], n_levels);
            feat_params[qix].resize(n_rows);
            LOG4_TRACE("Queue " << qix << ", start offset " << start_offset << ", in rows " << in_rows[qix] << ", quantise features size " << quantise_features_size <<
                ", chunks " << n_chunks_quantise << ", chunk rows " << chunk_len_quantise[qix]);
            OMP_TASKLOOP_(n_levels * in_rows[qix], firstprivate(n_levels, start_offset) SSIMD untied collapse(2))
            for (DTYPE(in_rows)::value_type r = 0; r < in_rows[qix]; ++r)
                for (DTYPE(n_levels) l = 0; l < n_levels; ++l)
                    decon[qix](r, l) = p_queue->at(start_offset + r)->at(adjacent_levels ^ l);

            OMP_TASKLOOP_(n_rows, SSIMD firstprivate(n_rows) untied)
            for (DTYPE(n_rows) r = 0; r < n_rows; ++r)
                feat_params[qix][r].ix_end = (lower_bound_before(*p_queue, label_times[r]->get_value_time() - horizon_duration) - p_queue->cbegin()) - start_offset;
        }
    }

    if (const auto feature_cols = n_levels * lag * n_queues; out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);
    do_features(out_features, n_rows, lag, coef_lag, coef_lag_, fm, stripe_period, chunk_len_quantise, in_rows, decon, feat_params, adjacent_levels);

    LOG4_END();
}


datamodel::t_model_train_data ModelService::train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model)
{
    const auto [p_features, p_labels, p_last_knowns, p_weights, p_times] = get_training_data(dataset, ensemble, model.get_decon_level());
    const auto last_value_time = p_times->back()->get_value_time();
    if (model.get_last_modeled_value_time() >= last_value_time) {
        LOG4_DEBUG("No new data to train model " << model << ", last modeled time " << model.get_last_modeled_value_time() << ", last value time " << last_value_time);
        return {};
    }
    if (last_value_time < model.get_last_modeled_value_time()) {
        LOG4_ERROR("Data is older " << last_value_time << " than last modeled time " << model.get_last_modeled_value_time());
        return {};
    }
    datamodel::t_model_train_data res;
    if (model.get_last_modeled_value_time() == bpt::min_date_time) {
#ifdef INTEGRATION_TEST
        res = std::make_tuple(ptr(*p_features), ptr(*p_labels), ptr(*p_last_knowns), ptr(*p_weights), otr(*p_times));
        const auto n_rows = p_labels->n_rows;
        const auto train_rows = n_rows - common::C_integration_test_validation_window;
        p_labels->shed_rows(train_rows, n_rows - 1);
        p_features->shed_rows(train_rows, n_rows - 1);
        p_last_knowns->shed_rows(train_rows, n_rows - 1);
        p_weights->shed_rows(train_rows, n_rows - 1);
        p_times->erase(p_times->begin() + train_rows, p_times->end());
#else
        res = std::make_tuple(p_features, p_labels, p_last_knowns, p_weights, p_times);
#endif
        train_batch(model, p_features->at(model.get_step()), p_labels, p_weights, last_value_time);
    } else {
        train_online(model, *p_features->at(model.get_step()), *p_labels, *p_weights, last_value_time);
        res = std::make_tuple(p_features, p_labels, p_last_knowns, p_weights, p_times);
    }
    model.set_last_modeled_value_time(last_value_time);
    model.set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO("Finished training model " << model);
    return res;
}


void
ModelService::train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::mat &weights, const bpt::ptime &last_value_time)
{
    arma::mat residuals, learn_labels = labels;
    UNROLL()
    for (uint16_t g = 0; g < model.get_gradient_count(); ++g) {
        const bool is_gradient = g < model.get_gradient_count() - 1;
        const auto &m = model.get_gradient(g);
        if (is_gradient) residuals = learn_labels - m->predict(features, last_value_time);

        const uint32_t temp_learn =
#ifdef LAST_KNOWN_LABEL
        !is_gradient ? learn_labels.n_rows - 1 :
#endif
        std::numeric_limits<DTYPE(temp_learn)>::max();

        PROFILE_INFO(m->learn(features, learn_labels, weights, last_value_time, temp_learn), "Online SVM train gradient " << g);

        if (is_gradient) learn_labels = residuals;
    }
}


void
ModelService::train_batch(
    datamodel::Model &model,
    const mat_ptr &p_features,
    const mat_ptr &p_labels,
    const mat_ptr &p_weights,
    const bpt::ptime &last_value_time)
{
    LOG4_BEGIN();

    datamodel::t_gradient_data gradient_data(p_features, p_labels);
    UNROLL()
    for (uint16_t gix = 0; gix < model.get_gradient_count(); ++gix) {
        const auto p_gradient = model.get_gradient(gix);
        if (!p_gradient)
            LOG4_THROW("SVR model for gradient " << gix << " not initialized " << model);
        PROFILE_INFO(p_gradient->batch_train(gradient_data.p_features, gradient_data.p_labels, p_weights, last_value_time),
                     "Train batch, gradient " << gix << ", labels " << arma::size(*gradient_data.p_labels) << ", features " << arma::size(*gradient_data.p_features) << ", last value time "
                                              << last_value_time);

        if (model.get_gradient_count() < 2 || gix == model.get_gradient_count() - 1) continue;

        gradient_data = model.get_gradient(gix)->produce_residuals();
        for (auto &p: model.get_gradient(gix + 1)->get_param_set())
            p->set_svr_decremental_distance(gradient_data.p_features->n_rows);
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble &ensemble, const uint16_t level, const datamodel::data_row_container &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size(), arma::fill::none);
    const auto p_aux_decon = ensemble.get_label_aux_decon();
    if (!p_aux_decon || p_aux_decon->empty())
        LOG4_THROW("No label auxiliary data for ensemble " << ensemble);
    const auto horizon_duration = resolution * PROPS.get_prediction_horizon();
    OMP_FOR_i_(res.size(), firstprivate(level)) {
        const auto &row = **business::lower_bound_before(*p_aux_decon, times[i]->get_value_time() - horizon_duration);
        res[i] = row[level];
        LOG4_TRACE("For time " << times[i]->get_value_time() << ", found last known " << *row);
    }
    return res;
}


void ModelService::predict(
    const datamodel::Ensemble &ensemble,
    datamodel::Model &model,
    const datamodel::t_level_predict_features &predict_features,
    const bpt::time_duration &resolution,
    tbb::mutex &insemx,
    datamodel::data_row_container &out)
{
    assert(model.get_gradients().size() > 0);
    arma::mat prediction(predict_features.p->n_rows, model.get_outputs());
    tbb::mutex predict_lock;
    const auto predict_time = predict_features.times.front()->get_value_time();
    OMP_FOR(model.get_gradient_count())
    for (const auto &p_svr: model.get_gradients()) {
        const auto this_prediction = p_svr->predict(*predict_features.p, predict_time);
        const tbb::mutex::scoped_lock lk(predict_lock);
        prediction += this_prediction;
    }
#ifdef EMO_DIFF
    const auto lk = get_last_knowns(ensemble, model.get_decon_level(), predict_features.times, resolution);
    OMP_FOR_i(prediction.n_cols) prediction.col(i) += lk;
#endif
    const tbb::mutex::scoped_lock lck(insemx);
    datamodel::DataRow::insert_rows(out, prediction, predict_features.times, model.get_decon_level(), ensemble.get_level_ct(), true);
    LOG4_TRACE("Predicted " << common::present(prediction) << " for " << predict_features.times.size() << " times, container " << common::to_string(out));
}

#ifdef INTEGRATION_TEST

void ModelService::predict(
    const datamodel::Ensemble &ensemble,
    datamodel::Model &model,
    const datamodel::t_level_predict_features &predict_features,
    const bpt::time_duration &resolution,
    tbb::mutex &insemx,
    const arma::mat &labels,
    datamodel::data_row_container &out)
{
    assert(model.get_gradients().size() > 0);
    arma::mat prediction(predict_features.p->n_rows, model.get_outputs());
    const auto predict_time = predict_features.times.front()->get_value_time();
    tbb::mutex predict_lock;
    OMP_FOR(model.get_gradient_count())
    for (const auto &p_svr: model.get_gradients()) {
        const auto this_prediction = p_svr->predict(*predict_features.p, labels, predict_time);
        const tbb::mutex::scoped_lock lk(predict_lock);
        prediction += this_prediction;
    }
#ifdef EMO_DIFF
    const auto lk = get_last_knowns(ensemble, model.get_decon_level(), predict_features.times, resolution);
    OMP_FOR_i(prediction.n_cols) prediction.col(i) += lk;
#endif
    const tbb::mutex::scoped_lock lck(insemx);
    datamodel::DataRow::insert_rows(out, prediction, predict_features.times, model.get_decon_level(), ensemble.get_level_ct(), true);
#ifndef NDEBUG
    LOG4_TRACE("Predicted " << common::present(prediction) << " for " << predict_features.times.size() << " times, container " << common::to_string(out));
#endif
}

#endif

void
ModelService::check_feature_data(
    const datamodel::DataRow::container &data,
    const datamodel::DataRow::container::const_iterator &iter,
    const bpt::time_duration &max_gap,
    const bpt::ptime &feat_time,
    const ssize_t lag_count)
{
    if (iter == data.end() || iter->get()->get_value_time() - feat_time > max_gap ||
        std::distance(data.begin(), iter) < lag_count) // We don't have lag count data
        THROW_EX_FS(common::insufficient_data,
                "Can't find data for prediction features. Need " + std::to_string(lag_count) + " values until " +
                bpt::to_simple_string(feat_time) +
                ", data available is from " + bpt::to_simple_string(data.front()->get_value_time()) + " until " +
                bpt::to_simple_string(data.back()->get_value_time()));
}

void ModelService::check_feature_data(
    const datamodel::DataRow::container &data, const datamodel::DataRow::container::const_iterator &iter, const bpt::time_duration &max_gap, const bpt::ptime &feat_time)
{
    if (iter == data.end() || iter->get()->get_value_time() - feat_time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                "Can't find data for prediction features. Needed value for " + bpt::to_simple_string(feat_time) +
                ", nearest data available is " +
                (iter == data.end() ? "not found" : "at " + bpt::to_simple_string((**iter).get_value_time())));
}


void ModelService::init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble &ensemble) const
{
    if (!check(ensemble.get_models(), p_dataset->get_model_count()) && ensemble.get_id())
        ensemble.set_models(model_dao.get_all_ensemble_models(ensemble.get_id()), false);
    tbb::mutex init_models_l;
    OMP_FOR_(p_dataset->get_model_count() * p_dataset->get_steps(), SSIMD collapse(2))
    for (uint16_t levix = 0; levix < p_dataset->get_spectral_levels(); levix += LEVEL_STEP)
        for (uint16_t stepix = 0; stepix < p_dataset->get_steps(); ++stepix)
            if (levix != p_dataset->get_trans_levix()) {
                tbb::mutex::scoped_lock lk(init_models_l);
                auto p_model = ensemble.get_model(levix, stepix);
                lk.release();
                if (!p_model) {
                    p_model = ptr<datamodel::Model>(0, ensemble.get_id(), levix, stepix, PROPS.get_outputs(), p_dataset->get_gradient_count(), p_dataset->get_max_chunk_size());
                    const tbb::mutex::scoped_lock lk2(init_models_l);
                    ensemble.get_models().emplace_back(p_model);
                }
                configure(p_dataset, ensemble, *p_model);
            }
}
} // business
} // svr
