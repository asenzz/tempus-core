#include "DeconQueueService.hpp"
#include <DAO/EnsembleDAO.hpp>
#include <atomic>
#include <execution>
#include <model/Ensemble.hpp>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"
#include "util/string_utils.hpp"
#include "cuqrsolve.cuh"
#include "firefly.hpp"
#include "common/constants.hpp"
#include "model/SVRParameters.hpp"
#include <tuple>
#include <string>
#include <deque>
#include <complex>
#include <cmath>
#include <algorithm>
#include <utility>
#include <memory>
#include <iterator>
#include <limits>
#include <cstdlib>
#include <armadillo>
#include <boost/date_time/posix_time/ptime.hpp>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_unordered_set.h>
#include <iostream>

#include "SVRParametersService.hpp"
#include "common/rtp_thread_pool.hpp"
#include "DAO/ModelDAO.hpp"
#include "model/Model.hpp"
#include "util/time_utils.hpp"
#include "util/validation_utils.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "model/User.hpp"
#include "DAO/DatasetDAO.hpp"
#include "common/defines.h"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/logging.hpp"
#include "onlinesvr.hpp"
#include "DQScalingFactorService.hpp"
#include "ModelService.hpp"
#include "appcontext.hpp"
#include "align_features.cuh"

namespace svr {
namespace business {

ModelService::ModelService(svr::dao::ModelDAO &model_dao) : model_dao(model_dao)
{}


size_t ModelService::to_level_ix(const size_t model_ix, const size_t level_ct)
{
    if (model_ix >= MIN_LEVEL_COUNT)
        return ((model_ix >= (level_ct / 4)) ? (model_ix + 1) : model_ix) * 2;
    else
        return 0;
}

size_t ModelService::to_level_ct(const size_t model_ct)
{
    if (model_ct >= MIN_LEVEL_COUNT / 2 - 1)
        return model_ct * 2 + 2;
    else
        return 1;
}

size_t ModelService::to_model_ct(const size_t level_ct)
{
    return level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 - 1 : 1;
}

size_t ModelService::to_model_ix(const size_t level_ix, const size_t level_ct)
{
    if (level_ct < MIN_LEVEL_COUNT)
        return 0;
    if (level_ix == level_ct / 2)
        LOG4_THROW("Illegal level index " << level_ix << ", of count " << level_ct);

    return level_ix / 2 - (level_ix > level_ct / 2 ? 1 : 0);
}

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
ModelService::validate(
        const size_t start_ix,
        const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model,
        const arma::mat &features, const arma::mat &labels, const arma::mat &last_knowns, const std::deque<bpt::ptime> &times,
        const bool online, const bool verbose)
{
    LOG4_BEGIN();
    if (labels.n_rows <= start_ix) {
        LOG4_WARN("Calling future validate at the end of labels array. MAE is 1000");
        return {common::C_bad_validation, common::C_bad_validation, {}, {}, 0., {}};
    }

    const size_t ix_fini = labels.n_rows - 1;
    const size_t num_preds = 1 + ix_fini - start_ix;
    const auto params = model.get_head_params();
    const auto level = params->get_decon_level();

    datamodel::t_level_predict_features predict_features{{times.cbegin() + start_ix, times.cend()}, ptr<arma::mat>(features.rows(start_ix, ix_fini))};
    LOG4_TRACE("Predicting features " << common::present<double>(*predict_features.p_features));
    data_row_container batch_predicted, cont_predicted_online;
    tbb::mutex mx;
    PROFILE_EXEC_TIME(ModelService::predict(
            ensemble, model, predict_features, dataset.get_input_queue()->get_resolution(), mx, batch_predicted), "Batch predict of " << num_preds << " rows, level " <<
                                                                                                                                      level << ", step "
                                                                                                                                      << model.get_step());
    if (batch_predicted.size() != num_preds)
        LOG4_THROW("Predicted size " << batch_predicted.size() << " not sane " << arma::size(*predict_features.p_features));

    LOG4_DEBUG("Batch predicted " << batch_predicted.size() << " values, parameters " << *params);
    const auto stepping = model.get_gradient()->get_dataset()->get_multistep();
    arma::vec predicted(num_preds), predicted_online(num_preds),
            actual = arma::mean(labels.rows(start_ix, ix_fini), 1),
            lastknown = arma::mean(last_knowns.rows(start_ix, ix_fini), 1);
#ifdef EMO_DIFF
#pragma omp parallel for simd num_threads(adj_threads(actual.n_cols))
    for (size_t i = 0; i < actual.n_cols; ++i) actual.col(i) = actual.col(i) + lastknown; // common::sexp<double>(actual.col(i)) + lastknown;
#endif
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    double batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (size_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted[ix] = stepping * batch_predicted[ix]->at(level);
        const double cur_absdiff_lk = std::abs(lastknown[ix] - actual[ix]);
        const double cur_absdiff_batch = std::abs(predicted[ix] - actual[ix]);
        const double cur_alpha_pct_batch = 100. * (1. - cur_absdiff_batch / cur_absdiff_lk);
        sum_abs_labels += std::abs(actual[ix]);
        sum_absdiff_batch += cur_absdiff_batch;
        sum_absdiff_lk += std::abs(actual[ix] - lastknown[ix]);
        batch_correct_predictions += cur_absdiff_batch < cur_absdiff_lk;
        batch_correct_directions += std::signbit(predicted[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);

        const auto ix_div = ix + 1.;
        const bool print_line = verbose || ix_future == ix_fini || ix % 115 == 0;
        std::stringstream row_report;
        if (print_line)
            row_report << "Position " << ix << ", level " << level << ", step " << model.get_step() << ", actual " << actual[ix] << ", batch predicted " << predicted[ix]
                       << ", last known " << lastknown[ix] << " batch MAE " << sum_absdiff_batch / ix_div << ", MAE last-known " << sum_absdiff_lk / ix_div
                       << ", batch MAPE "
                       << 100. * sum_absdiff_batch / sum_abs_labels << "pc, MAPE last-known " << 100. * sum_absdiff_lk / sum_abs_labels << "pc, batch alpha "
                       << 100. * (1. - sum_absdiff_batch / sum_absdiff_lk) << "pc, current batch alpha " << cur_alpha_pct_batch << "pc, batch correct predictions "
                       << 100. * batch_correct_predictions / ix_div << "pc, batch correct directions " << 100. * batch_correct_directions / ix_div << "pc";

        if (online) {
            PROFILE_EXEC_TIME(
                    ModelService::predict(
                            ensemble, model, datamodel::t_level_predict_features{{times[ix]}, ptr<arma::mat>(features.row(ix_future))},
                            dataset.get_input_queue()->get_resolution(), mx, cont_predicted_online),
                    "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                                      ", step " << model.get_step() << " at " << times[ix_future]);
            PROFILE_EXEC_TIME(
                    ModelService::train_online(model, features.row(ix_future), labels.row(ix_future), last_knowns.row(ix_future), times[ix_future]),
                    "Online learn " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                                    ", step " << model.get_step() << " at " << times[ix_future]);
            predicted_online[ix] = stepping * cont_predicted_online.front()->at(level);
#ifdef EMO_DIFF
            predicted_online[ix] = predicted_online[ix] + lastknown[ix]; // common::sexp(predicted_online[ix]) + lastknown[ix];
#endif
            const double cur_absdiff_online = std::abs(predicted_online[ix] - actual[ix]);
            const double cur_alpha_pct_online = 100. * (cur_absdiff_lk / cur_absdiff_online - 1.);
            sum_absdiff_online += cur_absdiff_online;
            online_correct_predictions += cur_absdiff_online < cur_absdiff_lk;
            online_correct_directions += std::signbit(predicted_online[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);
            if (print_line)
                row_report << ", online predicted " << predicted_online[ix] << ", online MAE " << sum_absdiff_online / ix_div << ", online MAPE " <<
                           100. * sum_absdiff_online / sum_abs_labels << "pc, online alpha " << 100. * (sum_absdiff_lk / sum_absdiff_online - 1.)
                           << "pc, current online alpha " << cur_alpha_pct_online << "pc, online correct predictions " << 100. * online_correct_predictions / ix_div
                           << "pc, online correct directions " << 100. * online_correct_directions / ix_div << "pc";
        }
        if (row_report.str().size()) LOG4_DEBUG(row_report.str());
    }
    const auto mape_lk = 100. * sum_absdiff_lk / sum_abs_labels;
    if (online)
        return {sum_absdiff_online / double(num_preds), 100. * sum_absdiff_online / sum_abs_labels, predicted_online, actual, mape_lk, lastknown};
    else
        return {sum_absdiff_batch / double(num_preds), 100. * sum_absdiff_batch / sum_abs_labels, predicted, actual, mape_lk, lastknown};
}


std::deque<::std::shared_ptr<::svr::datamodel::DataRow>>::const_iterator
ModelService::get_start(
        const datamodel::DataRow::container &cont,
        const size_t decremental_offset,
        const boost::posix_time::ptime &model_last_time,
        const boost::posix_time::time_duration &resolution)
{
    if (decremental_offset < 1) {
        LOG4_ERROR("Decremental offset " << decremental_offset << " returning end.");
        return cont.cend();
    }
    // Returns an iterator with the earliest value time needed to train a model with the most current data.
    LOG4_DEBUG("Size is " << cont.size() << " decrement " << decremental_offset);
    if (cont.size() <= decremental_offset) {
        LOG4_WARN("Container size " << cont.size() << " is less or equal to needed size " << decremental_offset);
        return cont.cbegin();
    } else if (model_last_time == boost::posix_time::min_date_time)
        return std::next(cont.cbegin(), cont.size() - decremental_offset);
    else
        return find_nearest(cont, model_last_time + resolution);
}


datamodel::Model_ptr ModelService::get_model_by_id(const bigint model_id)
{
    return model_dao.get_by_id(model_id);
}


datamodel::Model_ptr
ModelService::find(const std::deque<datamodel::Model_ptr> &models, const size_t levix, const size_t stepix)
{
    for (const auto &m: models)
        if (m->get_decon_level() == levix && m->get_step() == stepix)
            return m;
    LOG4_WARN("Model for level " << levix << ", step " << stepix << " not found among " << models.size() << " models.");
    return nullptr;
}

void ModelService::configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr &p_ensemble, datamodel::Model_ptr &p_model)
{
    if (!check(p_model->get_gradients(), p_model->get_gradient_count()) && p_model->get_id())
        p_model->set_gradients(model_dao.get_svr_by_model_id(p_model->get_id()), false);

    p_model->set_max_chunk_size(p_dataset->get_max_chunk_size());
    std::deque<datamodel::SVRParameters_ptr> model_params;
    if (p_dataset->get_id())
        model_params = APP.svr_parameters_service.get_by_dataset_column_level(p_dataset->get_id(), p_ensemble->get_column_name(), p_model->get_decon_level(),
                                                                              p_model->get_step());

    const size_t default_model_num_chunks = datamodel::OnlineMIMOSVR::get_num_chunks(model_params.empty() ? datamodel::C_default_svrparam_decrement_distance :
                                                                                     (**model_params.cbegin()).get_svr_decremental_distance(),
                                                                                     p_model->get_max_chunk_size());
    datamodel::dq_scaling_factor_container_t all_model_scaling_factors;
    if (p_model->get_id()) all_model_scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(p_model->get_id());

    t_omp_lock gradients_l;
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_gradient_count())) schedule(static, 1)
    for (size_t gix = 0; gix < p_dataset->get_gradient_count(); ++gix) {
        gradients_l.set();
        auto p_svr_model = p_model->get_gradient(gix);
        gradients_l.unset();

        bool set_params = false;
        size_t grad_num_chunks = 0;
        // Prepare this gradient parameters
        datamodel::t_param_set grad_params;
        if (p_svr_model) {
            grad_params = p_svr_model->get_param_set();
            grad_num_chunks = p_svr_model->get_num_chunks();
        } else {
            set_params = true;
            grad_num_chunks = default_model_num_chunks;
        }

        t_omp_lock grad_params_l;
#pragma omp parallel for num_threads(adj_threads(grad_num_chunks)) schedule(static, 1)
        for (size_t chix = 0; chix < grad_num_chunks; ++chix)
            if (SVRParametersService::slice(grad_params, chix, gix).empty()) {
                datamodel::t_param_set level_grad_param_set;
                const auto p_params = (level_grad_param_set = SVRParametersService::slice(model_params, chix, gix)).size() ?
                                      *level_grad_param_set.cbegin() :
                                      ptr<datamodel::SVRParameters>(
                                              0, p_dataset->get_id(), p_dataset->get_input_queue()->get_table_name(), p_ensemble->get_column_name(),
                                              p_dataset->get_transformation_levels(), p_model->get_decon_level(), p_model->get_step(), chix, gix);
                grad_params_l.set();
                grad_params.emplace(p_params);
                set_params = true;
                grad_params_l.unset();
            }

        if (!p_svr_model) {
            gradients_l.set();
            p_svr_model = p_model->get_gradients().emplace_back(ptr<datamodel::OnlineMIMOSVR>(0, p_model->get_id(), grad_params, p_dataset));
            gradients_l.unset();
        } else {
            if (set_params) p_svr_model->set_param_set(grad_params);
            if (!p_svr_model->get_dataset()) p_svr_model->set_dataset(p_dataset);
        }

        const auto adjacent_ct = (**grad_params.cbegin()).get_adjacent_levels().size();
        if (p_svr_model->get_scaling_factors().size() != p_svr_model->get_num_chunks()) {
#pragma omp parallel for num_threads(adj_threads(grad_num_chunks)) schedule(static, 1)
            for (size_t chix = 0; chix < grad_num_chunks; ++chix) {
                datamodel::DQScalingFactor_ptr p_sf;
                if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), p_model->get_id(), chix, p_svr_model->get_gradient_level(), p_model->get_step(),
                                                  p_model->get_decon_level(), false, true)
                    &&
                    (p_sf = DQScalingFactorService::find(all_model_scaling_factors, p_model->get_id(), chix, p_svr_model->get_gradient_level(), p_svr_model->get_step(),
                                                         p_model->get_decon_level(), false, true)))
                    p_svr_model->set_scaling_factor(p_sf);
                for (size_t levix = 0; levix < adjacent_ct; ++levix) {
                    if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), p_model->get_id(), chix, p_svr_model->get_gradient_level(),
                                                      p_svr_model->get_step(),
                                                      levix, true, false)
                        && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, p_model->get_id(), chix, p_svr_model->get_gradient_level(),
                                                                p_svr_model->get_step(),
                                                                levix, true, false)))
                        p_svr_model->set_scaling_factor(p_sf);
                }
            }
        }
    }
}


int ModelService::save(const datamodel::Model_ptr &model)
{
    common::reject_nullptr(model);
    if (!model->get_id()) model->set_id(model_dao.get_next_id());
    return model_dao.save(model);
}

bool ModelService::exists(const datamodel::Model_ptr &model)
{
    return model_dao.exists(model->get_id());
}

int ModelService::remove(const datamodel::Model_ptr &model)
{
    common::reject_nullptr(model);
    return model_dao.remove(model);
}

int ModelService::remove_by_ensemble_id(const bigint ensemble_id)
{
    return model_dao.remove_by_ensemble_id(ensemble_id);
}

std::deque<datamodel::Model_ptr> ModelService::get_all_models_by_ensemble_id(const bigint ensemble_id)
{
    return model_dao.get_all_ensemble_models(ensemble_id);
}

datamodel::Model_ptr ModelService::get_model(const bigint ensemble_id, const size_t decon_level)
{
    return model_dao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

bool ModelService::check(const std::deque<datamodel::Model_ptr> &models, const size_t model_ct)
{
    tbb::concurrent_vector<bool> present(model_ct, false);
#pragma omp parallel for num_threads(adj_threads(models.size())) schedule(static, 1)
    for (const auto &p: models) present[to_model_ix(p->get_decon_level(), to_level_ct(model_ct))] = true;
    return std::all_of(std::execution::par_unseq, present.begin(), present.end(), [](const auto p) { return p; });
}

bool ModelService::check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const size_t grad_ct)
{
    tbb::concurrent_vector<bool> present(grad_ct, false);
#pragma omp parallel for num_threads(adj_threads(models.size()))
    for (const auto &p_svr: models) present[p_svr->get_gradient_level()] = true;
    return std::all_of(std::execution::par_unseq, present.begin(), present.end(), [](const auto p) { return p; });
}

double
ModelService::get_quantized_feature(
        const size_t pos,
        const data_row_container::const_iterator &prev_end_iter,
        const size_t level,
        const double quantization_mul,
        const size_t lag)
{
    auto row_iter = prev_end_iter - (lag - pos) * quantization_mul + 1;
    double result = 0;
    for (size_t sub_pos = 0; sub_pos < quantization_mul; ++sub_pos, ++row_iter) result += (**row_iter)[level];
    result /= std::floor(quantization_mul);
#if 0 // Enable for extra caution
    if (!common::isnormalz(result))
        LOG4_WARN("Corrupt value " << result << " at " << row_iter->get()->get_value_time() << " pos " << pos << " level " <<
                    level << " lag count " << lag << " quantization mul " << quantization_mul);
#endif
    return result;
}


bool
ModelService::prepare_time_features(
        const bpt::ptime &value_time,
        arma::rowvec &row)
{
    LOG4_THROW("Not implemented!");
#if 0
    LOG4_TRACE("Processing row with value time " << value_time);

    // Add time features as one hot encoded structures

    for (size_t hour_of_day = 0; hour_of_day < 24; ++hour_of_day)
        row.add(hour_of_day == static_cast<size_t>(value_time.time_of_day().hours()) ? 1. : 0.);

    for (size_t day_of_week = 0; day_of_week < 7; ++day_of_week)
        row.add(day_of_week == static_cast<size_t>(value_time.date().day_of_week()) ? 1. : 0.);

    for (size_t day_of_month = 0; day_of_month < 31; ++day_of_month)
        row.add(day_of_month == static_cast<size_t>(value_time.date().day() - 1) ? 1. : 0.);

    for (size_t month_of_year = 0; month_of_year < 12; ++month_of_year)
        row.add(month_of_year == static_cast<size_t>(value_time.date().month() - 1) ? 1. : 0.);

    LOG4_TRACE("Row size is " << row.size());
#endif
    return true;
}

#
// TODO Rewrite dysfunctional
inline void
prepare_tick_volume_features(
        const std::deque<double>::const_iterator it_tick_volume_begin,
        const size_t lag,
        arma::rowvec &row)
{
    LOG4_THROW("Not implemented!");
#if 0
    std::deque<double>::const_iterator it{it_tick_volume_begin};
    for (size_t i = 0; i < lag; ++i, it += QUANTIZE_FIXED) row[i] = *it;
#endif
}

// Takes a decon queue and prepares feature vectors, using lag_count number of autoregressive features (and other misc features).
std::tuple<mat_ptr, mat_ptr, vec_ptr, times_ptr>
ModelService::get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, size_t dataset_rows)
{
    LOG4_BEGIN();

    const auto level = model.get_decon_level();
    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    auto &params = *model.get_head_params();
    if (!dataset_rows) dataset_rows = params.get_svr_decremental_distance() + C_test_len;
    const auto main_resolution = dataset.get_input_queue()->get_resolution();
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? main_resolution : dataset.get_aux_input_queue()->get_resolution();

    const datamodel::datarow_crange labels_range = {ModelService::get_start( // Main labels are used for timing
            label_decon, dataset_rows, model.get_last_modeled_value_time(), main_resolution), label_decon.get_data().cend(), label_decon};
    const auto [p_labels, p_last_knowns, p_label_times] = dataset.get_calc_cache().get_cached_labels(
            model.get_step(), labels_range, labels_aux, dataset.get_max_lookback_time_gap(), level, aux_resolution,
            model.get_last_modeled_value_time(), main_resolution, dataset.get_multistep());
    if (params.get_feature_mechanics().needs_tuning()) PROFILE_EXEC_TIME(
            tune_features(params, *p_labels, *p_label_times, ensemble.get_aux_decon_queues(), dataset.get_max_lookback_time_gap(), aux_resolution, main_resolution),
            "Tune features " << params);

    auto p_features = ptr<arma::mat>();
    PROFILE_EXEC_TIME(
            prepare_features(*p_features, *p_label_times, ensemble.get_aux_decon_queues(), params, dataset.get_max_lookback_time_gap(), aux_resolution, main_resolution),
            "Prepare features " << params);

    if (p_labels->n_rows != p_features->n_rows)
        LOG4_THROW("Labels size " << arma::size(*p_labels) << ", features size " << arma::size(*p_features) << " do not match.");

    return {p_features, p_labels, p_last_knowns, p_label_times};
}


void
ModelService::prepare_labels(
        arma::mat &all_labels,
        arma::vec &all_last_knowns,
        std::deque<bpt::ptime> &all_times,
        const datamodel::datarow_crange &main_data,
        const datamodel::datarow_crange &labels_aux,
        const bpt::time_duration &max_gap,
        const size_t level,
        const bpt::time_duration &aux_queue_res,
        const bpt::ptime &last_modeled_value_time,
        const bpt::time_duration &main_queue_resolution,
        const size_t multistep)
{
    LOG4_BEGIN();
    const auto main_to_aux_period_ratio = main_queue_resolution / aux_queue_res;
    const auto req_rows = main_data.distance();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    //LOG4_TRACE("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.begin()->get()->get_value_time() <<
    //                              " until " << main_data.rbegin()->get()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
    auto harvest_rows = [&](const datamodel::datarow_crange &harvest_range) {
        tbb::concurrent_unordered_set<arma::uword> shedded_rows;
        const auto expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, multistep);
        arma::vec last_knowns(expected_rows);
        tbb::concurrent_set<bpt::ptime> times;
        //LOG4_TRACE("Processing range " << (**harvest_range.begin()).get_value_time() << " to " << (**harvest_range.crbegin()).get_value_time() <<
        //                               ", expected " << expected_rows << " rows.");

#define SHED_ROW(msg) { LOG4_DEBUG(msg); shedded_rows.emplace(rowix); continue; }

        OMP_FOR(expected_rows)
        for (dtype(expected_rows) rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_time = harvest_range[rowix]->get_value_time();
            if (label_start_time <= last_modeled_value_time) SHED_ROW("Skipping already modeled row with value time " << label_start_time);

            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            const auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux(rowix * main_to_aux_period_ratio), label_start_time);
            if (label_aux_start_iter == labels_aux.contend()) SHED_ROW("Can't find aux labels start " << label_start_time)
            else if ((**label_aux_start_iter).get_value_time() >= label_start_time + .5 * main_queue_resolution) SHED_ROW(
                    "label aux start iter value time > label start time " << (**label_aux_start_iter).get_value_time() << " > "
                                                                          << label_start_time + .5 * main_queue_resolution);
            const auto label_aux_end_iter = lower_bound(labels_aux.get_container(), label_aux_start_iter, label_start_time + main_queue_resolution);
            const auto anchor_iter = lower_bound_back_before(labels_aux.get_container(), label_aux_start_iter,
                                                             label_start_time - main_queue_resolution * PROPS.get_prediction_offset());
            if (anchor_iter == labels_aux.contend()) SHED_ROW("Can't find aux labels start " << label_start_time)
            const arma::subview<double> labels_row = labels.row(rowix);
            if (!generate_twap(label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, aux_queue_res, level,
                               labels_row)) SHED_ROW("Failed generating TWAP prices for " << label_start_time)

            if (labels_row.has_nonfinite()) SHED_ROW(
                    "Sanity check of row at " << label_start_time << " failed, size " << arma::size(labels_row) << ", content " << labels_row)

            const auto p_anchor_row = *anchor_iter;
            last_knowns[rowix] = (*p_anchor_row)[level];
            times.emplace(label_start_time);
            if (ssize_t(rowix) >= harvest_range.distance() - 1)
                LOG4_TRACE(
                        "Added last data row " << rowix << ", value time " << label_start_time << ", label aux start time " << (**label_aux_start_iter).get_value_time()
                                               <<
                                               ", last known time " << p_anchor_row->get_value_time() << ", last last-known value " << last_knowns[rowix] << ", label "
                                               << labels_row << ", for level " << level);
        }
        if (shedded_rows.size()) {
            const auto ashedded_rows = common::toarmacol(shedded_rows);
            shedded_rows.clear();
            LOG4_DEBUG("Shedding rows " << ashedded_rows);
            labels.shed_rows(ashedded_rows);
            last_knowns.shed_rows(ashedded_rows);
        }
        if (labels.empty() && last_knowns.empty()) LOG4_WARN("Empty data for level " << level);
        else
            LOG4_TRACE("Returning labels " << common::present(labels) << ", last-knowns " << common::present(last_knowns) << " for level " << level);
        return std::tuple{labels, last_knowns, times};
    };

    if (all_labels.n_cols != multistep) all_labels.zeros(all_labels.n_rows, multistep);

    for (auto harvest_range = main_data;
         dtype(req_rows)(all_labels.n_rows) < req_rows
         && (**harvest_range.begin()).get_value_time() >= (**main_data.contbegin()).get_value_time()
         && (**harvest_range.begin()).get_value_time() > last_modeled_value_time
         && harvest_range.begin() != harvest_range.end();
         harvest_range.set_range(harvest_range(all_labels.n_rows - req_rows), harvest_range.begin())) {
        const auto [labels, last_knowns, times] = harvest_rows(harvest_range);
        all_labels = arma::join_cols(labels, all_labels);
        all_last_knowns = arma::join_cols(last_knowns, all_last_knowns);
        all_times.insert(all_times.cbegin(), times.cbegin(), times.cend());
    }

#ifdef LAST_KNOWN_LABEL
    // Add last known value if preparing online train
    if (last_modeled_value_time > bpt::min_date_time) {
        auto label_aux_start_iter = lower_bound(
                labels_aux.get_container(), labels_aux.it(main_data.distance() * .5),
                (**(main_data.begin() + main_data.distance() - 1)).get_value_time() + main_queue_resolution + main_queue_resolution * (1. - PROPS.get_prediction_offset()));
        if (label_aux_start_iter == labels_aux.contend()) --label_aux_start_iter;
        const bpt::ptime label_start_time = (**label_aux_start_iter).get_value_time();
        arma::rowvec labels_row(multiout);
        labels_row.fill((**label_aux_start_iter)[level]);
        const auto &anchor_row = **std::prev(lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * PROPS.get_prediction_offset()));
        all_labels = arma::join_cols(all_labels, labels_row);
        all_last_knowns = arma::join_cols(all_last_knowns, arma::rowvec(anchor_row[level]));
        all_times.emplace_back(label_start_time);
        LOG4_DEBUG("Temporary data last row, time " << label_start_time << " anchor time " << anchor_row.get_value_time());
    }
#endif

#ifdef EMO_DIFF
#pragma omp parallel for num_threads(adj_threads(all_labels.n_cols)) schedule(static, 1)
    for (size_t i = 0; i < all_labels.n_cols; ++i)
        all_labels.col(i) = all_labels.col(i) - all_last_knowns; // common::slog<double>(all_labels.col(i) - all_last_knowns);
#endif

    if (all_labels.empty() or all_last_knowns.empty())
        LOG4_WARN("No new data to prepare for training, labels " << arma::size(all_labels) << ", last-knowns " << arma::size(all_last_knowns));
    else
        LOG4_TRACE("Prepared level " << level << ", labels " << common::present(all_labels) << ", last-knowns " << common::present(all_last_knowns));
}

namespace {
const auto C_quantizations = [](){
    std::deque<double> r;
    for (unsigned i = 1; i < 50; ++i) r.emplace_back(i);
//    r.insert(r.end(), {50, 100, 250, 500, 1000, 3600});
    return r;
}();
// constexpr std::array<unsigned, 1> C_quantizations = {10};
}

void ModelService::tune_features(
        datamodel::SVRParameters &params,
        const arma::mat &labels,
        const std::deque<bpt::ptime> &label_times,
        const std::deque<datamodel::DeconQueue_ptr> &features_aux,
        const bpt::time_duration &max_gap,
        const bpt::time_duration &aux_queue_res,
        const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    assert(labels.n_rows == label_times.size());
    const auto lag = params.get_lag_count();
    const auto adjacent_levels = params.get_adjacent_levels();
    const arma::vec mean_L = arma::vec(arma::mean(labels, 1));
    const unsigned coef_lag = datamodel::C_features_superset_coef * lag;
    const auto coef_lag_lag = coef_lag - lag;
    const auto levels = adjacent_levels.size();
    const auto levels_q = levels * features_aux.size();
    arma::vec best_score(levels, arma::fill::value(std::numeric_limits<double>::infinity()));
    datamodel::t_feature_mechanics fm{
            arma::Col<unsigned>(levels_q, arma::fill::none), arma::vec(levels_q * lag, arma::fill::none), arma::uvec(levels_q * coef_lag_lag, arma::fill::none)};

    std::deque<t_omp_lock> ins_l(levels_q);
    const auto offset_period = main_queue_resolution * PROPS.get_prediction_offset();
#ifdef EMO_DIFF
    const auto stripe_period = aux_queue_res * (coef_lag + 1);
#else
    const auto stripe_period = aux_queue_res * coef_lag;
#endif
#pragma omp parallel for num_threads(adj_threads(C_quantizations.size())) schedule(static, 1)
    for (const auto quantize: C_quantizations) {
#pragma omp parallel for num_threads(adj_threads(levels)) schedule(static, 1)
        for (size_t adj_ix = 0; adj_ix < levels; ++adj_ix) {
            const auto adjacent_level = adjacent_levels ^ adj_ix;
#pragma omp parallel for num_threads(adj_threads(features_aux.size())) schedule(static, 1)
            for (size_t qix = 0; qix < features_aux.size(); ++qix) {
#ifdef EMO_DIFF
                arma::mat features(label_times.size(), coef_lag + 1, arma::fill::none);
#else
                arma::mat features(label_times.size(), coef_lag, arma::fill::none);
#endif
                const auto &f = features_aux[qix];
                OMP_FOR(label_times.size())
                for (size_t rowix = 0; rowix < label_times.size(); ++rowix) {
                    const auto label_time = label_times[rowix];
                    const auto last_feature_time = label_time - offset_period;
                    const auto last_feature_iter = lower_bound_back(*f, f->end(), last_feature_time);
                    const auto p_last_row = *last_feature_iter;
                    const auto start_feature_time = last_feature_time - stripe_period * quantize;
                    const auto start_feature_iter = lower_bound_or_before_back(*f, last_feature_iter, start_feature_time);
                    arma::subview<double> level_row = features.row(rowix);
                    level_row.zeros();
                    generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_row);
#ifdef EMO_DIFF
                    level_row.cols(1, level_row.n_cols - 1) -= level_row.cols(0, level_row.n_cols - 2);
#endif
#ifndef NDEBUG
                    LOG4_TRACE("Added features with coef lag " << coef_lag << ", adjacent level " << adjacent_level << ", features row at "
                                                               << p_last_row->get_value_time() << ", " << common::present(level_row));
#endif
                }
                features.shed_col(0);
                const auto adj_ix_q = adj_ix + qix * levels;
                arma::vec scores(coef_lag, arma::fill::none), stretches(coef_lag, arma::fill::none);
                const unsigned num_gpu_chunks = _CEILDIV(features.n_elem * sizeof(double), common::gpu_handler_hid::get().get_max_gpu_data_chunk_size());
                const unsigned cols_gpu = _CEILDIV(coef_lag, num_gpu_chunks);
                OMP_FOR(num_gpu_chunks)
                for (size_t i = 0; i < coef_lag; i += cols_gpu) PROFILE_EXEC_TIME(align_features(
                        features.colptr(i), mean_L.mem, scores.memptr() + i, stretches.memptr() + i, mean_L.n_rows, std::min<unsigned>(coef_lag, i + cols_gpu) - i),
                                                                                  "Align features " << labels.n_rows << "x" << cols_gpu << ", quantize " << quantize);
                const arma::uvec trims = arma::uvec(arma::stable_sort_index(scores)).tail(coef_lag - lag);
                scores.shed_rows(trims);
                const double q_score = arma::accu(scores);
                ins_l[adj_ix_q].set();
                if (q_score < best_score[adj_ix_q]) {
                    LOG4_DEBUG("Best score " << q_score << ", quantize " << quantize << ", level " << adjacent_level << ", aux queue " << qix << ", lag " << lag <<
                                             ", coef lag " << coef_lag);
                    best_score[adj_ix_q] = q_score;
                    fm.quantization[adj_ix_q] = quantize;
                    stretches.shed_rows(trims);
                    fm.stretches.rows(adj_ix_q * lag, (adj_ix_q + 1) * lag - 1) = stretches;
                    fm.trims.rows(adj_ix_q * coef_lag_lag, (adj_ix_q + 1) * coef_lag_lag - 1) = (qix * levels + adj_ix) * coef_lag + trims;
                }
                ins_l[adj_ix_q].unset();
            }
        }
    }

    release_cont(ins_l);
    params.set_feature_mechanics(fm);
    LOG4_END();
}

void
ModelService::prepare_features(arma::mat &features, const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux,
                               const datamodel::SVRParameters &params, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res,
                               const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    const auto lag = params.get_lag_count();
    const auto coef_lag = datamodel::C_features_superset_coef * lag;
    const auto &adjacent_levels = params.get_adjacent_levels();
    const auto levels = adjacent_levels.size();
    const auto levels_lag = levels * lag;
    const auto feature_cols = levels_lag * features_aux.size();
    const auto offset_period = main_queue_resolution * PROPS.get_prediction_offset();
    const auto &quant = params.get_feature_mechanics().quantization;
#ifdef EMO_DIFF
    const auto stripe_period = aux_queue_res * (coef_lag + 1);
    if (features.n_rows != label_times.size() || features.n_cols != feature_cols) features.set_size(label_times.size(), feature_cols);
#else
    const auto stripe_period = aux_queue_res * lag;
    if (features.n_rows != label_times.size() || features.n_cols != feature_cols) features.zeros(label_times.size(), feature_cols);
#endif
#pragma omp parallel for num_threads(adj_threads(features_aux.size())) schedule(static, 1)
    for (size_t q = 0; q < features_aux.size(); ++q) {
        const auto &f = features_aux[q];
        const auto r_start = q * levels_lag;
#pragma omp parallel for num_threads(adj_threads(label_times.size())) schedule(static, 1)
        for (size_t rowix = 0; rowix < label_times.size(); ++rowix) {
            const auto label_time = label_times[rowix];
            const auto last_feature_time = label_time - offset_period;
            LOG4_TRACE("Adding features row to training matrix with value time " << label_time << ", last feature time " << last_feature_time);
#pragma omp unroll // parallel for num_threads(adj_threads(adjacent_levels.size())) schedule(static, 1)
            for (size_t adj_ix = 0; adj_ix < levels; ++adj_ix) {
                const auto adjacent_level = adjacent_levels ^ adj_ix;
                const auto start_feature_time = last_feature_time - stripe_period * quant[q * levels + adj_ix];
                const auto last_feature_iter = lower_bound_back(*f, last_feature_time);
                const auto start_feature_iter = lower_bound_or_before_back(*f, last_feature_iter, start_feature_time);
                const auto p_last_row = *last_feature_iter;
#ifdef EMO_DIFF
                arma::rowvec level_row(coef_lag + 1);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_row.row(0));
                level_row.tail(coef_lag) -= level_row.head(coef_lag);
                level_row.shed_col(0);
                level_row.shed_cols(params.get_feature_mechanics().trims);
                features.submat(rowix, r_start + adj_ix * lag, rowix, r_start + (adj_ix + 1) * lag - 1) = level_row; // common::slog_I(level_row);
#else
                arma::subview<double> level_row = features.submat(rowix, r_start + adj_ix * lag, rowix, r_start + (adj_ix + 1) * lag - 1);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_row);
#endif
                LOG4_TRACE(
                        "Added features with autoregressive lag " << lag << ", adjacent level " << adjacent_level << ", features row at " << p_last_row->get_value_time()
                                                        << ", " << common::present(level_row));
            }
        }
    }
    if (features.empty()) LOG4_WARN("No new data to prepare for training, features " << arma::size(features));
}


void
ModelService::train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model)
{
    const auto [p_features, p_labels, p_last_knowns, p_times] = get_training_data(dataset, ensemble, model);
    const auto last_value_time = p_times->back();
    if (model.get_last_modeled_value_time() >= last_value_time) {
        LOG4_ERROR("Data is older " << last_value_time << " than last modeled time " << model.get_last_modeled_value_time());
        return;
    }
    if (model.get_last_modeled_value_time() > bpt::min_date_time)
        train_online(model, *p_features, *p_labels, *p_last_knowns, last_value_time);
    else
        train_batch(model, p_features, p_labels, p_last_knowns, last_value_time);
    model.set_last_modeled_value_time(last_value_time);
    model.set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO("Finished training model " << model);
}


void
ModelService::train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::vec &last_knowns, const bpt::ptime &last_value_time)
{
    arma::mat residuals, learn_labels = labels;
    for (size_t g = 0; g < model.get_gradient_count(); ++g) {
        const bool is_gradient = g < model.get_gradient_count() - 1;
        const auto &m = model.get_gradient(g);
        if (is_gradient) residuals = learn_labels - m->predict(features);
#ifdef LAST_KNOWN_LABEL
            if (learn_labels.n_rows > 1)
                PROFILE_EXEC_TIME(m->learn(features.rows(0, features.n_rows - 2),
                                           learn_labels.rows(0, learn_labels.n_rows - 2),
                                           last_knowns.rows(0, learn_labels.n_rows - 2),
                                           new_last_modeled_value_time),
                              "Online SVM train gradient " << i);
            PROFILE_EXEC_TIME(m->learn(
                   features.row(features_data.n_rows - 1), learn_labels.row(learn_labels.n_row - 1),
                   last_knowns.row(learn_labels.n_rows - 1), new_last_modeled_value_time, true),
                              "Online SVM train last-known gradient " << i);
#else
        PROFILE_EXEC_TIME(m->learn(features, learn_labels, last_knowns, last_value_time), "Online SVM train gradient " << g);
#endif
        if (is_gradient) learn_labels = residuals;
    }
}


void
ModelService::train_batch(
        datamodel::Model &model,
        const mat_ptr &p_features,
        const mat_ptr &p_labels,
        const vec_ptr &p_last_knowns,
        const bpt::ptime &last_value_time)
{
    LOG4_BEGIN();

    datamodel::t_gradient_data gradient_data(p_features, p_labels, p_last_knowns);
    for (size_t gix = 0; gix < model.get_gradient_count(); ++gix) {
        const auto p_gradient = model.get_gradient(gix);
        if (!p_gradient) LOG4_THROW("SVR model for gradient " << gix << " not initialized " << model);
        p_gradient->batch_train(gradient_data.p_features, gradient_data.p_labels, gradient_data.p_last_knowns, last_value_time);
        if (model.get_gradient_count() > 1 && gix < model.get_gradient_count() - 1) {
            gradient_data = model.get_gradient(gix)->produce_residuals();
#pragma unroll
            for (auto &p: model.get_gradient(gix + 1)->get_param_set())
                p->set_svr_decremental_distance(gradient_data.p_features->n_rows - C_test_len);
        }
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble &ensemble, const size_t level, const std::deque<bpt::ptime> &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size());
    const auto p_aux_decon = ensemble.get_label_aux_decon();
    const auto lastknown_offset = PROPS.get_prediction_offset() * resolution;
#pragma omp parallel for num_threads(adj_threads(res.size())) schedule(static, 1)
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = (**lower_bound_back_before(*p_aux_decon, times[i] - lastknown_offset))[level];
    return res;
}


void
ModelService::predict(
        const datamodel::Ensemble &ensemble,
        datamodel::Model &model,
        const datamodel::t_level_predict_features &predict_features,
        const bpt::time_duration &resolution,
        tbb::mutex &insemx,
        data_row_container &out)
{
    arma::mat prediction(predict_features.p_features->n_rows, model.get_multiout());
    t_omp_lock predict_lock;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(model.get_gradient_count()))
    for (const auto &p_svr: model.get_gradients()) {
        const auto this_prediction = p_svr->predict(*predict_features.p_features, predict_features.times.back());
        predict_lock.set();
        prediction += this_prediction;
        predict_lock.unset();
    }
#ifdef EMO_DIFF
    const auto lk = get_last_knowns(ensemble, model.get_decon_level(), predict_features.times, resolution);
#pragma omp parallel for simd num_threads(adj_threads(prediction.n_cols))
    for (size_t i = 0; i < prediction.n_cols; ++i) prediction.col(i) = prediction.col(i) + lk; // common::sexp<double>(prediction.col(i)) + lk;
#endif
    prediction /= model.get_gradients().front()->get_dataset()->get_multistep();
    const tbb::mutex::scoped_lock l(insemx);
    datamodel::DataRow::insert_rows(out, prediction, predict_features.times, model.get_decon_level(), ensemble.get_level_ct(), true);
}


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

void
ModelService::check_feature_data(
        const datamodel::DataRow::container &data,
        const datamodel::DataRow::container::const_iterator &iter,
        const bpt::time_duration &max_gap,
        const bpt::ptime &feat_time)
{
    if (iter == data.end() || iter->get()->get_value_time() - feat_time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                    "Can't find data for prediction features. Needed value for " + bpt::to_simple_string(feat_time) +
                    ", nearest data available is " +
                    (iter == data.end() ? "not found" : "at " + bpt::to_simple_string((**iter).get_value_time())));
}


void
ModelService::init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble)
{
    if (!check(p_ensemble->get_models(), p_dataset->get_model_count()) && p_ensemble->get_id())
        p_ensemble->set_models(model_dao.get_all_ensemble_models(p_ensemble->get_id()), false);

    t_omp_lock init_models_l;
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_model_count() * p_dataset->get_multistep())) schedule(static, 1) collapse(2)
    for (size_t levix = 0; levix < p_dataset->get_transformation_levels(); levix += 2)
        for (size_t stepix = 0; stepix < p_dataset->get_multistep(); ++stepix)
            if (levix != p_dataset->get_half_levct()) {
                auto p_model = p_ensemble->get_model(levix, stepix);
                if (!p_model) {
                    p_model = ptr<datamodel::Model>(0, p_ensemble->get_id(), levix, stepix, PROPS.get_multiout(), p_dataset->get_gradient_count(),
                                                    p_dataset->get_max_chunk_size());
                    init_models_l.set();
                    p_ensemble->get_models().emplace_back(p_model);
                    init_models_l.unset();
                }
                configure(p_dataset, p_ensemble, p_model);
            }
}

} // business
} // svr
