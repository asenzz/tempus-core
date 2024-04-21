#include "DeconQueueService.hpp"
#include <DAO/EnsembleDAO.hpp>
#include <atomic>
#include <execution>
#include <model/Ensemble.hpp>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"
#include "util/string_utils.hpp"
#include "cuqrsolve.hpp"
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


namespace svr {
namespace business {

ModelService::ModelService(svr::dao::ModelDAO &model_dao) : model_dao(model_dao)
{}


size_t ModelService::to_level_ix(const size_t model_ix, const size_t level_ct)
{
    if (model_ix >= MIN_LEVEL_COUNT)
        return ( (model_ix >= (level_ct / 4)) ? (model_ix + 1) : model_ix) * 2;
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
        return {C_bad_validation, C_bad_validation, {}, {}, 0., {}};
    }

    const size_t ix_fini = labels.n_rows - 1;
    const size_t num_preds = 1 + ix_fini - start_ix;
    const auto params = model.get_head_params();
    const auto level = params->get_decon_level();

    datamodel::t_level_predict_features predict_features{{times.cbegin() + start_ix, times.cend()}, ptr<arma::mat>(features.rows(start_ix, ix_fini))};
    LOG4_TRACE("Predicting features " << common::present<double>(*predict_features.p_features));
    data_row_container batch_predicted, cont_predicted_online;
    PROFILE_EXEC_TIME(ModelService::predict(
            ensemble, model, predict_features, dataset.get_input_queue()->get_resolution(), batch_predicted),
                      "Batch predict of " << num_preds << " rows, level " << level);
    if (batch_predicted.size() != num_preds)
        LOG4_THROW("Predicted size " << batch_predicted.size() << " not sane " << arma::size(*predict_features.p_features));

    LOG4_DEBUG("Batch predicted " << batch_predicted.size() << " values, parameters " << *params);

    arma::vec predicted(num_preds), predicted_online(num_preds),
        actual = arma::mean(labels.rows(start_ix, ix_fini), 1),
        lastknown = arma::mean(last_knowns.rows(start_ix, ix_fini), 1);
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    double batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (size_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted[ix] = batch_predicted[ix]->at(level);
#ifdef EMO_DIFF
        predicted[ix] += lastknown[ix];
        actual[ix] += lastknown[ix];
#endif
        const double cur_absdiff_lk = std::abs(lastknown[ix] - actual[ix]);
        const double cur_absdiff_batch = std::abs(predicted[ix] - actual[ix]);
        const double cur_alpha_pct_batch = 100. * (cur_absdiff_lk / cur_absdiff_batch - 1.);
        sum_abs_labels += std::abs(actual[ix]);
        sum_absdiff_batch += cur_absdiff_batch;
        sum_absdiff_lk += std::abs(actual[ix] - lastknown[ix]);
        batch_correct_predictions += cur_absdiff_batch < cur_absdiff_lk;
        batch_correct_directions += std::signbit(predicted[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);

        const auto ix_div = ix + 1.;
        const bool print_line = verbose || ix_future == ix_fini || ix % 115 == 0;
        std::stringstream row_report;
        if (print_line)
            row_report << "Position " << ix << ", level " << level << ", actual " << actual[ix] << ", batch predicted " << predicted[ix] << ", last known " <<
                       lastknown[ix] << " batch MAE " << sum_absdiff_batch / ix_div << ", MAE last-known " << sum_absdiff_lk / ix_div << ", batch MAPE "
                       << 100. * sum_absdiff_batch / sum_abs_labels << " pct., MAPE last-known " << 100. * sum_absdiff_lk / sum_abs_labels << " pct., batch alpha "
                       << 100. * (sum_absdiff_lk / sum_absdiff_batch - 1.) << " pct., current batch alpha " << cur_alpha_pct_batch << " pct., batch correct predictions "
                       << 100. * batch_correct_predictions / ix_div << " pct., batch correct directions " << 100. * batch_correct_directions / ix_div << " pct.";

        if (online) {
            PROFILE_EXEC_TIME(
                    ModelService::predict(
                            ensemble, model, datamodel::t_level_predict_features{{times[ix]}, ptr<arma::mat>(features.row(ix_future))},
                            dataset.get_input_queue()->get_resolution(), cont_predicted_online),
                    "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols  << " labels per row, level " << level <<
                    " at " << times[ix_future]);
            PROFILE_EXEC_TIME(
                    ModelService::train_online(model, features.row(ix_future), labels.row(ix_future), last_knowns.row(ix_future), times[ix_future]),
                    "Online learn " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                    " at " << times[ix_future]);
            predicted_online[ix] = cont_predicted_online.front()->at(level);
#ifdef EMO_DIFF
            predicted_online[ix] += lastknown[ix];
#endif
            const double cur_absdiff_online = std::abs(predicted_online[ix] - actual[ix]);
            const double cur_alpha_pct_online = 100. * (cur_absdiff_lk / cur_absdiff_online - 1.);
            sum_absdiff_online += cur_absdiff_online;
            online_correct_predictions += cur_absdiff_online < cur_absdiff_lk;
            online_correct_directions += std::signbit(predicted_online[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);
            if (print_line)
                row_report << ", online predicted " << predicted_online[ix] << ", online MAE " << sum_absdiff_online / ix_div << ", online MAPE " <<
                           100. * sum_absdiff_online / sum_abs_labels << " pct., online alpha " << 100. * (sum_absdiff_lk / sum_absdiff_online - 1.)
                           << " pct., current online alpha " << cur_alpha_pct_online << " pct., online correct predictions " << 100. * online_correct_predictions / ix_div
                           << " pct., online correct directions " << 100. * online_correct_directions / ix_div << " pct.";
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
ModelService::find(const std::deque<datamodel::Model_ptr> &models, const size_t levix)
{
    for (const auto &m: models)
        if (m->get_decon_level() == levix)
            return m;
    LOG4_WARN("Model for level " << levix << " not found among " << models.size() << " models.");
    return nullptr;
}

void ModelService::configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr &p_ensemble, datamodel::Model_ptr &p_model)
{
    if (!check(p_model->get_gradients(), p_model->get_gradient_count()) && p_model->get_id())
        p_model->set_gradients(model_dao.get_svr_by_model_id(p_model->get_id()), false);

    p_model->set_max_chunk_size(p_dataset->get_max_chunk_size());
    std::deque<datamodel::SVRParameters_ptr> model_params;
    if (p_dataset->get_id())
        model_params = APP.svr_parameters_service.get_by_dataset_column_level(p_dataset->get_id(), p_ensemble->get_column_name(), p_model->get_decon_level());

    const size_t default_model_num_chunks = datamodel::OnlineMIMOSVR::get_num_chunks(model_params.empty() ? DEFAULT_SVRPARAM_DECREMENT_DISTANCE :
                                                                                     (**model_params.cbegin()).get_svr_decremental_distance(), p_model->get_max_chunk_size());
    datamodel::dq_scaling_factor_container_t all_model_scaling_factors;
    if (p_model->get_id()) all_model_scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(p_model->get_id());

    OMP_LOCK(gradients_l)
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_gradient_count())) schedule(static, 1)
    for (size_t gix = 0; gix < p_dataset->get_gradient_count(); ++gix) {
        omp_set_lock(&gradients_l);
        auto p_svr_model = p_model->get_gradient(gix);
        omp_unset_lock(&gradients_l);

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

        OMP_LOCK(grad_params_l)
#pragma omp parallel for num_threads(adj_threads(grad_num_chunks)) schedule(static, 1)
        for (size_t chix = 0; chix < grad_num_chunks; ++chix)
            if (SVRParametersService::slice(grad_params, chix, gix).empty()) {
                datamodel::t_param_set level_grad_param_set;
                const auto p_params = (level_grad_param_set = SVRParametersService::slice(model_params, chix, gix)).size() ?
                                      *level_grad_param_set.cbegin() :
                                      ptr<datamodel::SVRParameters>(
                                              0, p_dataset->get_id(), p_dataset->get_input_queue()->get_table_name(), p_ensemble->get_column_name(),
                                              p_dataset->get_transformation_levels(), p_model->get_decon_level(), chix, gix);
                omp_set_lock(&grad_params_l);
                grad_params.emplace(p_params);
                set_params = true;
                omp_unset_lock(&grad_params_l);
            }

        if (!p_svr_model) {
            omp_set_lock(&gradients_l);
            p_svr_model = p_model->get_gradients().emplace_back(ptr<datamodel::OnlineMIMOSVR>(0, p_model->get_id(), grad_params, p_dataset));
            omp_unset_lock(&gradients_l);
        } else {
            if (set_params) p_svr_model->set_param_set(grad_params);
            if (!p_svr_model->get_dataset()) p_svr_model->set_dataset(p_dataset);
        }

        const auto adjacent_ct = (**grad_params.cbegin()).get_adjacent_levels().size();
        if (p_svr_model->get_scaling_factors().size() != p_svr_model->get_num_chunks()) {
#pragma omp parallel for num_threads(adj_threads(grad_num_chunks)) schedule(static, 1)
            for (size_t chix = 0; chix < grad_num_chunks; ++chix) {
                datamodel::DQScalingFactor_ptr p_sf;
                if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), p_model->get_id(), chix, p_svr_model->get_gradient_level(),
                                                  p_model->get_decon_level(), false, true)
                    && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, p_model->get_id(), chix, p_svr_model->get_gradient_level(),
                                                            p_model->get_decon_level(), false, true)))
                    p_svr_model->set_scaling_factor(p_sf);
                for (size_t levix = 0; levix < adjacent_ct; ++levix) {
                    if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), p_model->get_id(), chix, p_svr_model->get_gradient_level(),
                                                      levix, true, false)
                        && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, p_model->get_id(), chix, p_svr_model->get_gradient_level(),
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
std::tuple<mat_ptr, mat_ptr, vec_ptr, std::deque<bpt::ptime>>
ModelService::get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, size_t dataset_rows)
{
    LOG4_BEGIN();

    std::deque<bpt::ptime> label_times;
    auto p_labels = ptr<arma::mat>();
    auto p_last_knowns = ptr<arma::vec>();
    const auto level = model.get_decon_level();
    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    auto head_parameters = *model.get_head_params();
    if (!dataset_rows) dataset_rows = head_parameters.get_svr_decremental_distance() + C_emo_test_len;
    const datamodel::datarow_crange labels_range = {ModelService::get_start( // Main labels are used for timing
            label_decon,
            dataset_rows,
            model.get_last_modeled_value_time(),
            dataset.get_input_queue()->get_resolution()),
                                                    label_decon.get_data().cend(),
                                                    label_decon};
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? dataset.get_input_queue()->get_resolution() : dataset.get_aux_input_queue()->get_resolution();
    const auto adjacent_indexes = head_parameters.get_adjacent_levels();

    prepare_labels(
            *p_labels, *p_last_knowns, label_times, labels_range, labels_aux, dataset.get_max_lookback_time_gap(), level, aux_resolution,
            model.get_last_modeled_value_time(), dataset.get_input_queue()->get_resolution(), model.get_multiout());

    const auto p_features = dataset.get_calc_cache().get_cached_features(
            label_times, ensemble.get_aux_decon_queues(), head_parameters.get_lag_count(), adjacent_indexes, dataset.get_max_lookback_time_gap(), aux_resolution,
            dataset.get_input_queue()->get_resolution());

    if (p_labels->n_rows != p_features->n_rows)
        LOG4_THROW("Labels size " << arma::size(*p_labels) << ", features size " << arma::size(*p_features) << " do not match.");

    return {p_features, p_labels, p_last_knowns, label_times};
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
        const size_t multiout)
{
    LOG4_BEGIN();
    const auto main_to_aux_period_ratio = main_queue_resolution / aux_queue_res;
    const auto req_rows = main_data.distance();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    //LOG4_TRACE("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.begin()->get()->get_value_time() <<
    //                              " until " << main_data.rbegin()->get()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
    auto harvest_rows = [&](const datamodel::datarow_crange &harvest_range)
    {
        tbb::concurrent_unordered_set<arma::uword> shedded_rows;
        const auto expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, multiout);
        arma::vec last_knowns(expected_rows);
        tbb::concurrent_set<bpt::ptime> times;
        //LOG4_TRACE("Processing range " << (**harvest_range.begin()).get_value_time() << " to " << (**harvest_range.crbegin()).get_value_time() <<
        //                               ", expected " << expected_rows << " rows.");

#define SHED_ROW(msg) {                 \
        LOG4_DEBUG(msg);                \
        shedded_rows.emplace(rowix);    \
        continue;                       \
    }

#pragma omp parallel for num_threads(adj_threads(expected_rows)) schedule(static, 1 + expected_rows / adj_threads(expected_rows))
        for (std::decay_t<dtype(expected_rows)> rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_time = harvest_range[rowix]->get_value_time();
            if (label_start_time <= last_modeled_value_time)
                SHED_ROW("Skipping already modeled row with value time " << label_start_time);

            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            const auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux(rowix * main_to_aux_period_ratio), label_start_time);
            if (label_aux_start_iter == labels_aux.contend())
                SHED_ROW("Can't find aux labels start " << label_start_time)
            else if ((**label_aux_start_iter).get_value_time() >= label_start_time + .5 * main_queue_resolution)
                SHED_ROW("label aux start iter value time > label start time " << (**label_aux_start_iter).get_value_time() << " > " << label_start_time + .5 * main_queue_resolution);
            const auto label_aux_end_iter = lower_bound(labels_aux.get_container(), label_aux_start_iter, label_start_time + main_queue_resolution);
            const auto anchor_iter = lower_bound_back_before(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
            if (anchor_iter == labels_aux.contend())
                SHED_ROW("Can't find aux labels start " << label_start_time)
            const arma::subview<double> labels_row = labels.row(rowix);
            if (!generate_twap(label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, aux_queue_res, level, labels_row))
                SHED_ROW("Failed generating TWAP prices for " << label_start_time)

            if (labels_row.has_nonfinite())
                SHED_ROW("Sanity check of row at " << label_start_time << " failed, size " << arma::size(labels_row) << ", content " << labels_row)

            const auto p_anchor_row = *anchor_iter;
            last_knowns[rowix] = (*p_anchor_row)[level];
#ifdef EMO_DIFF
            labels_row -= last_knowns[rowix];
#endif
            times.emplace(label_start_time);
            if (ssize_t(rowix) >= harvest_range.distance() - 1)
                LOG4_DEBUG(
                        "Added last data row " << rowix << ", value time " << label_start_time << ", label aux start time " << (**label_aux_start_iter).get_value_time() <<
                        ", last known time " << p_anchor_row->get_value_time() << ", last last-known value " << last_knowns[rowix] << ", label " << labels_row << ", for level " << level);
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

    if (all_labels.n_cols != multiout) all_labels.zeros(all_labels.n_rows, multiout);

    for (auto harvest_range = main_data;
         dtype(req_rows)(all_labels.n_rows) < req_rows
         && (**harvest_range.begin()).get_value_time() >= (**main_data.contbegin()).get_value_time()
         && (**harvest_range.begin()).get_value_time() > last_modeled_value_time
         && harvest_range.begin() != harvest_range.end();
         harvest_range.set_range(harvest_range(all_labels.n_rows - req_rows), harvest_range.begin()))
    {
        const auto [labels, last_knowns, times] = harvest_rows(harvest_range);
        all_labels = arma::join_cols(labels, all_labels);
        all_last_knowns = arma::join_cols(last_knowns, all_last_knowns);
        all_times.insert(all_times.begin(), times.begin(), times.end());
        // std::copy(std::execution::par_unseq, times.begin(), times.end(), std::front_inserter(all_times));
    }

#ifdef LAST_KNOWN_LABEL
    // Add last known value if preparing online train
    if (last_modeled_value_time > bpt::min_date_time) {
        auto label_aux_start_iter = lower_bound(
                labels_aux.get_container(), labels_aux.it(main_data.distance() * .5),
                (**(main_data.begin() + main_data.distance() - 1)).get_value_time() + main_queue_resolution + main_queue_resolution * (1. - OFFSET_PRED_MUL));
        if (label_aux_start_iter == labels_aux.contend()) --label_aux_start_iter;
        const bpt::ptime label_start_time = (**label_aux_start_iter).get_value_time();
        arma::rowvec labels_row(multiout);
        labels_row.fill((**label_aux_start_iter)[level]);
        const auto &anchor_row = **std::prev(lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL));
#ifdef EMO_DIFF
        labels_row = labels_row - anchor_row[level];
#endif
        all_labels = arma::join_cols(all_labels, labels_row);
        all_last_knowns = arma::join_cols(all_last_knowns, arma::rowvec(anchor_row[level]));
        all_times.emplace_back(label_start_time);
        LOG4_DEBUG("Temporary data last row, time " << label_start_time << " anchor time " << anchor_row.get_value_time());
    }
#endif
    if (all_labels.empty() or all_last_knowns.empty())
        LOG4_WARN("No new data to prepare for training, labels " << arma::size(all_labels) << ", last-knowns " << arma::size(all_last_knowns));
    else
        LOG4_TRACE("Prepared level " << level << ", labels " << common::present(all_labels) << ", last-knowns " << common::present(all_last_knowns));
}


void
ModelService::prepare_features(
        arma::mat &features,
        const std::deque<bpt::ptime> &label_times,
        const std::deque<datamodel::DeconQueue_ptr> &features_aux,
        const size_t lag,
        const std::set<size_t> &adjacent_levels,
        const bpt::time_duration &max_gap,
        const bpt::time_duration &aux_queue_res,
        const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    const auto main_to_aux_period_ratio = main_queue_resolution / aux_queue_res;
    const auto feature_cols = adjacent_levels.size() * lag * features_aux.size();
    if (features.n_rows != label_times.size() || features.n_cols != feature_cols) features.zeros(label_times.size(), feature_cols);
#pragma omp parallel for num_threads(adj_threads(label_times.size())) schedule(static, 1)
    for (size_t rowix = 0; rowix < label_times.size(); ++rowix) {
        const auto label_time = label_times[rowix];
        const auto last_feature_time = label_time - main_queue_resolution * OFFSET_PRED_MUL;
        LOG4_TRACE("Adding features row to training matrix with value time " << label_time << ", last feature time " << last_feature_time);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(features_aux.size()))
        for (size_t q = 0; q < features_aux.size(); ++q) {
            const auto &f = features_aux[q];
            const auto r_start = q * adjacent_levels.size() * lag;
            const auto last_feature_iter = lower_bound_back(*f, f->end(), last_feature_time);
            const auto p_last_row = *last_feature_iter;
            const auto levels = p_last_row->size();
#pragma omp parallel for num_threads(adj_threads(adjacent_levels.size())) schedule(static, 1)
            for (size_t adj_ix = 0; adj_ix < adjacent_levels.size(); ++adj_ix) {
                const auto adjacent_level = adjacent_levels ^ adj_ix;
                const auto start_feature_time = last_feature_time - aux_queue_res * lag * common::calc_quant_offset_mul(main_to_aux_period_ratio, adjacent_level, levels);
                const auto start_feature_iter = lower_bound_or_before_back(*f, last_feature_iter, start_feature_time);
#ifdef EMO_DIFF
                arma::rowvec level_row(lag + 1);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_feats);
                level_row.cols(1, level_row.n_cols - 1) -= level_row.cols(0, level_row.n_cols - 2);
                level_row.shed_col(0);
                // level_row[i] -= level_row.back(); // Emo's way
                // level_row.shed_col(level_row.n_elem - 1);
#else
                arma::subview<double> level_row = features.submat(rowix, r_start + adj_ix * lag, rowix, r_start + (adj_ix + 1) * lag - 1);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_row);
#endif
                if (level_row.has_nonfinite()) {
                    LOG4_THROW("Row " << p_last_row->get_value_time() << ", lag " << lag << ", adjacent level " << adjacent_level << " contains illegal values or empty "
                                      << arma::size(level_row) << ", " << level_row);
                } else
                    LOG4_TRACE("Added features with lag " << lag << ", adjacent level " << adjacent_level << ", features row at " << p_last_row->get_value_time() << ", " << common::present(level_row));
            }
        }
    }
    if (features.empty())
        LOG4_WARN("No new data to prepare for training, features " << arma::size(features));
}


void
ModelService::train(
        datamodel::Model &model,
        const mat_ptr &p_features,
        const mat_ptr &p_labels,
        const vec_ptr &p_last_knowns,
        const bpt::ptime &new_last_modeled_value_time)
{
    if (model.get_last_modeled_value_time() > new_last_modeled_value_time) {
        LOG4_ERROR("Data is older " << new_last_modeled_value_time << " than last modeled time " << model.get_last_modeled_value_time());
        return;
    }

    if (model.get_last_modeled_value_time() > bpt::min_date_time)
        train_online(model, *p_features, *p_labels, *p_last_knowns, new_last_modeled_value_time);
    else
        train_batch(model, p_features, p_labels, p_last_knowns, new_last_modeled_value_time);

    model.set_last_modeled_value_time(new_last_modeled_value_time);
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
            for (auto &p: model.get_gradient(gix + 1)->get_param_set())
                p->set_svr_decremental_distance(gradient_data.p_features->n_rows - C_emo_test_len);
        }
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble &ensemble, const size_t level, const std::deque<bpt::ptime> &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size());
    const auto p_aux_decon = ensemble.get_label_aux_decon();
    const auto lastknown_offset = OFFSET_PRED_MUL * resolution;
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
        data_row_container &out)
{
    arma::mat prediction(predict_features.p_features->n_rows, model.get_multiout());
    OMP_LOCK(predict_lock);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(model.get_gradient_count()))
    for (const auto &p_svr: model.get_gradients()) {
        const auto this_prediction = p_svr->predict(*predict_features.p_features);
        omp_set_lock(&predict_lock);
        prediction += this_prediction;
        omp_unset_lock(&predict_lock);
    }

#ifdef EMO_DIFF
    prediction += get_last_knowns(p_ensemble, p_model.get_decon_level(), predict_features.row_times, resolution);
#endif

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

    OMP_LOCK(init_models_l)
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_model_count())) schedule(static, 1)
    for (size_t levix = 0; levix < p_dataset->get_transformation_levels(); levix += 2) {
        if (levix == p_dataset->get_half_levct()) continue;

        auto p_model = p_ensemble->get_model(levix);
        if (!p_model) {
            p_model = ptr<datamodel::Model>(0, p_ensemble->get_id(), levix, p_dataset->get_multiout(), p_dataset->get_gradient_count(), p_dataset->get_max_chunk_size());
            omp_set_lock(&init_models_l);
            p_ensemble->get_models().emplace_back(p_model);
            omp_unset_lock(&init_models_l);
        }
        configure(p_dataset, p_ensemble, p_model);
    }
}

} // business
} // svr
