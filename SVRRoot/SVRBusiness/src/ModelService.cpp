#include "DeconQueueService.hpp"
#include <DAO/EnsembleDAO.hpp>
#include <atomic>
#include <execution>
#include <model/Ensemble.hpp>
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
#include "util/ValidationUtils.hpp"
#include "util/math_utils.hpp"
#include "model/User.hpp"
#include "DAO/DatasetDAO.hpp"
#include "common/defines.h"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/Logging.hpp"
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
ModelService::future_validate(
        const size_t start_ix, datamodel::OnlineMIMOSVR &online_svr, const arma::mat &features, const arma::mat &labels, const arma::mat &last_knowns,
        const std::deque<bpt::ptime> &times, const datamodel::dq_scaling_factor_container_t &scaling_factors, const bool online_learn, const bool verbose)
{
    LOG4_BEGIN();
    if (labels.n_rows <= start_ix) {
        LOG4_WARN("Calling future validate at the end of labels array. MAE is 1000");
        return {BAD_VALIDATION, BAD_VALIDATION, {}, {}, 0., {}};
    }

    const size_t ix_fini = labels.n_rows - 1;
    const size_t num_preds = 1 + ix_fini - start_ix;
    const auto params = online_svr.get_params_ptr();
    const auto level = params->get_decon_level();

    LOG4_TRACE("Predicting features " << common::present<double>(features.rows(start_ix, ix_fini)));
    arma::mat svr_batch_predict;
    PROFILE_EXEC_TIME(svr_batch_predict = online_svr.predict(features.rows(start_ix, ix_fini)), "Batch predict of " << num_preds << " rows, level " << level);
    if (svr_batch_predict.n_rows != num_preds)
        LOG4_ERROR("Predicted size " << arma::size(svr_batch_predict) << " not sane " << arma::size(features.rows(start_ix, ix_fini)));

    LOG4_DEBUG("Batch predict result " << common::present(svr_batch_predict) << ", parameters " << *params);

    const auto &scaling_factor = **business::DQScalingFactorService::slice(scaling_factors, params->get_dataset_id(), params->get_input_queue_column_name(), {level}).cbegin();

    arma::vec predicted(num_preds), predicted_online(num_preds), actual(num_preds), lastknown(num_preds);
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    double batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (size_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted[ix] = DQScalingFactorService::unscale(arma::mean(svr_batch_predict.row(ix)), scaling_factor);
        lastknown[ix] = DQScalingFactorService::unscale(arma::mean(last_knowns.row(ix_future)), scaling_factor);
        actual[ix] = DQScalingFactorService::unscale(arma::mean(labels.row(ix_future)), scaling_factor);
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
        std::stringstream row_report;
        if (verbose || ix_future == ix_fini || ix == 115)
            row_report << "Position " << ix << ", level " << level << ", actual " << actual[ix] << ", batch predicted " << predicted[ix] << ", last known " <<
                       lastknown[ix] << " batch MAE " << sum_absdiff_batch / ix_div << ", MAE last-known " << sum_absdiff_lk / ix_div << ", batch MAPE "
                       << 100. * sum_absdiff_batch / sum_abs_labels << ", MAPE last-known " << 100. * sum_absdiff_lk / sum_abs_labels << ", batch alpha "
                       << 100. * (sum_absdiff_lk / sum_absdiff_batch - 1.) << ", current batch alpha " << cur_alpha_pct_batch << " pct., batch correct predictions "
                       << 100. * batch_correct_predictions / ix_div << " pct, batch correct directions " << 100. * batch_correct_directions / ix_div << " pct.";

        if (online_learn) {
            PROFILE_EXEC_TIME(
                    predicted_online[ix] = DQScalingFactorService::unscale(arma::mean(arma::vectorise(
                            online_svr.predict(features.row(ix_future)))), scaling_factor),
                    "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols
                                      << " labels per row, level " << level << " at " << times[ix_future]);
            PROFILE_EXEC_TIME(
                    online_svr.learn(features.row(ix_future), labels.row(ix_future), last_knowns.row(ix_future), times[ix_future], false),
                    "Online learn " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols
                                    << " labels per row, level " << level << " at " << times[ix_future]);
#ifdef EMO_DIFF
            predicted_online[ix] += lastknown[ix];
#endif
            const double cur_absdiff_online = std::abs(predicted_online[ix] - actual[ix]);
            const double cur_alpha_pct_online = 100. * (cur_absdiff_lk / cur_absdiff_online - 1.);
            sum_absdiff_online += cur_absdiff_online;
            online_correct_predictions += cur_absdiff_online < cur_absdiff_lk;
            online_correct_directions += std::signbit(predicted_online[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);
            if (verbose || ix_future == ix_fini || ix == 115)
                row_report << ", online predicted " << predicted_online[ix] << ", online MAE " << sum_absdiff_online / ix_div << ", online MAPE " <<
                           100. * sum_absdiff_online / sum_abs_labels << ", online alpha " << 100. * (sum_absdiff_lk / sum_absdiff_online - 1.)
                           << ", current online alpha " << cur_alpha_pct_online << " pct., online correct predictions " << 100. * online_correct_predictions / ix_div
                           << " pct, online correct directions " << 100. * online_correct_directions / ix_div << " pct.";
        }
        if (row_report.str().size()) LOG4_DEBUG(row_report.str());
    }
    const auto mape_lk = 100. * sum_absdiff_lk / sum_abs_labels;
    if (online_learn)
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
    std::deque<datamodel::SVRParameters_ptr> level_params;
    if (p_dataset->get_id())
        level_params = APP.svr_parameters_service.get_by_dataset_column_level(
                p_dataset->get_id(), p_ensemble->get_column_name(), p_model->get_decon_level());

    const size_t model_num_chunks = datamodel::OnlineMIMOSVR::get_num_chunks(level_params.empty() ? DEFAULT_SVRPARAM_DECREMENT_DISTANCE :
                                                                  (**level_params.cbegin()).get_svr_decremental_distance(), p_model->get_max_chunk_size());

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
        if (p_svr_model && p_svr_model->get_param_set().size()) {
            grad_params = p_svr_model->get_param_set();
            grad_num_chunks = p_svr_model->get_num_chunks();
        } else {
            set_params = true;
            grad_num_chunks = model_num_chunks;
        }

        OMP_LOCK(grad_params_l)
#pragma omp parallel for num_threads(adj_threads(grad_num_chunks)) schedule(static, 1)
        for (size_t chix = 0; chix < grad_num_chunks; ++chix)
            if (SVRParametersService::slice(grad_params, chix, gix).empty()) {
                datamodel::t_param_set level_grad_param_set;
                const auto p_params = (level_grad_param_set = SVRParametersService::slice(level_params, chix, gix)).size() ?
                                      *level_grad_param_set.cbegin() :
                                      ptr<datamodel::SVRParameters>(
                                              0, p_dataset->get_id(), p_dataset->get_input_queue()->get_table_name(), p_ensemble->get_column_name(),
                                              p_model->get_decon_level(), chix, gix);
                omp_set_lock(&grad_params_l);
                grad_params.emplace(p_params);
                set_params = true;
                omp_unset_lock(&grad_params_l);
            }

        if (!p_svr_model) {
            omp_set_lock(&gradients_l);
            p_model->get_gradients().emplace_back(ptr<datamodel::OnlineMIMOSVR>(0, p_model->get_id(), grad_params, p_dataset));
            omp_unset_lock(&gradients_l);
        } else {
            if (set_params) p_svr_model->set_param_set(grad_params);
            if (!p_svr_model->get_dataset()) p_svr_model->set_dataset(p_dataset);
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
    std::deque<bool> present(model_ct, false);
    for (const auto &p: models) present[to_model_ix(p->get_decon_level(), to_level_ct(model_ct))] = true;
    return std::all_of(std::execution::par_unseq, present.begin(), present.end(), [](const auto p) { return p; });
}

bool ModelService::check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const size_t grad_ct)
{
    std::deque<bool> present(grad_ct, false);
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
std::tuple<matrix_ptr, matrix_ptr, vec_ptr>
ModelService::get_training_data(
        std::deque<bpt::ptime> &times,
        datamodel::Dataset &dataset,
        const datamodel::Ensemble &ensemble,
        const datamodel::Model &model,
        size_t dataset_rows)
{
    LOG4_BEGIN();

    std::deque<bpt::ptime> label_times;
    auto p_labels = ptr<arma::mat>();
    auto p_last_knowns = ptr<arma::vec>();
    const auto level = model.get_decon_level();
    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    const auto head_parameters = *model.get_head_params();
    if (!dataset_rows) dataset_rows = head_parameters.get_svr_decremental_distance() + EMO_TEST_LEN;
    const datamodel::datarow_crange labels_range = {ModelService::get_start( // Main labels are used for timing
                label_decon,
                dataset_rows,
                model.get_last_modeled_value_time(),
                dataset.get_input_queue()->get_resolution()),
                                                    label_decon.get_data().cend(),
                                                    label_decon};
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? dataset.get_input_queue()->get_resolution() : dataset.get_aux_input_queue()->get_resolution();
    const auto adjacent_indexes = common::get_adjacent_indexes(level, head_parameters.get_svr_adjacent_levels_ratio(), dataset.get_transformation_levels());

    get_label_data(
            *p_labels, *p_last_knowns, label_times, labels_range, labels_aux, dataset.get_max_lookback_time_gap(), level, aux_resolution,
            model.get_last_modeled_value_time(), dataset.get_input_queue()->get_resolution(), model.get_multiout());

    const auto p_features = dataset.get_calc_cache().get_cached_features(
            label_times, ensemble.get_aux_decon_queues(), head_parameters.get_lag_count(), adjacent_indexes, dataset.get_max_lookback_time_gap(), aux_resolution,
            dataset.get_input_queue()->get_resolution());

    if (p_labels->n_rows != p_features->n_rows)
        LOG4_THROW("Labels size " << arma::size(*p_labels) << ", features size " << arma::size(*p_features) << " do not match.");

    if (!level) times = label_times;
    return {p_features, p_labels, p_last_knowns};
}


void
ModelService::get_label_data(
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
        LOG4_TRACE("Processing range " << (**harvest_range.begin()).get_value_time() << " to " << (**harvest_range.crbegin()).get_value_time() <<
                                       ", expected " << expected_rows << " rows.");
#define SHED_ROW(msg) { \
        LOG4_DEBUG(msg); \
        shedded_rows.emplace(rowix); \
        continue; \
    }

#pragma omp parallel for num_threads(adj_threads(expected_rows)) schedule(static, 1)
        for (std::remove_const_t<decltype(expected_rows)> rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_time = harvest_range[rowix]->get_value_time();
            if (label_start_time <= last_modeled_value_time)
                SHED_ROW("Skipping already modeled row with value time " << label_start_time);

            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            arma::rowvec labels_row(multiout);
            const auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux(rowix * main_to_aux_period_ratio), label_start_time);
            if (label_aux_start_iter == labels_aux.contend())
                SHED_ROW("Can't find aux labels start " << label_start_time)
            else if ((**label_aux_start_iter).get_value_time() >= label_start_time + .5 * main_queue_resolution)
                SHED_ROW("label aux start iter value time > label start time " << (**label_aux_start_iter).get_value_time() << " > " << label_start_time + .5 * main_queue_resolution);
            const auto label_aux_end_iter = lower_bound(labels_aux.get_container(), label_aux_start_iter, label_start_time + main_queue_resolution);
            const auto anchor_iter = lower_bound_back_before(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
            if (anchor_iter == labels_aux.contend())
                SHED_ROW("Can't find aux labels start " << label_start_time)

            if (!generate_twap(label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, aux_queue_res, level, labels_row))
                SHED_ROW("Failed generating TWAP prices for " << label_start_time)

            if (!common::sane(labels_row)) SHED_ROW("Sanity check of row at " << label_start_time << " failed, size " << arma::size(labels_row) << ", content " << labels_row)

            const auto p_anchor_row = *anchor_iter;
#ifdef EMO_DIFF
            labels_row -= p_anchor_row->get_value(level);
#endif
            labels.row(rowix) = labels_row;
            last_knowns[rowix] = (*p_anchor_row)[level];
            times.emplace(label_start_time);
            if (ssize_t(rowix) >= harvest_range.distance() - 1)
                LOG4_DEBUG(
                        "Added last data row " << rowix << ", value time " << label_start_time << ", label aux start time " << (**label_aux_start_iter).get_value_time() <<
                        ", last known time " << p_anchor_row->get_value_time() << ", last last-known value " << last_knowns[rowix] << ", label " <<
                        labels.row(rowix).back() << ", for level " << level);

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
            LOG4_TRACE("Returning labels " << common::present(labels) << ", last knowns " << common::present(last_knowns) << " for level " << level);
        return std::tuple{labels, last_knowns, times};
    };

    if (all_labels.n_cols != multiout) all_labels.set_size(all_labels.n_rows, multiout);

    for (auto harvest_range = main_data;
         decltype(req_rows)(all_labels.n_rows) < req_rows
         && (**harvest_range.begin()).get_value_time() >= (**main_data.contbegin()).get_value_time()
         && (**harvest_range.begin()).get_value_time() > last_modeled_value_time
         && harvest_range.begin() != harvest_range.end();
         harvest_range.set_range(harvest_range(all_labels.n_rows - req_rows), harvest_range.begin()))
    {
        const auto [labels, last_knowns, times] = harvest_rows(harvest_range);
        all_labels = arma::join_cols(labels, all_labels);
        all_last_knowns = arma::join_cols(last_knowns, all_last_knowns);
        all_times.insert(all_times.begin(), times.begin(), times.end());
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
ModelService::get_features_data(
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
    if (features.n_rows != label_times.size() || features.n_cols != feature_cols) features.set_size(label_times.size(), feature_cols);
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
                const auto start_feature_time = last_feature_time - aux_queue_res * lag * svr::common::calc_quant_offset_mul(main_to_aux_period_ratio, adjacent_level, levels);
                const auto start_feature_iter = lower_bound_or_before_back(*f, last_feature_iter, start_feature_time);
#ifdef EMO_DIFF
                arma::rowvec level_row(lag + 1);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_feats);
                level_row.cols(1, level_row.n_cols - 1) -= level_row.cols(0, level_row.n_cols - 2);
                level_row.shed_col(0);
                // level_row[i] -= level_row.back(); // Emo's way
                // level_row.shed_col(level_row.n_elem - 1);
#else
                arma::rowvec level_row(lag);
                generate_twap(start_feature_iter, last_feature_iter, start_feature_time, last_feature_time, aux_queue_res, adjacent_level, level_row);
#endif
                if (!common::sane(level_row)) {
                    LOG4_THROW("Row " << p_last_row->get_value_time() << ", lag " << lag << ", adjacent level " << adjacent_level << " contains illegal values or empty "
                                      << arma::size(level_row) << ", " << level_row);
                } else
                    LOG4_TRACE("Added features with lag " << lag << ", adjacent level " << adjacent_level << ", features row at " << p_last_row->get_value_time() << ", " << common::present(level_row));
                features.submat(rowix, r_start + adj_ix * lag, rowix, r_start + (adj_ix + 1) * lag - 1) = level_row;
            }
        }
    }
    if (features.empty())
        LOG4_WARN("No new data to prepare for training, features " << arma::size(features));
}


void
ModelService::get_features_row(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::Ensemble_ptr &p_ensemble,
        const datamodel::Model_ptr &p_model,
        const bpt::ptime &pred_time,
        const std::set<size_t> &adjacent_levels,
        arma::rowvec &features_row)
{
    LOG4_BEGIN();

    get_features_data(
            features_row, {pred_time}, p_ensemble->get_aux_decon_queues(), p_model->get_head_params()->get_lag_count(), adjacent_levels, p_dataset->get_max_lookback_time_gap(),
            p_dataset->get_aux_input_queues().empty() ? p_dataset->get_input_queue()->get_resolution() : p_dataset->get_aux_input_queue()->get_resolution(),
            p_dataset->get_input_queue()->get_resolution());

    LOG4_DEBUG("Prepared prediction features for " << pred_time << ", level " << p_model->get_decon_level() << ", row " << arma::size(features_row));
}

bool
ModelService::needs_tuning(const datamodel::t_param_set_ptr &p_param_set)
{
    for (const auto &p: *p_param_set)
        if (p->get_svr_kernel_param() == 0)
            return true;
    return false;
}


void
ModelService::train(
        datamodel::Model &model,
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels,
        const vec_ptr &p_last_knowns,
        const bpt::ptime &new_last_modeled_value_time)
{
    if (!p_labels || p_labels->empty()
        || !p_features || p_features->empty()
        || p_labels->n_rows != p_features->n_rows
        || model.get_last_modeled_value_time() > new_last_modeled_value_time) {
        LOG4_ERROR("Invalid learning data, or data is older " << new_last_modeled_value_time << " than last modeled time " << new_last_modeled_value_time);
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
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels,
        const vec_ptr &p_last_knowns,
        const bpt::ptime &last_value_time)
{
    LOG4_BEGIN();

    datamodel::t_gradient_data gradient_data{p_features, p_labels, p_last_knowns};
    for (size_t g = 0; g < model.get_gradient_count(); ++g) {
        const auto p_gradient = model.get_gradient(g);
        if (!p_gradient) LOG4_THROW("SVR model for gradient " << g << " not initialized " << model);
        p_gradient->batch_train(gradient_data.p_features, gradient_data.p_labels, gradient_data.p_last_knowns, last_value_time);
        if (model.get_gradient_count() > 1) gradient_data = model.get_gradient(g)->produce_residuals();
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble_ptr &p_ensemble, const size_t level, const std::set<bpt::ptime> &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size());
    const auto p_aux_decon = p_ensemble->get_aux_decon_queue(p_ensemble->get_column_name());
    const auto lastknown_offset = OFFSET_PRED_MUL * resolution;
#pragma omp parallel for num_threads(adj_threads(res.size())) schedule(static, 1)
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = (**find_nearest_before(p_aux_decon->get_data(), (times ^ i) - lastknown_offset)).at(level);
    return res;
}


void
ModelService::predict(
        const datamodel::Ensemble_ptr &p_ensemble,
        datamodel::Model_ptr &p_model,
        const datamodel::dq_scaling_factor_container_t &aux_dq_scaling_factors,
        const std::pair<std::set<bpt::ptime>, arma::mat> &predict_features,
        const bpt::time_duration &resolution,
        data_row_container &out)
{
    arma::mat prediction(predict_features.second.n_rows, p_model->get_multiout());
    OMP_LOCK(predict_lock);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_model->get_gradient_count()))
    for (const auto &p_svr: p_model->get_gradients()) {
        const auto this_prediction = p_svr->predict(predict_features.second);
        omp_set_lock(&predict_lock);
        prediction += this_prediction;
        omp_unset_lock(&predict_lock);
    }

    prediction = arma::mean(DQScalingFactorService::unscale(
            prediction, p_model->get_decon_level(), p_model->get_head_params()->get_input_queue_column_name(), aux_dq_scaling_factors), 1);
#ifdef EMO_DIFF
    prediction += get_last_knowns(p_ensemble, p_model->get_decon_level(), predict_features.first, resolution);
#endif

    datamodel::DataRow::insert_rows(out, prediction, predict_features.first, p_model->get_decon_level(), p_ensemble->get_level_ct(), true);
}


const datamodel::datarow_range
ModelService::prepare_feat_range(
        const datamodel::DataRow::container &data,
        const boost::posix_time::time_duration &max_gap,
        const boost::posix_time::ptime &predict_time,
        const ssize_t lag_count)
{
    LOG4_BEGIN();
    if (data.empty()) LOG4_THROW("Data is empty!");
    const auto it_last_feature = find_nearest_after(data, predict_time, max_gap, lag_count);
    auto it_lookback = it_last_feature;
    if (std::distance(data.begin(), it_lookback) <= lag_count)
        LOG4_THROW("Distance between begin " << data.front()->get_value_time() << " and lookback " << it_lookback->get()->get_value_time() <<
                                             " too small " << std::distance(data.begin(), it_lookback));
    it_lookback -= lag_count;
    LOG4_TRACE("Prepared feature range for predict time " << predict_time << " feature start time " <<
                                                          it_lookback->get()->get_value_time() << " features end time - 1 "
                                                          << std::prev(it_last_feature)->get()->get_value_time());
    return datamodel::datarow_range(
            common::remove_constness(const_cast<datamodel::DataRow::container &>(data), it_lookback),
            common::remove_constness(const_cast<datamodel::DataRow::container &>(data), it_last_feature),
            const_cast<datamodel::DataRow::container &>(data));
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
        THROW_EX_FS(svr::common::insufficient_data,
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
        THROW_EX_FS(svr::common::insufficient_data,
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
