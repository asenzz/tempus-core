#include "util/string_utils.hpp"
#include "cuqrsolve.hpp"
#include "calc_kernel_inversions.hpp"
#include "firefly.hpp"
#include "common/constants.hpp"
#include "model/SVRParameters.hpp"
#include "kernel_factory.hpp"
#include </opt/intel/oneapi/tbb/latest/include/tbb/parallel_reduce.h>
#include <tuple>
#include <string>
#include <mutex>
#include <deque>
#include <complex>
#include <cmath>
#include "SVRParametersService.hpp"
#include "common/gpu_handler.hpp"
#include "common/rtp_thread_pool.hpp"
#include "model/User.hpp"
#include "DAO/DatasetDAO.hpp"
#include "onlinesvr.hpp"
#include "common/defines.h"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/Logging.hpp"
#include "DQScalingFactorService.hpp"
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <map>
#include <iterator>
#include <limits>
#include <cstdlib>
#include <armadillo>
#include <boost/date_time/posix_time/ptime.hpp>
#include <iostream>
#include <oneapi/tbb/parallel_for.h>

#include "ModelService.hpp"
#include "EnsembleService.hpp"
#include "appcontext.hpp"
#include "DAO/ModelDAO.hpp"
#include "common/thread_pool.hpp"
#include "model/Model.hpp"
#include "util/TimeUtils.hpp"
#include "util/ValidationUtils.hpp"
#include "util/math_utils.hpp"

namespace svr {
namespace business {

ModelService::ModelService(svr::dao::ModelDAO &model_dao) : model_dao(model_dao)
{}


size_t svr::business::ModelService::to_level_ix(const size_t model_ix)
{
    return (model_ix >= 16 ? (model_ix + 1) : model_ix) * 2;
}

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
svr::business::ModelService::future_validate(
        const size_t start_ix,
        OnlineMIMOSVR &online_svr,
        const arma::mat &features,
        const arma::mat &labels,
        const arma::mat &last_knowns,
        const std::deque<bpt::ptime> &times,
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const std::string &column,
        const bool online_learn)
{
    LOG4_BEGIN();
    if (labels.n_rows <= start_ix) {
        LOG4_WARN("Calling future validate at the end of labels array. MAE is 1000");
        return {BAD_VALIDATION, BAD_VALIDATION, {}, {}, 0., {}};
    }

    const size_t ix_fini = std::min<size_t>(labels.n_rows - 1, labels.n_rows - 1);
    const size_t num_preds = 1 + ix_fini - start_ix;
    const auto params = online_svr.get_params();
    const auto level = params.get_decon_level();

    arma::mat svr_batch_predict;
    PROFILE_EXEC_TIME(svr_batch_predict = online_svr.predict(features.rows(start_ix, ix_fini), times[start_ix], false),
                      "Batch predict of " << num_preds << " rows, level " << level);
    if (svr_batch_predict.n_rows != num_preds) LOG4_ERROR("predicted_values.n_rows " << svr_batch_predict.n_rows << " != num_preds " << num_preds);

    arma::vec predicted(num_preds), predicted_online(num_preds), actual(num_preds), lastknown(num_preds);
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    size_t batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (size_t i_future = start_ix; i_future <= ix_fini; ++i_future) {
        const auto ix = i_future - start_ix;
        lastknown[ix] = DQScalingFactorService::unscale(arma::mean(last_knowns.row(i_future)), level, column, scaling_factors);
        predicted[ix] = DQScalingFactorService::unscale(arma::mean(svr_batch_predict.row(ix)), level, column, scaling_factors);
        actual[ix] = DQScalingFactorService::unscale(arma::mean(labels.row(i_future)), level, column, scaling_factors);
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

        std::stringstream row_report;
        row_report << "Position " << ix << ", level " << level << ", actual " << actual[ix] << ", batch predicted " << predicted[ix] << ", last known " << lastknown[ix]
                   << " batch MAE " << sum_absdiff_batch / double(ix) << ", MAE last-known " << sum_absdiff_lk / double(ix) << ", batch MAPE "
                   << 100. * sum_absdiff_batch / sum_abs_labels << ", MAPE last-known " << 100. * sum_absdiff_lk / sum_abs_labels << ", batch alpha "
                   << 100. * (sum_absdiff_lk / sum_absdiff_batch - 1.) << ", current batch alpha " << cur_alpha_pct_batch << " pct., batch correct predictions "
                   << 100. * double(batch_correct_predictions) / double(ix) << " pct, batch correct directions " << 100. * double(batch_correct_directions) / double(ix) << " pct.";

        if (online_learn) {
            PROFILE_EXEC_TIME(
                    predicted_online[ix] = DQScalingFactorService::unscale(arma::mean(arma::vectorise(
                            online_svr.predict(features.row(i_future), times[i_future], false))), level, column, scaling_factors),
                    "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols
                                      << " labels per row, level " << level << " at " << times[i_future]);
            PROFILE_EXEC_TIME(
                    online_svr.learn(features.row(i_future), labels.row(i_future), false, {}, times[i_future]),
                    "Online learn " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols
                                    << " labels per row, level " << level << " at " << times[i_future]);
#ifdef EMO_DIFF
            predicted_online[ix] += lastknown[ix];
#endif
            const double cur_absdiff_online = std::abs(predicted_online[ix] - actual[ix]);
            const double cur_alpha_pct_online = 100. * (cur_absdiff_lk / cur_absdiff_online - 1.);
            sum_absdiff_online += cur_absdiff_online;
            online_correct_predictions += cur_absdiff_online < cur_absdiff_lk;
            online_correct_directions += std::signbit(predicted_online[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);
            row_report << ", online predicted " << predicted_online[ix] << ", online MAE " << sum_absdiff_online / double(ix) << ", online MAPE " <<
                       100. * sum_absdiff_online / sum_abs_labels << ", online alpha " << 100. * (sum_absdiff_lk / sum_absdiff_online - 1.) << ", current online alpha "
                       << cur_alpha_pct_online << " pct., online correct predictions " << 100. * double(online_correct_predictions) / double(ix)
                       << " pct, online correct directions " << 100. * double(online_correct_directions) / double(ix) << " pct.";
        }
        LOG4_DEBUG(row_report.str());
    }
    LOG4_DEBUG(
            "Future predict from row " << start_ix << " until " << ix_fini << ", predictions " << num_preds << ", parameters " << params);
    const auto mape_lk = 100. * sum_absdiff_lk / sum_abs_labels;
    if (online_learn)
        return {sum_absdiff_online / double(num_preds), 100. * sum_absdiff_online / sum_abs_labels, predicted_online, actual, mape_lk, lastknown};
    else
        return {sum_absdiff_batch / double(num_preds), 100. * sum_absdiff_batch / sum_abs_labels, predicted, actual, mape_lk, lastknown};
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
    const auto default_num_chunks = DEFAULT_SVRPARAM_DECREMENT_DISTANCE / p_dataset->get_chunk_size();

    auto params = APP.svr_parameters_service.get_by_dataset_column_level(
            p_ensemble->get_dataset_id(), p_ensemble->get_decon_queue()->get_input_queue_column_name(), p_model->get_decon_level());

#pragma omp parallel for num_threads(adj_threads(p_model->get_gradient_count()))
    for (size_t g = 0; g < p_model->get_gradient_count(); ++g) {
        datamodel::t_param_set_ptr grad_params = SVRParametersService::slice(params, std::numeric_limits<size_t>::max(), g);
        if (grad_params->empty())
            for (size_t c = 0; c < default_num_chunks; ++c)
                grad_params->emplace(std::make_shared<datamodel::SVRParameters>(
                        0, p_dataset->get_id(),
                        p_ensemble->get_decon_queue()->get_input_queue_table_name(),
                        p_ensemble->get_decon_queue()->get_input_queue_column_name(),
                        p_model->get_decon_level(), c, g));
        p_model->get_gradient(g)->set_param_set(grad_params);
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

double
ModelService::get_quantized_feature(const size_t pos, const data_row_container::const_iterator &end_iter, const size_t level, const double quantization_mul,
                                    const size_t lag)
{
    auto row_iter = end_iter - (lag - pos) * quantization_mul;
    double result = 0;
    for (size_t sub_pos = 0; sub_pos < quantization_mul; ++sub_pos, ++row_iter)
        result += row_iter->get()->at(level);
    result /= std::floor(quantization_mul);
    /*
    if (!common::isnormalz(result))
        LOG4_WARN("Corrupt value " << result << " at " << row_iter->get()->get_value_time() << " pos " << pos << " level " <<
                    level << " lag count " << lag << " quantization mul " << quantization_mul);
    */
    return result;
}

bool
ModelService::prepare_features(
        const std::set<size_t> &adjacent_levels,
        const size_t lag,
        const data_row_container::const_iterator &end_iter,
        const bpt::time_duration &max_gap,
        const double main_to_aux_period_ratio,
        arma::rowvec &row)
{
    LOG4_BEGIN();
    const auto r_start = row.size();
    const auto p_last_row = std::prev(end_iter)->get();
    const auto levels = p_last_row->size();
    row.resize(r_start + adjacent_levels.size() * lag);
#pragma omp parallel for num_threads(adj_threads(adjacent_levels.size())) schedule(static, 1)
    for (size_t adj_ix = 0; adj_ix < adjacent_levels.size(); ++adj_ix) {
        const auto adjacent_level = adjacent_levels ^ adj_ix;
        const auto quantization_mul = svr::common::calc_quant_offset_mul(main_to_aux_period_ratio, adjacent_level, levels);
#ifdef EMO_DIFF
        arma::rowvec level_row(lag + 1);
        for (size_t i = 0; i < level_row.n_elem; ++i)
            level_row[i] = get_quantized_feature(i, end_iter, adjacent_level, quantization_mul, lag + 1);
        level_row.cols(1, level_row.n_cols - 1) -= level_row.cols(0, level_row.n_cols - 2);
        level_row.shed_col(0);
        // level_row[i] -= level_row.back(); // Emo's way
        // level_row.shed_col(level_row.n_elem - 1);
#else
        arma::rowvec level_row(lag);
        for (size_t i = 0; i < level_row.n_elem; ++i)
            level_row[i] = get_quantized_feature(i, end_iter, adjacent_level, quantization_mul, lag);
#endif
        row.cols(r_start + adj_ix * lag, r_start + (adj_ix + 1) * lag - 1) = level_row;
    }
    LOG4_TRACE("Lag " << lag << " adjacent_levels " << adjacent_levels.size() << " features row at " << p_last_row->get_value_time() << ", " << common::present(row));
    if (!common::sane(row)) {
        LOG4_ERROR("Row contains illegal values or empty " << arma::size(row) << ", " << row);
        return false;
    } else
        return true;
}


bool
ModelService::prepare_labels(
        const size_t level_ix,
        const data_row_container::const_iterator &label_aux_start_iter,
        const data_row_container::const_iterator &label_aux_end_iter,
        const boost::posix_time::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        arma::rowvec &labels_row,
        const bpt::time_duration &aux_resolution)
{
    const auto res = generate_labels(label_aux_start_iter, label_aux_end_iter, start_time, end_time, aux_resolution, level_ix, labels_row);
    return res && common::sane(labels_row);
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
bool
ModelService::get_training_data(
        arma::mat &all_features,
        arma::mat &all_labels,
        arma::mat &all_last_knowns,
        std::deque<bpt::ptime> &all_times,
        const datamodel::datarow_range &main_data,
        const datamodel::datarow_range &labels_aux,
        const std::deque<datamodel::datarow_range> &features_aux,
        const size_t lag,
        const std::set<size_t> &adjacent_levels,
        const bpt::time_duration &max_gap,
        const size_t level,
        const bpt::time_duration &aux_queue_res,
        const bpt::ptime &last_modeled_value_time,
        const bpt::time_duration &main_queue_resolution,
        const size_t multiout)
{
    LOG4_BEGIN();
    const auto main_to_aux_period_ratio = double(main_queue_resolution.ticks()) / double(aux_queue_res.ticks());
    const size_t req_rows = main_data.distance();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    LOG4_DEBUG("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.begin()->get()->get_value_time() <<
                                  " until " << main_data.rbegin()->get()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
    auto harvest_rows = [&](const datamodel::datarow_range &harvest_range) {
        tbb::concurrent_vector<arma::uword> shedded_rows;
        const size_t expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, multiout), features(expected_rows, 0), last_knowns(expected_rows, 1);
        std::deque<bpt::ptime> label_times;
        if (!level) label_times.resize(expected_rows);
//        LOG4_DEBUG("Processing range " << harvest_range.begin()->get()->get_value_time() << " to " << harvest_range.rbegin()->get()->get_value_time() << ", expected " << expected_rows << " rows.");

#pragma omp parallel for num_threads(adj_threads(expected_rows))
        for (size_t rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_iter = *(harvest_range.begin() + rowix);
            const bpt::ptime label_start_time = label_start_iter->get_value_time();
            if (label_start_time <= last_modeled_value_time) {
                LOG4_DEBUG("Skipping already modeled row with value time " << label_start_time);
                shedded_rows.emplace_back(rowix);
                continue;
            }
            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            arma::rowvec labels_row(multiout);
            const auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux.it(rowix * main_to_aux_period_ratio), label_start_time);
            if (label_aux_start_iter == labels_aux.contend()) {
                LOG4_ERROR("Can't find aux labels start " << label_start_time);
                shedded_rows.emplace_back(rowix);
                continue;
            } else if (label_aux_start_iter->get()->get_value_time() >= label_start_time + .5 * main_queue_resolution) {
                LOG4_ERROR("label aux start iter value time > label start time " << label_aux_start_iter->get()->get_value_time() << " > "
                                                                                 << label_start_time + .7 * main_queue_resolution);
                shedded_rows.emplace_back(rowix);
                continue;
            }
            const auto label_aux_end_iter = lower_bound(labels_aux.get_container(), label_aux_start_iter, label_start_time + main_queue_resolution);
            arma::rowvec features_row;
            bool feat_rc = true;
#pragma omp parallel for ordered schedule(static, 1) num_threads(adj_threads(features_aux.size()))
            for (const auto &f: features_aux) {
                const auto feature_end_iter = lower_bound_back(f.get_container(), f.end(), label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
#pragma omp ordered
                feat_rc &= prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row);
            }
            if (feat_rc &&
                prepare_labels(level, label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, labels_row, aux_queue_res)) {
                if (features_row.size() % lag > 0) LOG4_ERROR("Features row size " << features_row.size() << " not divisible by lag " << lag);
#pragma omp critical
                {
                    if (features.n_rows != expected_rows || features.n_cols != features_row.size()) features.set_size(expected_rows, features_row.size());
                    if (labels.n_rows != expected_rows || labels.n_cols != multiout) labels.set_size(expected_rows, multiout);
                    if (last_knowns.n_rows != expected_rows || last_knowns.n_cols != 1) last_knowns.set_size(expected_rows, 1);
                }

                features.row(rowix) = features_row;
                const auto p_anchor_row = *std::prev(
                        lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL));
#ifdef EMO_DIFF
                labels_row -= p_anchor_row->get_value(level);
#endif
                labels.row(rowix) = labels_row;
                last_knowns(rowix, 0) = p_anchor_row->get_value(level);
                if (!level) label_times[rowix] = label_start_time;
                if (ssize_t(rowix) >= harvest_range.distance() - 1)
                    LOG4_DEBUG(
                            "Last data row " << rowix << ", value time " << label_start_time << ", label aux start time "
                                             << label_aux_start_iter->get()->get_value_time() << ", last known time " << p_anchor_row->get_value_time() <<
                                             ", last last-known value " << last_knowns(rowix, 0) << ", label " << labels.row(rowix).back() << ", level " << level);
            } else {
                LOG4_WARN("For row at " << label_start_time << " can't assemble features " << arma::size(features_row) << " or labels " << arma::size(labels_row)
                                        << ", skipping.");
                shedded_rows.emplace_back(rowix);
            }
        }
        if (!shedded_rows.empty()) {
            const arma::uvec ashedded_rows = common::toarmacol(shedded_rows);
            shedded_rows.clear();
            LOG4_DEBUG("Shedding rows " << ashedded_rows);
            features.shed_rows(ashedded_rows);
            labels.shed_rows(ashedded_rows);
            last_knowns.shed_rows(ashedded_rows);
            if (!level) {
                std::deque<bpt::ptime> clean_label_times;
                for (arma::uword r = 0; r < label_times.size(); ++r) {
                    if (!arma::find(ashedded_rows == r).is_empty()) continue;
                    clean_label_times.emplace_back(label_times[r]);
                }
                label_times = clean_label_times;
            }
        }
        if (!labels.empty() && !features.empty() && !last_knowns.empty())
            LOG4_DEBUG("Returning labels " << common::present(labels) << ", features " << common::present(features) << ", last knowns " << common::present(last_knowns)
                                           << " for level " << level);
        return std::make_tuple(features, labels, last_knowns, label_times);
    };

    for (auto harvest_range = main_data;
         all_labels.n_rows < req_rows
         && harvest_range.begin()->get()->get_value_time() >= main_data.contbegin()->get()->get_value_time()
         && harvest_range.begin()->get()->get_value_time() > last_modeled_value_time
         && harvest_range.begin() != harvest_range.end();
         harvest_range.set_range(harvest_range.it(all_labels.n_rows - req_rows), harvest_range.begin())) {
        const auto [features, labels, last_knowns, label_times] = harvest_rows(harvest_range);
        if (all_features.n_cols != features.n_cols) all_features.set_size(all_features.n_rows, features.n_cols);
        if (all_labels.n_cols != labels.n_cols) all_labels.set_size(all_labels.n_rows, labels.n_cols);
        if (all_last_knowns.n_cols != 1) all_last_knowns.set_size(all_last_knowns.n_rows, 1);
        all_features = arma::join_cols(features, all_features);
        all_labels = arma::join_cols(labels, all_labels);
        all_last_knowns = arma::join_cols(last_knowns, all_last_knowns);
        if (!level) all_times.insert(all_times.begin(), label_times.begin(), label_times.end());
    }

#ifdef LAST_KNOWN_LABEL
    // Add last known value if preparing online train
    if (last_modeled_value_time > bpt::min_date_time) {
        auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux.it(main_data.distance() * .5), (main_data.begin() + main_data.distance() - 1)->get()->get_value_time() + main_queue_resolution + main_queue_resolution * (1. - OFFSET_PRED_MUL));
        if (label_aux_start_iter == labels_aux.contend()) --label_aux_start_iter;
        const bpt::ptime label_start_time = label_aux_start_iter->get()->get_value_time();
        arma::rowvec features_row;
        bool feat_rc = true;
        for (const auto &f: features_aux) {
            const auto feature_end_iter = lower_bound_back(f.get_container(), f.end(), label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
            feat_rc &= prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row);
        }

        if (feat_rc) {
            arma::rowvec labels_row(multiout);
            all_features = arma::join_cols(all_features, features_row);
            labels_row.fill(label_aux_start_iter->get()->get_value(level));
            const auto p_anchor_row = std::prev(lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL))->get();
#ifdef EMO_DIFF
            labels_row = labels_row - p_anchor_row->get_value(level);
#endif
            all_labels = arma::join_cols(all_labels, labels_row);
            all_last_knowns = arma::join_cols(all_last_knowns, arma::rowvec(p_anchor_row->get_value(level)));
            if (!level) all_times.emplace_back(label_start_time);
            LOG4_DEBUG("Temporary data last row, time " << label_start_time << " anchor time " << p_anchor_row->get_value_time());
        } else {
            LOG4_ERROR("Failed adding temporary row with time " << label_start_time << ", features size " << arma::size(features_row));
        }
    }
#endif
    if (all_labels.empty() or all_features.empty())
        LOG4_WARN("No new data to prepare for training, labels size " << arma::size(all_labels) << ", features size " << arma::size(all_features));
    else
        LOG4_DEBUG("Prepared level " << level << ", labels " << arma::size(all_labels) << ", features " << arma::size(all_features) << ", last knowns "
                                     << arma::size(all_last_knowns));

#if 0 // Save training data to file
    if (level == 0) {
        static size_t call_ct;
        all_labels.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/labels_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        all_features.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/features_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        all_last_knowns.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/last_knowns_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        ++call_ct;
    }
#endif

    return true;
}


void
ModelService::get_features_row(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::Ensemble_ptr &p_ensemble,
        const datamodel::Model_ptr &p_model,
        const bpt::ptime &pred_time,
        arma::rowvec &features_row)
{
    LOG4_BEGIN();

    const auto level = p_model->get_decon_level();
    const auto &aux_decons = p_ensemble->get_aux_decon_queues();
    const auto learning_levels = common::get_adjacent_indexes(level, p_model->get_params().get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
    const auto aux_resolution = p_dataset->get_aux_input_queues().empty() ? p_dataset->get_input_queue()->get_resolution() : p_dataset->get_aux_input_queue()->get_resolution();
    const auto lag = p_model->get_params().get_lag_count();
    const auto main_resolution = p_dataset->get_input_queue()->get_resolution();

    if (aux_decons.empty()) {
        LOG4_ERROR("Features queue is empty to predict " << pred_time);
        features_row.clear();
        return;
    }
    const double main_to_aux_period_ratio = double(main_resolution.ticks()) / double(aux_resolution.ticks());
    const bpt::ptime last_feat_expected_time = pred_time - main_resolution * OFFSET_PRED_MUL;
#pragma omp parallel for ordered schedule(static, 1) num_threads(adj_threads(aux_decons.size()))
    for (const auto &d: aux_decons) {
        const auto feature_aux_end_iter = lower_bound_back(d->get_data(), last_feat_expected_time);
        const size_t distance_from_start = std::distance(d->begin(), feature_aux_end_iter);
        if (distance_from_start < (1 + lag) * QUANTIZE_FIXED)
            LOG4_THROW("Not enough data to predict " << pred_time << ", found " << distance_from_start);

        if (std::prev(feature_aux_end_iter)->get()->get_value_time() != last_feat_expected_time - onesec)
            LOG4_THROW(
                    "Last feature time " << std::prev(feature_aux_end_iter)->get()->get_value_time() << " does not match expected " << last_feat_expected_time - onesec
                                         << " for " << pred_time << " of " << d->size() << " rows, starting " << d->front()->get_value_time() << ", ending "
                                         << d->back()->get_value_time() << ", p_predictions will be of lower quality, skipping!");
#pragma omp ordered
        if (!prepare_features(learning_levels, lag, feature_aux_end_iter, p_dataset->get_max_lookback_time_gap(), main_to_aux_period_ratio, features_row))
            LOG4_THROW("Failed preparing features for time " << pred_time);

        LOG4_DEBUG("Prepared prediction features for label at " << pred_time << " level " << level << " row size " << arma::size(features_row) <<
                                                                " features until " << std::prev(feature_aux_end_iter)->get()->get_value_time() << ", main to aux ratio "
                                                                << main_to_aux_period_ratio << ", lag " << lag);
    }

    if (features_row.size() % lag) LOG4_ERROR("Features row size dubious " << arma::size(features_row));

    LOG4_END();
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
ModelService::tune(
        datamodel::Model_ptr p_model,
        std::deque<t_gradient_tuned_parameters> &tune_parameters,
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels,
        const matrix_ptr &p_last_knowns)
{
#pragma omp parallel for num_threads(adj_threads(p_model->get_gradient_count()))
    for (size_t i = 0; i < p_model->get_gradient_count(); ++i) {
        const auto template_parameters = p_model->get_param_set(std::numeric_limits<size_t>::max(), i);
        if (!needs_tuning(template_parameters)) continue;
        OnlineMIMOSVR::tune(tune_parameters[i], *template_parameters, *p_features, *p_labels, *p_last_knowns, i);
    }
}

void
ModelService::train(
        datamodel::Model_ptr p_model,
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels,
        const bpt::ptime &new_last_modeled_value_time)
{
    if (p_model->get_params().get_skip()) {
        LOG4_DEBUG("Skipping training on model " << p_model->get_decon_level());
        p_model->set_last_modified(bpt::second_clock::local_time());
        return;
    }
    if (p_labels->empty() or p_features->empty() or p_labels->n_rows != p_features->n_rows) {
        LOG4_ERROR("Invalid learning data, labels matrix row count is " << arma::size(*p_labels) << " training features matrix row count is " << arma::size(*p_features));
        return;
    }

    if (p_model->get_last_modeled_value_time() > bpt::min_date_time)
        train_online(*p_features, *p_labels, p_model->get_gradients(), new_last_modeled_value_time);
    else
        train_batch(p_model->get_param_set_ptr(), p_model, p_features, p_labels);


    p_model->set_last_modeled_value_time(new_last_modeled_value_time);
    p_model->set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO("Finished training model " << *p_model);
}


void
ModelService::train_online(
        const arma::mat &features_data,
        const arma::mat &labels_data,
        std::deque<OnlineMIMOSVR_ptr> &svr_models,
        const bpt::ptime &new_last_modeled_value_time)
{
#ifdef LAST_KNOWN_LABEL
    for (size_t r = 0; r < labels_data.n_rows - 1; ++r) // TODO Implement online batch training of multiple rows
        PROFILE_EXEC_TIME(
                (void) p_svr_model->learn(features_data.row(r), labels_data.row(r), false, PROPS.get_dont_update_r_matrix()),
                "Online SVM train");
    p_svr_model->learn(features_data.row(features_data.n_rows - 1), labels_data.row(labels_data.n_rows - 1), true, PROPS.get_dont_update_r_matrix());
#else
    arma::mat residuals;
    for (size_t g = 0; g < svr_models.size(); ++g) {
        const auto &m = svr_models[g];
        arma::mat new_residuals;
        if (svr_models.size() > 1) new_residuals = labels_data - m->predict(features_data);
        if (g) PROFILE_EXEC_TIME(m->learn(features_data, residuals, false, {}, new_last_modeled_value_time), "Online SVM train gradient" << g)
        else PROFILE_EXEC_TIME(m->learn(features_data, labels_data, false, {}, new_last_modeled_value_time), "Online SVM train base gradient");
        if (svr_models.size() > 1) residuals = new_residuals;
    }
#endif
}


void
ModelService::train_batch(
        datamodel::t_param_set_ptr &p_param_set,
        datamodel::Model_ptr &p_model,
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels)
{
    LOG4_BEGIN();

    std::pair<matrix_ptr, matrix_ptr> gradient_data;
    for (size_t g = 0; g < p_model->get_gradient_count(); ++g) {
        auto gradient_params = SVRParametersService::slice(*p_param_set, std::numeric_limits<size_t>::max(), g);
        if (gradient_params->empty()) LOG4_THROW("Parameters for model " << *p_model << " not initialized.");
        bool save_parameters = false;
        for (auto &p: *gradient_params)
            save_parameters |= p->get_svr_kernel_param() == 0;
        const auto p_svr_model = g ? std::make_shared<OnlineMIMOSVR>(
                gradient_params, gradient_data.first, gradient_data.second, nullptr, p_model->get_multiout(), p_model->get_chunk_size()) :
                                 std::make_shared<OnlineMIMOSVR>(
                                         gradient_params, p_features, p_labels, nullptr, p_model->get_multiout(), p_model->get_chunk_size());
        p_model->set_gradient(g, p_svr_model);
        if (save_parameters)
            for (const auto &p: *gradient_params) {
                APP.svr_parameters_service.remove(p);
                APP.svr_parameters_service.save(p);
            }
        if (p_model->get_gradient_count() > 1) gradient_data = p_model->get_gradient(g)->produce_residuals();
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
    for (size_t i = 0; i < res.size(); ++i) {
        const auto t = times ^ i;
        const auto anchor_row = find_nearest_before(p_aux_decon->get_data(), t - lastknown_offset);
        res[i] = anchor_row->get()->get_value(level);
    }
    return res;
}

data_row_container
ModelService::predict(
        const datamodel::Ensemble_ptr &p_ensemble,
        const datamodel::Model_ptr &p_model,
        const datamodel::dq_scaling_factor_container_t &aux_dq_scaling_factors,
        const std::pair<std::set<bpt::ptime>, arma::mat> &predict_features,
        const bpt::time_duration &resolution)
{
    arma::mat prediction(predict_features.second.n_rows, p_model->get_multiout(), arma::fill::zeros);

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_model->get_gradients().size()))
    for (const auto &p_svr: p_model->get_gradients()) {
        const auto this_prediction = p_svr->predict(predict_features.second, *predict_features.first.rbegin(), true);
#pragma omp critical
        prediction += this_prediction;
    }

    prediction = arma::mean(
            DQScalingFactorService::unscale(
                    prediction, p_model->get_decon_level(), p_model->get_params().get_input_queue_column_name(), aux_dq_scaling_factors), 1);
#ifdef EMO_DIFF
    prediction += get_last_knowns(p_ensemble, p_model->get_decon_level(), predict_features.first, resolution);
#endif
    return datamodel::DataRow::insert_rows(prediction, predict_features.first);
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
                    (iter == data.end() ? "not found" : "at " + bpt::to_simple_string(iter->get()->get_value_time())));
}


void
ModelService::init_default_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble)
{
    auto &models = p_ensemble->get_models();
    const auto prev_size = models.size();
    models.resize(p_dataset->get_model_count());
#pragma omp parallel for num_threads(adj_threads(models.size()))
    for (size_t modix = prev_size; modix < p_dataset->get_model_count(); ++modix)
        models[modix] = std::make_shared<datamodel::Model>(
                0, p_ensemble->get_id(), to_level_ix(modix), p_dataset->get_multiout(), p_dataset->get_gradients(), p_dataset->get_chunk_size());
}

} // business
} // svr
