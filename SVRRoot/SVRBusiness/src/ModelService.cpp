#include <iostream>
#include <fstream>
#include <tbb/parallel_for.h>

#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DAO/ModelDAO.hpp"
#include "common/thread_pool.hpp"
#include "util/TimeUtils.hpp"
#include "util/math_utils.hpp"


using namespace svr::common;
using namespace svr::datamodel;


namespace svr {
namespace business {


ModelService::ModelService(
        svr::dao::ModelDAO &model_dao,
        const bool update_r_matrix,
        const size_t max_smo_iterations,
        const double smo_epsilon_divisor,
        const size_t max_segment_length,
        const size_t multistep_len)
        : model_dao(model_dao)
          //svm_batch(update_r_matrix, max_smo_iterations, smo_epsilon_divisor, multistep_len)
{}


Model_ptr ModelService::get_model_by_id(const bigint model_id)
{
    return model_id ? model_dao.get_by_id(model_id) : nullptr;
}

int ModelService::save(const Model_ptr &model)
{
    reject_nullptr(model);
    if (!model->get_id()) model->set_id(model_dao.get_next_id());
    return model_dao.save(model);
}

bool ModelService::exists(const Model_ptr &model)
{
    return model_dao.exists(model->get_id());
}

int ModelService::remove(const Model_ptr &model)
{
    reject_nullptr(model);
    return model_dao.remove(model);
}

int ModelService::remove_by_ensemble_id(const bigint ensemble_id)
{
    return model_dao.remove_by_ensemble_id(ensemble_id);
}

std::vector<Model_ptr> ModelService::get_all_models_by_ensemble_id(const bigint ensemble_id)
{
    try {
        return model_dao.get_all_ensemble_models(ensemble_id);
    } catch (...) {
        LOG4_ERROR("Cannot read models from the DB.");
        return {};
    }
}

Model_ptr ModelService::get_model(const bigint ensemble_id, const size_t decon_level)
{
    return model_dao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

double
ModelService::get_quantized_feature(const size_t pos, const data_row_container::const_iterator &end_iter, const size_t level, const double quantization_mul, const size_t lag)
{
    double result = 0;
#ifdef EMO_DIFF_FEATS
    auto row_iter = end_iter - (lag + 1 - pos) * quantization_mul;
#else
    auto row_iter = end_iter - (lag - pos) * quantization_mul;
#endif
#ifndef LAST_QUANT_FEAT
    for (size_t sub_pos = 0; sub_pos < quantization_mul; ++sub_pos, ++row_iter)
        result += row_iter->get()->get_value(level);
    result /= std::floor(quantization_mul);
#else
    result = (row_iter + quantization_mul - 1)->get()->get_value(level);
#endif
    if (std::isnan(result) or std::isinf(result))
        LOG4_WARN("Corrupt value " << result << " at " << row_iter->get()->get_value_time() << " pos " << pos << " level " << level << " lag count " << lag << " quantization mul " << quantization_mul);
    return result;
}

bool
ModelService::prepare_features(
        const std::set<size_t> &adjacent_levels,
        const size_t lag,
        const data_row_container::const_iterator &end_iter,
        const bpt::time_duration &max_gap,
        const double main_to_aux_period_ratio,
#ifndef NEW_SCALING
        const svr::datamodel::dq_scaling_factor_container_t &scaling_factors,
#endif
        arma::rowvec &row)
{
    LOG4_TRACE("Processing row with lag " << lag << " until " << std::prev(end_iter)->get()->get_value_time());

    row.resize(row.size() + adjacent_levels.size() * lag);
    size_t adj_ix = 0;
    for (const auto adjacent_level: adjacent_levels) {
        const auto quantization_mul = svr::common::calc_quant_offset_mul(main_to_aux_period_ratio, adjacent_level, std::prev(end_iter)->get()->get_values().size());
#ifdef EMO_DIFF_FEATS
        arma::rowvec level_row(lag + 1);
#else
        arma::rowvec level_row(lag);
#endif
        for (size_t i = 0; i < level_row.n_elem; ++i)
            level_row[i] = get_quantized_feature(i, end_iter, adjacent_level, quantization_mul, lag);
#ifdef EMO_DIFF_FEATS
        for (size_t i = 0; i < level_row.n_elem - 1; ++i)
            level_row[i] = level_row[i] - level_row[level_row.n_elem - 1];
        level_row.shed_col(level_row.n_elem - 1);
#endif

        LOG4_TRACE("Unscaled features " << adjacent_level << " vector for time " << end_iter->get()->get_value_time() << ", first " << level_row.front() << ", last " << level_row.back() << ", max " << level_row.max() << ", min " << level_row.min());
#ifndef NEW_SCALING
        level_row = scale(level_row, adjacent_level, scaling_factors, false);
#endif
        row.cols(adj_ix * lag, (adj_ix + 1) * lag - 1) = level_row;
        ++adj_ix;
    }
    LOG4_TRACE("Scaled features row for time " << end_iter->get()->get_value_time() << ", first " << row.front() << ", last " << row.back() << ", max " << row.max() << ", min " << row.min());

    LOG4_TRACE("Lag " << lag << " end feature time " << end_iter->get()->get_value_time() << " row size is " << row.size() << " adjacent_levels " << adjacent_levels.size() << " lag " << lag);
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
        const std::vector<double>::const_iterator it_tick_volume_begin,
        const size_t lag,
        arma::rowvec &row)
{
    LOG4_THROW("Not implemented!");
#if 0
    std::vector<double>::const_iterator it{it_tick_volume_begin};
    for (size_t i = 0; i < lag; ++i, it += QUANTIZE_FIXED) row[i] = *it;
#endif
}

// Takes a decon queue and prepares feature vectors, using lag_count number of autoregressive features (and other misc features).
bool
ModelService::get_training_data(
        arma::mat &all_features,
        arma::mat &all_labels,
        arma::mat &all_last_knowns,
        std::vector<bpt::ptime> &all_times,
#ifndef NEW_SCALING
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const datamodel::dq_scaling_factor_container_t &aux_scaling_factors,
#endif
        const datamodel::datarow_range &main_data_range,
        const datamodel::datarow_range &aux_data_range,
        const size_t lag,
        const std::set<size_t> &adjacent_levels,
        const bpt::time_duration &max_gap,
        const size_t level,
        const double main_to_aux_period_ratio,
        const bpt::ptime &last_modeled_value_time,
        const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    const auto aux_queue_res = main_queue_resolution / main_to_aux_period_ratio;
    const size_t req_rows = main_data_range.distance();
    if (req_rows < 1 or main_data_range.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    LOG4_DEBUG("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data_range.begin()->get()->get_value_time() <<
                                  " until " << main_data_range.rbegin()->get()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
#ifdef COMPRESS_LABEL_TIME
    const auto last_label_time = std::next(main_data_range.begin(), main_data_range.distance() - 1)->get()->get_value_time();
#endif
    auto harvest_rows = [&](const datamodel::datarow_range &harvest_range) {
        std::mutex mx_resize, mx_shedded_rows;
        arma::uvec shedded_rows;
        const size_t expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, PROPS.get_multistep_len()), features(expected_rows, 0), last_knowns(expected_rows, 1);
        std::vector<bpt::ptime> label_times(expected_rows);
        LOG4_DEBUG("Processing range " << harvest_range.begin()->get()->get_value_time() << " to " << harvest_range.rbegin()->get()->get_value_time() << ", expected " << expected_rows << " rows.");

        __tbb_pfor(row_ix, 0, expected_rows,
            const auto label_start_iter = harvest_range.begin() + row_ix;
            const bpt::ptime label_start_time = label_start_iter->get()->get_value_time();
            if (label_start_time <= last_modeled_value_time) {
                LOG4_DEBUG("Skipping already modeled row with value time " << label_start_time);
                AVEC_PUSH_TS(shedded_rows, row_ix, mx_shedded_rows);
                continue;
            }
            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            arma::rowvec features_row;
            arma::rowvec labels_row(PROPS.get_multistep_len());
            const auto label_aux_start_iter = lower_bound(aux_data_range.get_container(), aux_decon_hint + row_ix * main_to_aux_period_ratio, label_start_time);
            if (label_aux_start_iter == aux_data_range.get_container().end()) {
                LOG4_ERROR("Can't find aux labels start " << label_start_time);
                AVEC_PUSH_TS(shedded_rows, row_ix, mx_shedded_rows);
                continue;
            } else if (label_aux_start_iter->get()->get_value_time() >= label_start_time + .5 * main_queue_resolution) {
                LOG4_ERROR("label_aux_start_iter->get()->get_value_time() > label_start_time " << label_aux_start_iter->get()->get_value_time() << " > " << label_start_time + .7 * main_queue_resolution);
                AVEC_PUSH_TS(shedded_rows, row_ix, mx_shedded_rows);
                continue;
            }
            const auto label_aux_end_iter = lower_bound(aux_data_range.get_container(), label_aux_start_iter, label_start_time + main_queue_resolution);
            const auto feature_end_iter = lower_bound_back(aux_data_range.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
            if (prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row) and
                prepare_labels(level, label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, labels_row, aux_queue_res)) {

                std::unique_lock l(mx_resize);
                if (features_row.size() % lag > 0) LOG4_ERROR("features_row.size() != lag " << features_row.size() << " != " << lag);
                if (features.n_rows != expected_rows || features.n_cols != features_row.size()) features.set_size(expected_rows, features_row.size()); // Last row is special, contains the last known value
                if (labels.n_rows != expected_rows || labels.n_cols != PROPS.get_multistep_len()) labels.set_size(expected_rows, PROPS.get_multistep_len());
                if (last_knowns.n_rows != expected_rows || last_knowns.n_cols != 1) last_knowns.set_size(expected_rows, 1);
                l.unlock();

                features.row(row_ix) = features_row;
                const auto p_anchor_row = std::prev(feature_end_iter)->get();
#ifdef EMO_DIFF
                labels_row = labels_row - p_anchor_row->get_value(level);
#endif
                labels.row(row_ix) = labels_row;
                last_knowns(row_ix, 0) = p_anchor_row->get_value(level);
                label_times[row_ix] = label_start_time;
                if (row_ix >= harvest_range.distance() - 1)
                    LOG4_DEBUG(
                            "Last data row " << row_ix << ", value time " << label_start_time << ", label aux start time " << label_aux_start_iter->get()->get_value_time() << ", last known time " << p_anchor_row->get_value_time() <<
                                             ", last last-known value " << last_knowns(row_ix, 0) << ", label " << labels.row(row_ix).back() << ", level " << level);
            } else {
                LOG4_WARN("For row at " << label_start_time << " can't assemble features " << arma::size(features_row) << " or labels " << arma::size(labels_row) << ", skipping.");
                AVEC_PUSH_TS(shedded_rows, row_ix, mx_shedded_rows);
            }
        );
        if (!shedded_rows.empty()) {
            LOG4_DEBUG("Shedding rows " << shedded_rows);
            features.shed_rows(shedded_rows);
            labels.shed_rows(shedded_rows);
            last_knowns.shed_rows(shedded_rows);
            std::vector<bpt::ptime> clean_label_times;
            for (arma::uword r = 0; r < label_times.size(); ++r) {
                if (!arma::find(shedded_rows == r).is_empty()) continue;
                clean_label_times.emplace_back(label_times[r]);
            }
            label_times = clean_label_times;
        }
        if (!labels.empty() && !features.empty() && !last_knowns.empty()) LOG4_DEBUG("Returning labels " << common::present(labels) << ", features " << common::present(features) << ", last knowns " << common::present(last_knowns) << " for level " << level);
        return std::make_tuple(features, labels, last_knowns, label_times);
    };

    for (datamodel::datarow_range harvest_range = main_data_range;
        all_labels.n_rows < req_rows && harvest_range.begin()->get()->get_value_time() >= main_data_range.get_container().begin()->get()->get_value_time() && harvest_range.begin()->get()->get_value_time() > last_modeled_value_time;
        harvest_range = datamodel::datarow_range(harvest_range.begin() - req_rows + all_labels.n_rows, harvest_range.begin(), harvest_range.get_container()))
    {
        arma::mat features, labels, last_knowns;
        std::vector<bpt::ptime> label_times;
        std::tie(features, labels, last_knowns, label_times) = harvest_rows(harvest_range);
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
        auto label_aux_start_iter = lower_bound(aux_data_range.get_container(), aux_decon_hint + main_data_range.distance() * .5, (main_data_range.begin() + main_data_range.distance() - 1)->get()->get_value_time() + main_queue_resolution + main_queue_resolution * (1. - OFFSET_PRED_MUL));
        if (label_aux_start_iter == aux_data_range.get_container().end()) --label_aux_start_iter;
        const bpt::ptime label_start_time = label_aux_start_iter->get()->get_value_time();
        const auto feature_end_iter = lower_bound_back(aux_data_range.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
        arma::rowvec features_row;
        arma::rowvec labels_row(PROPS.get_multistep_len());
        if (prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row)) {
            all_features = arma::join_cols(all_features, features_row);
            labels_row.fill(label_aux_start_iter->get()->get_value(level));
            const auto p_anchor_row = std::prev(feature_end_iter)->get();
#ifdef EMO_DIFF
            labels_row = labels_row - p_anchor_row->get_value(level);
#endif
            all_labels = arma::join_cols(all_labels, labels_row);
            all_last_knowns = arma::join_cols(all_last_knowns, arma::rowvec(p_anchor_row->get_value(level)));
            if (!level) all_times.emplace_back(label_start_time);
            LOG4_DEBUG("Temporary data last row, time " << label_start_time << " anchor time " << p_anchor_row->get_value_time());
        } else {
            LOG4_ERROR("Failed adding temporary row with time " << label_start_time << ", features size " << arma::size(features_row) << ", labels " << labels_row);
        }
    }
#endif
    if (all_labels.empty() or all_features.empty())
        LOG4_WARN("No new data to prepare for training, labels size " << arma::size(all_labels) << ", features size " << arma::size(all_features));
    else
        LOG4_DEBUG("Prepared level " << level << ", labels " << arma::size(all_labels) << ", features " << arma::size(all_features) << ", last knowns " << arma::size(all_last_knowns));

#if 0 // Save training data to file
    if (level == 0) {
        static size_t call_ct;
        labels.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/labels_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        features.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/features_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        p_last_knowns.save(
                svr::common::formatter() << "/mnt/slowstore/var/tmp/last_knowns_" << level << "_" << call_ct << ".out", arma::csv_ascii);
        ++call_ct;
    }
#endif

    return true;
}


void
ModelService::get_features_row(
        const bpt::ptime &pred_time,
        const datarow_range &aux_data_range,
        const svr::datamodel::dq_scaling_factor_container_t &aux_scaling_factors,
        const std::set<size_t> &adjacent_levels,
        const size_t current_level,
        const bpt::time_duration &max_gap,
        const double main_to_aux_period_ratio,
        const size_t lag,
        const bpt::time_duration &main_queue_resolution,
        arma::rowvec &features_row)
{
    LOG4_BEGIN();

    if (aux_data_range.get_container().empty()) {
        LOG4_ERROR("Features queue is empty to predict " << pred_time);
        features_row.clear();
        return;
    }

    const bpt::ptime last_feat_expected_time = pred_time - main_queue_resolution * OFFSET_PRED_MUL;
    const auto feature_aux_end_iter = lower_bound_back(aux_data_range.get_container(), last_feat_expected_time);
    const size_t distance_from_start = std::distance(aux_data_range.get_container().begin(), feature_aux_end_iter);
    if (distance_from_start < (1 + lag) * QUANTIZE_FIXED) {
        LOG4_ERROR("Not enough data to predict " << pred_time << ", found " << distance_from_start);
        features_row.clear();
        return;
    }

    if (std::prev(feature_aux_end_iter)->get()->get_value_time() != last_feat_expected_time - onesec) {
        LOG4_ERROR(
                "Last feature time " << std::prev(feature_aux_end_iter)->get()->get_value_time() << " does not match expected " << last_feat_expected_time - onesec << " for " << pred_time << " of " <<
                                     aux_data_range.get_container().size() << " rows, starting " << aux_data_range.get_container().front()->get_value_time() << ", ending " << aux_data_range.get_container().back()->get_value_time() << ", p_predictions will be of lower quality, skipping!");
        features_row.clear();
        return;
    }

#ifndef NEW_SCALING
    if (!prepare_features(adjacent_levels, lag, feature_aux_end_iter, max_gap, main_to_aux_period_ratio, aux_scaling_factors, features_row)) {
#else
    if (!prepare_features(adjacent_levels, lag, feature_aux_end_iter, max_gap, main_to_aux_period_ratio, features_row)) {
#endif
        LOG4_ERROR("Failed preparing features for time " << pred_time);
        features_row.clear();
        return;
    }

    if (features_row.size() % lag) LOG4_ERROR("Features row size dubious " << arma::size(features_row));

    LOG4_DEBUG("Prepared prediction features for label at " << pred_time << " level " << current_level << " row size " << arma::size(features_row) <<
                " features until " << std::prev(feature_aux_end_iter)->get()->get_value_time() << ", main to aux ratio " << main_to_aux_period_ratio << ", lag " << lag);

    LOG4_END();
}


void
ModelService::train(
        SVRParameters_ptr &p_svr_parameters, Model_ptr &p_model, const matrix_ptr &p_features,
        const matrix_ptr &p_labels, const bpt::ptime &new_last_modeled_value_time)
{
    if (p_svr_parameters->get_skip()) {
        LOG4_DEBUG("Skipping training on model " << p_model->get_decon_level());
        p_model->set_last_modified(bpt::second_clock::local_time());
        return;
    }
    if (p_labels->empty() or p_features->empty() or p_labels->n_rows != p_features->n_rows) {
        LOG4_ERROR(
                "Invalid learning data, labels matrix row count is " << arma::size(*p_labels) << " training features matrix row count is "
                                                                     << arma::size(*p_features));
        return;
    }

    if (p_model->get_svr_model())
        train_online(*p_features, *p_labels, p_model->get_svr_model());
    else
        train_batch(p_svr_parameters, p_model, p_features, p_labels);


    p_model->set_last_modeled_value_time(new_last_modeled_value_time);
    p_model->set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO(
            "Finished training model for level " << p_model->get_svr_model()->get_svr_parameters().get_decon_level()
             << ", input queue name " << p_model->get_svr_model()->get_svr_parameters().get_input_queue_table_name()
             << ", input queue column " << p_model->get_svr_model()->get_svr_parameters().get_input_queue_column_name()
             << ", samples trained number " << p_model->get_svr_model()->get_samples_trained_number()
             << ", last modeled value time " << new_last_modeled_value_time
             << ", last modified time " << p_model->get_last_modified());
}


void
ModelService::train_online(
        const arma::mat &features_data,
        const arma::mat &labels_data,
        OnlineMIMOSVR_ptr &p_svr_model)
{
#ifdef LAST_KNOWN_LABEL
    for (size_t r = 0; r < labels_data.n_rows - 1; ++r) // TODO Implement online batch training of multiple rows
        PROFILE_EXEC_TIME(
                (void) p_svr_model->learn(features_data.row(r), labels_data.row(r), false, PROPS.get_dont_update_r_matrix()),
                "Online SVM train");
    p_svr_model->learn(features_data.row(features_data.n_rows - 1), labels_data.row(labels_data.n_rows - 1), true, PROPS.get_dont_update_r_matrix());
#else
    for (size_t r = 0; r < labels_data.n_rows; ++r) // TODO Implement online batch training of multiple rows
        PROFILE_EXEC_TIME(p_svr_model->learn(features_data.row(r), labels_data.row(r), false, PROPS.get_dont_update_r_matrix()), "Online SVM train");
#endif
}


void
ModelService::train_batch(
        SVRParameters_ptr &p_svr_parameters,
        Model_ptr &p_model,
        const matrix_ptr &p_features,
        const matrix_ptr &p_labels)
{
    LOG4_BEGIN();

    const bool save_parameters = p_svr_parameters->get_svr_kernel_param() == 0 || p_svr_parameters->get_svr_kernel_param2() == 0;
    auto p_svr_model = std::make_shared<OnlineMIMOSVR>(
            p_svr_parameters, p_features, p_labels, not PROPS.get_dont_update_r_matrix(), nullptr, false, MimoType::single, PROPS.get_multistep_len());

    if (save_parameters) {
        APP.svr_parameters_service.remove(p_svr_model->get_svr_parameters());
        APP.svr_parameters_service.save(p_svr_model->get_svr_parameters());
    }

    p_model->set_svr_model(p_svr_model);

    LOG4_END();
}


const datarow_range
ModelService::prepare_feat_range(
        const DataRow::container &data,
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
    return datarow_range(
            remove_constness(const_cast<DataRow::container &>(data), it_lookback),
            remove_constness(const_cast<DataRow::container &>(data), it_last_feature),
            const_cast<DataRow::container &>(data));
}


void
ModelService::check_feature_data(
        const DataRow::container &data,
        const DataRow::container::const_iterator &iter,
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
        const DataRow::container &data,
        const DataRow::container::const_iterator &iter,
        const bpt::time_duration &max_gap,
        const bpt::ptime &feat_time)
{
    if (iter == data.end() || iter->get()->get_value_time() - feat_time > max_gap)
        THROW_EX_FS(svr::common::insufficient_data,
                "Can't find data for prediction features. Needed value for " + bpt::to_simple_string(feat_time) +
                ", nearest data available is " +
                (iter == data.end() ? "not found" : "at " + bpt::to_simple_string(iter->get()->get_value_time())));
}

arma::colvec
ModelService::predict(
        const Model_ptr &p_model,
        const data_row_container &main_decon_data,
        const std::vector<data_row_container_ptr> &aux_data_rows_containers,
        const boost::posix_time::ptime &prediction_time,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap)
{
    if (!p_model->get_svr_model() or p_model->get_svr_model()->get_svr_parameters().get_skip())
    {
        LOG4_DEBUG("Skipping prediction on model " << p_model->get_decon_level());
        return arma::colvec(PROPS.get_multistep_len()).fill(0.);
    }

    LOG4_DEBUG("Predicting using " << p_model->to_string());

    if (main_decon_data.empty())
        THROW_EX_F(svr::common::insufficient_data,"Decon queue is empty for model " << p_model->get_decon_level());

    const auto lag_count = p_model->get_svr_model()->get_svr_parameters().get_lag_count();
    const auto lookback_range = prepare_feat_range(main_decon_data, max_gap, prediction_time, lag_count);

    LOG4_DEBUG(
            "Predicting time " << prediction_time <<
                               ", model for level " << p_model->get_decon_level() <<
                               ", learning levels " << svr::common::deep_to_string(p_model->get_learning_levels()) <<
                               ", samples trained " << p_model->get_svr_model()->get_samples_trained_number() <<
                               ", lookback range begin " << lookback_range.begin()->get()->get_value_time() <<
                               ", lookback range end " << lookback_range.rbegin()->get()->get_value_time() <<
                               ", lookback range size " << lookback_range.get_container().size() <<
                               ", lookback range distance " << lookback_range.distance() <<
                               ", lookback range duration " << lookback_range.rbegin()->get()->get_value_time() - lookback_range.begin()->get()->get_value_time());

    std::vector<datarow_range> aux_lookback_ranges;
    for (const data_row_container_ptr &p_aux_data: aux_data_rows_containers)
        if (p_aux_data) aux_lookback_ranges.push_back(prepare_feat_range(*p_aux_data, max_gap, prediction_time, lag_count));
        else LOG4_WARN("Aux data pointer is empty!");

    if (lookback_range.rbegin()->get()->get_value_time() < prediction_time - resolution) {
        if (lookback_range.rbegin()->get()->get_value_time() < prediction_time - resolution * lag_count)
            THROW_EX_F(svr::common::missing_data_fatal,
                        "Missing more than half lag values in lookback queue. Feature missing to predict "
                              << prediction_time << " last lag value " << lookback_range.rbegin()->get()->get_value_time());
        else
            THROW_EX_F(svr::common::missing_data,
                    "In main decon queue: Feature missing to predict " << prediction_time << " " << lookback_range.rbegin()->get()->get_value_time());
    }

    for (auto aux_lookback_range: aux_lookback_ranges) {
        if (aux_lookback_range.rbegin()->get()->get_value_time() < prediction_time - resolution) {
            if (aux_lookback_range.rbegin()->get()->get_value_time() < prediction_time - resolution * lag_count)
                THROW_EX_F(svr::common::missing_data_fatal, "Missing more than lag values in aux lookback queue. Feature missing to predict " << prediction_time << " last lag value " << lookback_range.rbegin()->get()->get_value_time());
            else
                THROW_EX_F(svr::common::missing_data, "In aux decon queue: Feature missing to predict in aux queue " << prediction_time << ", last aux queue time " << aux_lookback_range.rbegin()->get()->get_value_time());
        }
    }

    arma::rowvec prediction_vector;
    //get_features_row(
    //        lookback_range, p_model->get_learning_levels(), p_model->get_decon_level(), max_gap, prediction_vector);
    if (prediction_vector.size() < 1) THROW_EX_F(svr::common::missing_data_fatal, "Unable to assemble features for " << prediction_time);
#ifdef WHITEBOX_TEST // WHITEBOX_TEST
    {
        std::stringstream ss_row;
        ss_row.precision(std::numeric_limits<double>::max_digits10 + 1);
        ss_row << "Predicting value time " << prediction_time << ", Level: " << p_model->get_decon_level() << ", Features: ";
        for (int i = 0; i < prediction_vector.size(); ++i) ss_row << i << ":" << prediction_vector.get_value(i) << ", ";
        LOG4_DEBUG(ss_row.str());
    }
#endif /* WHITEBOX_TEST */

#ifdef OUTPUT_LIBSVM_TRAIN_DATA
    {
        static size_t call_counter;
        char cm_separator = ',';
        std::stringstream ss;
        using namespace svr::common;
        auto old_cm_separator = cm_separator;
        cm_separator = '_';
        const auto &p_svr_parameters = p_model->get_svr_model()->get_svr_parameters();
        ss <<
           "predict_dataset_" << p_svr_parameters.get_dataset_id() << "_" <<
           p_svr_parameters.get_input_queue_table_name() << "_" << p_svr_parameters.get_input_queue_column_name() <<
           "_level_" << p_model->get_decon_level() <<
           "_adjacent_levels_" << p_model->get_learning_levels().size() <<
           "_lag_" << p_svr_parameters.get_lag_count() <<
           "_call_" << call_counter++ <<
           ".libsvm.txt";
        std::ofstream of(ss.str());
        of.precision(std::numeric_limits<double>::max_digits10);
        of << prediction_time;
        for (ssize_t col_ix = 0; col_ix < prediction_vector.size(); ++col_ix)
            of << " " << col_ix << ":" << prediction_vector[col_ix];
        of << "\n";
        cm_separator = old_cm_separator;
    }
#endif /* #ifdef OUTPUT_LIBSVM_TRAIN_DATA */

    arma::mat ret_values;
    PROFILE_EXEC_TIME(ret_values = p_model->get_svr_model()->chunk_predict(prediction_vector), "Predict MIMO values");
    {
        std::stringstream ss;
        ss.precision(11);
        ret_values.raw_print(ss);
        LOG4_DEBUG("Predicted values for level " << p_model->get_decon_level() << " are " << ss.str());
    }

#if 0
    if (p_model->get_decon_level() == 0) {
        static size_t call_ct = 0;
        std::stringstream ss_row;
        prediction_vector.raw_print(ss_row);
        std::stringstream ss_predictions;
        arma::vectorise(ret_values).raw_print(ss_predictions);
        std::stringstream ss_file;
        ss_file << "/mnt/faststore/prediction_vector_level_0_call_" << call_ct << ".csv";
        LOG4_FILE(ss_file.str(),
                "Prediction row: " << prediction_time << " prediction: " << ss_predictions.str() << " features: " << prediction_vector.size() << ": " << ss_row.str());
        call_ct++;
    }
#endif

    return arma::vectorise(ret_values);
}

// TODO Clean up
// Used by paramtune
data_row_container
ModelService::predict(
        const Model_ptr &p_model,
        const datarow_range &main_data_range,
        const std::vector<datarow_range> &aux_data_ranges,
        const ptimes_set_t &prediction_times,
        const bpt::time_duration &max_gap,
        const bpt::time_duration &resolution)
{
    LOG4_BEGIN();

    if (main_data_range.distance() < 1) LOG4_THROW("Decon queue is empty.");
    if (!p_model->get_svr_model()) LOG4_THROW("Model is not initialized!");

    const auto lag_count = p_model->get_svr_model()->get_svr_parameters().get_lag_count();
    { // Do initial sanity checks of input data
        const auto feat_time = *prediction_times.begin();
        (void) find_nearest_before(main_data_range.get_container(), feat_time, max_gap, lag_count);
        for (auto &aux_data_range: aux_data_ranges)
            (void) find_nearest_before(aux_data_range.get_container(), feat_time, max_gap, lag_count);
    }

    data_row_container result;
    const auto multistep_len = PROPS.get_multistep_len();
    LOG4_DEBUG("Predicting range " << *prediction_times.begin() << " to " << *prediction_times.rbegin());
    const auto multistep_prediction_times = to_multistep_times(prediction_times, resolution, multistep_len);
    arma::mat prediction_matrix(multistep_prediction_times.size(), 0);
    for (size_t time_ix = 0; time_ix < multistep_prediction_times.size(); ++time_ix) { // TODO Parallelize
        const auto &prediction_time = *std::next(multistep_prediction_times.begin(), time_ix);
        LOG4_TRACE("Preparing prediction features for " << prediction_time);
        try {
            //const auto pred_range = prepare_feat_range(
            //        main_data_range.get_container(), max_gap, prediction_time, lag_count);
            arma::rowvec prediction_vector;
            //get_features_row(
            //        pred_range, p_model->get_learning_levels(), p_model->get_decon_level(), max_gap, prediction_vector);
            if (not prediction_matrix.n_cols) prediction_matrix.reshape(prediction_matrix.n_cols, prediction_vector.size());
            prediction_matrix.row(time_ix) = prediction_vector;
            LOG4_DEBUG("Added row for time " << prediction_time << " successfully.");

#ifdef OUTPUT_TRAINING_DATA
            {
                if (p_model->get_decon_level() == 0) {
                    std::stringstream ss_row;
                    for (size_t i = 0; i < prediction_vector.size(); ++i)
                        ss_row << prediction_vector(i) << " ";
                    LOG4_FILE("/mnt/faststore/prediction_vector_level_0.csv", "Prediction row " << time_ix << ": " << prediction_time << " values "
                                                << prediction_vector.size() << ": " << ss_row.str());
                }
            }
#endif

        } catch (const std::exception &ex) {
            LOG4_WARN(
                    "Failed preparing features for prediction time " << prediction_time <<
                    ", skipping. " << ex.what());
        }
    }

    for (const auto &prediction_time: prediction_times)
        result.push_back(std::make_shared<DataRow>(prediction_time));

    LOG4_DEBUG(
            "Predicting range " << *prediction_times.begin() << " to " << *prediction_times.rbegin() <<
                                ", model: " << p_model->to_string() <<
                                ", learning levels " << svr::common::deep_to_string(p_model->get_learning_levels()) <<
                                ", samples trained " << p_model->get_svr_model()->get_samples_trained_number() <<
                                ", data range begin " << main_data_range.begin()->get()->get_value_time() <<
                                ", data range end " << main_data_range.rbegin()->get()->get_value_time() <<
                                ", data range size " << main_data_range.get_container().size() <<
                                ", data range distance " << main_data_range.distance() <<
                                ", data range duration " << main_data_range.rbegin()->get()->get_value_time() - main_data_range.begin()->get()->get_value_time() <<
                                ", aux lookback ranges count " << aux_data_ranges.size() <<
                                ", features matrix columns count " << prediction_matrix.n_cols <<
                                ", features matrix rows count " << prediction_matrix.n_cols);

    const auto result_values = p_model->get_svr_model()->chunk_predict(prediction_matrix);

    if (result.size() < decltype(result.size())(prediction_times.size()))
        LOG4_WARN("Predicted values size " << result.size() << " less than prediction times requested " << prediction_times.size());

    LOG4_TRACE("Received values " << result_values.n_cols * result_values.n_rows << " vs count" << prediction_times.size());

    ssize_t row_ctr = 0;
    ssize_t mimo_ctr = -1;
    for (auto &row: result) {
        if (multistep_prediction_times.find(row.get()->get_value_time()) != multistep_prediction_times.end()) {
            row_ctr = 0;
            ++mimo_ctr;
            row.get()->set_value(0, result_values(mimo_ctr, 0));
        }
        else {
            row.get()->set_value(0, result_values(mimo_ctr, row_ctr++));
        }
    }

    LOG4_END();

    return result;
}


/*
 * TODO Deadlock in algorithm below, fix!
 */
#if 1

static arma::colvec
predict_wrapper(
        const Model_ptr &p_model,
        const data_row_container &p_main_decon_data,
        const std::vector<data_row_container_ptr> &aux_data_rows_containers,
        const boost::posix_time::ptime &prediction_time,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap)
{
    return ModelService::predict(
            p_model, p_main_decon_data, aux_data_rows_containers, prediction_time, resolution, max_gap);
}

arma::mat
ModelService::predict(
        const std::vector<Model_ptr> &models,
        const boost::posix_time::ptime &prediction_time,
        const boost::posix_time::time_duration &resolution,
        const boost::posix_time::time_duration &max_gap,
        const svr::datamodel::DataRow::container &main_decon_data,
        const std::vector<data_row_container_ptr> &aux_decon_data)
{
    LOG4_DEBUG("Predicting MIMO time " << prediction_time);

#ifdef PRINTOUT_PER_LEVEL_VALUES
    {
        std::stringstream ss;
        ss << "Actual values at " << main_decon_data.rbegin()->get()->get_value_time() << ": ";
        for (const auto val: main_decon_data.rbegin()->get()->get_values()) ss << val << ", ";
        LOG4_DEBUG(ss.str());
    }
#endif

    const auto multistep_len = PROPS.get_multistep_len();
    arma::mat predicted_values(multistep_len, models.size());
    predicted_values.fill(0);
    std::vector<svr::future<arma::colvec>> results;
    for (size_t model_counter = 0; model_counter < models.size(); ++model_counter)
        results.push_back(svr::async(predict_wrapper, std::cref(models[model_counter]), main_decon_data, aux_decon_data, prediction_time, resolution, max_gap));

    for (size_t model_counter = 0; model_counter < results.size(); ++model_counter) {
        try {
            predicted_values.col(model_counter) = results[model_counter].get();
        } catch (const svr::common::missing_data &ex) {
            LOG4_WARN("Caught missing value exception. " << ex.what());
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception &ex) {
            LOG4_ERROR(
                    "Failed predicting all coefficients for time " << prediction_time << ". " << ex.what());
            throw;
        }
    }

#ifdef PRINTOUT_PER_LEVEL_VALUES
    {
        std::stringstream ss;
        ss << "Predicted values at " << prediction_time << ": ";
        for (ssize_t i = 0; i < predicted_values.get_length_cols(); ++i) ss << predicted_values(0, i) << ", ";
        LOG4_DEBUG(ss.str());
    }
#endif

    return predicted_values;
}

#else

vmatrix<double>
ModelService::predict(
        const std::vector<Model_ptr> &models,
        const boost::posix_time::ptime &prediction_time,
        const boost::posix_time::time_duration &resolution,
        const boost::posix_time::time_duration &max_gap,
        const svr::datamodel::DataRow::container &main_decon_data,
        const std::vector<data_row_container_ptr> &aux_decon_data)
{
    LOG4_DEBUG("Predicting MIMO time " << prediction_time);
    const auto multistep_len = PROPS.get_multistep_len();
    vmatrix<double> predicted_values(multistep_len, models.size());
    std::vector<std::future<vektor<double>>> results(models.size());
    for (size_t model_counter = 0; model_counter < models.size(); ++model_counter)
        predicted_values.set_col_copy_at(
                predict(models[model_counter], main_decon_data, aux_decon_data,
                                                 prediction_time, resolution, max_gap), model_counter);

    LOG4_END();
    return predicted_values;
}

#endif

} // business
} // svr
