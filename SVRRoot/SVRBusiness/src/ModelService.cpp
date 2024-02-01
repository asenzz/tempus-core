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

void ModelService::load(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr &p_ensemble, datamodel::Model_ptr &p_model)
{
    const auto default_num_chunks = DEFAULT_SVRPARAM_DECREMENT_DISTANCE / p_dataset->get_chunk_size();

    auto params = APP.svr_parameters_service.get_by_dataset_column_level(
            p_ensemble->get_dataset_id(), p_ensemble->get_decon_queue()->get_input_queue_column_name(), p_model->get_decon_level());

    std::deque<OnlineMIMOSVR_ptr> svr_models(p_dataset->get_gradients());
#pragma omp parallel for
    for (size_t g = 0; g < svr_models.size(); ++g) {
        datamodel::t_param_set_ptr grad_params = SVRParametersService::slice(params, std::numeric_limits<size_t>::max(), g);
        if (grad_params->empty())
            for (size_t c = 0; c < default_num_chunks; ++c)
                grad_params->emplace(std::make_shared<datamodel::SVRParameters>(
                        0, p_dataset->get_id(),
                        p_ensemble->get_decon_queue()->get_input_queue_table_name(),
                        p_ensemble->get_decon_queue()->get_input_queue_column_name(),
                        p_model->get_decon_level(), c, g));
        svr_models[g] = std::make_shared<svr::OnlineMIMOSVR>(grad_params, p_dataset->get_multiout(), p_dataset->get_chunk_size());
    }

    p_model->set_learning_levels(
            common::get_adjacent_indexes(
                    p_model->get_decon_level(), svr_models.front()->get_params().get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels()));
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
    try {
        return model_dao.get_all_ensemble_models(ensemble_id);
    } catch (...) {
        LOG4_ERROR("Cannot read models from the database.");
        return {};
    }
}

datamodel::Model_ptr ModelService::get_model(const bigint ensemble_id, const size_t decon_level)
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
        LOG4_WARN("Corrupt value " << result << " at " << row_iter->get()->get_value_time() << " pos " << pos << " level " <<
                    level << " lag count " << lag << " quantization mul " << quantization_mul);
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
    LOG4_TRACE("Processing row with lag " << lag << " until " << std::prev(end_iter)->get()->get_value_time());
    const auto r_start = row.size();
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
        row.cols(r_start + adj_ix * lag, r_start + (adj_ix + 1) * lag - 1) = level_row;
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
        const double main_to_aux_period_ratio,
        const bpt::ptime &last_modeled_value_time,
        const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    const auto aux_queue_res = main_queue_resolution / main_to_aux_period_ratio;
    const size_t req_rows = main_data.distance();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    LOG4_DEBUG("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.begin()->get()->get_value_time() <<
                                  " until " << main_data.rbegin()->get()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
    auto harvest_rows = [&](const datamodel::datarow_range &harvest_range) {
        tbb::concurrent_vector<arma::uword> shedded_rows;
        const size_t expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, PROPS.get_multistep_len()), features(expected_rows, 0), last_knowns(expected_rows, 1);
        std::deque<bpt::ptime> label_times;
        if (!level) label_times.resize(expected_rows);
        LOG4_DEBUG("Processing range " << harvest_range.begin()->get()->get_value_time() << " to " << harvest_range.rbegin()->get()->get_value_time() << ", expected " << expected_rows << " rows.");

#pragma omp parallel for
        for (size_t rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_iter = *(harvest_range.begin() + rowix);
            const bpt::ptime label_start_time = label_start_iter->get_value_time();
            if (label_start_time <= last_modeled_value_time) {
                LOG4_DEBUG("Skipping already modeled row with value time " << label_start_time);
                shedded_rows.emplace_back(rowix);
                continue;
            }
            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            arma::rowvec labels_row(PROPS.get_multistep_len());
            const auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux.begin() + rowix * main_to_aux_period_ratio, label_start_time);
            if (label_aux_start_iter == labels_aux.get_container().end()) {
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
#pragma omp parallel for ordered schedule(static, 1)
            for (const auto &f: features_aux) {
                const auto feature_end_iter = lower_bound_back(f.get_container(), f.end(), label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
#pragma omp ordered
                feat_rc &= prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row);
            }
            if (feat_rc && prepare_labels(level, label_aux_start_iter, label_aux_end_iter, label_start_time, label_start_time + main_queue_resolution, labels_row, aux_queue_res)) {
                if (features_row.size() % lag > 0) LOG4_ERROR("features_row.size() != lag " << features_row.size() << " != " << lag);
#pragma omp critical
                {
                    if (features.n_rows != expected_rows || features.n_cols != features_row.size()) features.set_size(expected_rows, features_row.size());
                    if (labels.n_rows != expected_rows || labels.n_cols != PROPS.get_multistep_len()) labels.set_size(expected_rows, PROPS.get_multistep_len());
                    if (last_knowns.n_rows != expected_rows || last_knowns.n_cols != 1) last_knowns.set_size(expected_rows, 1);
                }

                features.row(rowix) = features_row;
                const auto p_anchor_row = *std::prev(lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * OFFSET_PRED_MUL));
#ifdef EMO_DIFF
                labels_row = labels_row - p_anchor_row->get_value(level);
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
            LOG4_DEBUG("Returning labels " << common::present(labels) << ", features " << common::present(features) << ", last knowns " << common::present(last_knowns) << " for level " << level);
        return std::make_tuple(features, labels, last_knowns, label_times);
    };

    for (auto harvest_range = main_data;
        all_labels.n_rows < req_rows && harvest_range.begin()->get()->get_value_time() >= main_data.get_container().begin()->get()->get_value_time()
        && harvest_range.begin()->get()->get_value_time() > last_modeled_value_time;
        harvest_range.set_range(harvest_range.begin() - req_rows + all_labels.n_rows, harvest_range.begin()))
    {
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
        auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux.begin() + main_data.distance() * .5, (main_data.begin() + main_data.distance() - 1)->get()->get_value_time() + main_queue_resolution + main_queue_resolution * (1. - OFFSET_PRED_MUL));
        if (label_aux_start_iter == labels_aux.get_container().end()) --label_aux_start_iter;
        const bpt::ptime label_start_time = label_aux_start_iter->get()->get_value_time();
        arma::rowvec features_row;
        bool feat_rc = true;
        for (const auto &f: features_aux) {
            const auto feature_end_iter = lower_bound_back(f.get_container(), f.end(), label_start_time - main_queue_resolution * OFFSET_PRED_MUL);
            feat_rc &= prepare_features(adjacent_levels, lag, feature_end_iter, max_gap, main_to_aux_period_ratio, features_row);
        }

        if (feat_rc) {
            arma::rowvec labels_row(PROPS.get_multistep_len());
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
        LOG4_DEBUG("Prepared level " << level << ", labels " << arma::size(all_labels) << ", features " << arma::size(all_features) << ", last knowns " << arma::size(all_last_knowns));

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
        const bpt::ptime &pred_time,
        const std::deque<datamodel::DeconQueue_ptr> &aux_decons,
        const std::set<size_t> &adjacent_levels,
        const size_t level,
        const bpt::time_duration &max_gap,
        const double main_to_aux_period_ratio,
        const size_t lag,
        const bpt::time_duration &main_resolution,
        arma::rowvec &features_row)
{
    LOG4_BEGIN();

    if (aux_decons.empty()) {
        LOG4_ERROR("Features queue is empty to predict " << pred_time);
        features_row.clear();
        return;
    }

    const bpt::ptime last_feat_expected_time = pred_time - main_resolution * OFFSET_PRED_MUL;
#pragma omp parallel for ordered schedule(static, 1)
    for (const auto &d: aux_decons) {
        const auto feature_aux_end_iter = lower_bound_back(d->get_data(), last_feat_expected_time);
        const size_t distance_from_start = std::distance(d->get_data().begin(), feature_aux_end_iter);
        if (distance_from_start < (1 + lag) * QUANTIZE_FIXED)
            LOG4_THROW("Not enough data to predict " << pred_time << ", found " << distance_from_start);

        if (std::prev(feature_aux_end_iter)->get()->get_value_time() != last_feat_expected_time - onesec)
            LOG4_THROW(
                    "Last feature time " << std::prev(feature_aux_end_iter)->get()->get_value_time() << " does not match expected " << last_feat_expected_time - onesec
                     << " for " << pred_time << " of " << d->get_data().size() << " rows, starting " << d->get_data().front()->get_value_time() << ", ending " << d->get_data().back()->get_value_time() << ", p_predictions will be of lower quality, skipping!");
#pragma omp ordered
        if (!prepare_features(adjacent_levels, lag, feature_aux_end_iter, max_gap, main_to_aux_period_ratio, features_row))
            LOG4_THROW("Failed preparing features for time " << pred_time);

        LOG4_DEBUG("Prepared prediction features for label at " << pred_time << " level " << level << " row size " << arma::size(features_row) <<
                                                                " features until " << std::prev(feature_aux_end_iter)->get()->get_value_time() << ", main to aux ratio " << main_to_aux_period_ratio << ", lag " << lag);
    }

    if (features_row.size() % lag) LOG4_ERROR("Features row size dubious " << arma::size(features_row));

    LOG4_END();
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
        train_online(*p_features, *p_labels, p_model->get_gradients());
    else
        train_batch(p_model->get_param_set_ptr(), p_model, p_features, p_labels);


    p_model->set_last_modeled_value_time(new_last_modeled_value_time);
    p_model->set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO(
            "Finished training model for level " << p_model->get_params().get_decon_level()
             << ", input queue name " << p_model->get_params().get_input_queue_table_name()
             << ", input queue column " << p_model->get_params().get_input_queue_column_name()
             << ", last modeled value time " << new_last_modeled_value_time
             << ", last modified time " << p_model->get_last_modified());
}


void
ModelService::train_online(
        const arma::mat &features_data,
        const arma::mat &labels_data,
        std::deque<OnlineMIMOSVR_ptr> &svr_models)
{
#ifdef LAST_KNOWN_LABEL
    for (size_t r = 0; r < labels_data.n_rows - 1; ++r) // TODO Implement online batch training of multiple rows
        PROFILE_EXEC_TIME(
                (void) p_svr_model->learn(features_data.row(r), labels_data.row(r), false, PROPS.get_dont_update_r_matrix()),
                "Online SVM train");
    p_svr_model->learn(features_data.row(features_data.n_rows - 1), labels_data.row(labels_data.n_rows - 1), true, PROPS.get_dont_update_r_matrix());
#else
    for (auto &m: svr_models)
        PROFILE_EXEC_TIME(m->learn(features_data, labels_data, false), "Online SVM train");
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

#pragma omp parallel for
    for (size_t g = 0; g < p_model->get_gradient_count(); ++g) {
        auto gradient_params = SVRParametersService::slice(*p_param_set, std::numeric_limits<size_t>::max(), g);
        if (gradient_params->empty()) LOG4_THROW("Parameters for model " << *p_model << " not initialized.");
        bool save_parameters = false;
        for (auto &p: *gradient_params)
            save_parameters |= p->get_svr_kernel_param() == 0;
        p_model->set_gradient(g, std::make_shared<OnlineMIMOSVR>(gradient_params, p_features, p_labels, nullptr, p_model->get_multiout(), p_model->get_chunk_size()));
        if (save_parameters) {
            for (const auto &p: *gradient_params) {
                APP.svr_parameters_service.remove(p);
                APP.svr_parameters_service.save(p);
            }
        }
    }

    LOG4_END();
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

// TODO Rewrite
arma::colvec
ModelService::predict(
        const datamodel::Model_ptr &p_model,
        const data_row_container &main_decon_data,
        const std::deque<data_row_container_ptr> &aux_data_rows_containers,
        const boost::posix_time::ptime &prediction_time,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap)
{
    if (p_model->get_gradients().empty() or p_model->get_params().get_skip())
    {
        LOG4_DEBUG("Skipping prediction on model " << p_model->get_decon_level());
        return arma::colvec(PROPS.get_multistep_len()).fill(0.);
    }

    LOG4_DEBUG("Predicting using " << p_model->to_string());

    if (main_decon_data.empty())
        THROW_EX_F(svr::common::insufficient_data, "Decon queue is empty for model " << p_model->get_decon_level());

    const auto lag_count = p_model->get_params().get_lag_count();
    const auto lookback_range = prepare_feat_range(main_decon_data, max_gap, prediction_time, lag_count);

    LOG4_DEBUG(
            "Predicting time " << prediction_time <<
                               ", model for level " << p_model->get_decon_level() <<
                               ", learning levels " << svr::common::to_string(p_model->get_learning_levels()) <<
                               ", samples trained " << p_model->get_gradient(0)->get_samples_trained_number() <<
                               ", lookback range begin " << lookback_range.begin()->get()->get_value_time() <<
                               ", lookback range end " << lookback_range.rbegin()->get()->get_value_time() <<
                               ", lookback range size " << lookback_range.get_container().size() <<
                               ", lookback range distance " << lookback_range.distance() <<
                               ", lookback range duration " << lookback_range.rbegin()->get()->get_value_time() - lookback_range.begin()->get()->get_value_time());

    std::deque<datamodel::datarow_range> aux_lookback_ranges;
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
//    get_features_row(
//            pred_time, lookback_range, p_model->get_learning_levels(), p_model->get_decon_level(), max_gap, prediction_vector);
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
        const auto &p_param_set = p_model->get_gradients()->get_param_set();
        ss <<
           "predict_dataset_" << p_param_set.get_dataset_id() << "_" <<
           p_param_set.get_input_queue_table_name() << "_" << p_param_set.get_input_queue_column_name() <<
           "_level_" << p_model->get_decon_level() <<
           "_adjacent_levels_" << p_model->get_learning_levels().size() <<
           "_lag_" << p_param_set.get_lag_count() <<
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
    //PROFILE_EXEC_TIME(ret_values = p_model->get_gradients()->predict(prediction_vector), "Predict MIMO values");
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
        const datamodel::Model_ptr &p_model,
        const datamodel::datarow_range &main_data,
        const std::deque<datamodel::datarow_range> &aux_data_ranges,
        const ptimes_set_t &prediction_times,
        const bpt::time_duration &max_gap,
        const bpt::time_duration &resolution)
{
    LOG4_BEGIN();

    if (main_data.distance() < 1) LOG4_THROW("Decon queue is empty.");
    if (p_model->get_gradients().empty()) LOG4_THROW("Model is not initialized!");

    const auto lag_count = p_model->get_params().get_lag_count();
    { // Do initial sanity checks of input data
        const auto feat_time = *prediction_times.begin();
        (void) find_nearest_before(main_data.get_container(), feat_time, max_gap, lag_count);
        for (auto &aux_data_range: aux_data_ranges)
            (void) find_nearest_before(aux_data_range.get_container(), feat_time, max_gap, lag_count);
    }

    data_row_container result;
    const auto multistep_len = PROPS.get_multistep_len();
    LOG4_DEBUG("Predicting range " << *prediction_times.begin() << " to " << *prediction_times.rbegin());
    const auto multistep_prediction_times = common::to_multistep_times(prediction_times, resolution, multistep_len);
    arma::mat prediction_matrix(multistep_prediction_times.size(), 0);
    for (size_t time_ix = 0; time_ix < multistep_prediction_times.size(); ++time_ix) { // TODO Parallelize
        const auto &prediction_time = multistep_prediction_times^time_ix;
        LOG4_TRACE("Preparing prediction features for " << prediction_time);
        try {
            //const auto pred_range = prepare_feat_range(
            //        main_data.get_container(), max_gap, prediction_time, lag_count);
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
        result.push_back(std::make_shared<datamodel::DataRow>(prediction_time));

    LOG4_DEBUG(
            "Predicting range " << *prediction_times.begin() << " to " << *prediction_times.rbegin() <<
                                ", model: " << p_model->to_string() <<
                                ", learning levels " << svr::common::to_string(p_model->get_learning_levels()) <<
                                ", samples trained " << p_model->get_gradient(0)->get_samples_trained_number() <<
                                ", data range begin " << main_data.begin()->get()->get_value_time() <<
                                ", data range end " << main_data.rbegin()->get()->get_value_time() <<
                                ", data range size " << main_data.get_container().size() <<
                                ", data range distance " << main_data.distance() <<
                                ", data range duration " << main_data.rbegin()->get()->get_value_time() - main_data.begin()->get()->get_value_time() <<
                                ", aux lookback ranges count " << aux_data_ranges.size() <<
                                ", features matrix columns count " << prediction_matrix.n_cols <<
                                ", features matrix rows count " << prediction_matrix.n_cols);

    const auto result_values = p_model->get_gradient(0)->predict(prediction_matrix);

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


} // business
} // svr
