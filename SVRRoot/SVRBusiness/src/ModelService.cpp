#include "DeconQueueService.hpp"

#include <DAO/EnsembleDAO.hpp>
#include <execution>
#include <model/Ensemble.hpp>
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
#include <oneapi/tbb/concurrent_unordered_set.h>
#include <iostream>

#include "model/DataRow.hpp"
#include "util/string_utils.hpp"
#include "cuqrsolve.cuh"
#include "firefly.hpp"
#include "common/constants.hpp"
#include "model/SVRParameters.hpp"
#include "SVRParametersService.hpp"
#include "common/rtp_thread_pool.hpp"
#include "DAO/ModelDAO.hpp"
#include "model/Model.hpp"
#include "util/time_utils.hpp"
#include "util/validation_utils.hpp"
#include "util/math_utils.hpp"
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


const std::deque<unsigned> ModelService::C_quantisations = []() {
    constexpr unsigned C_divisor = 20;
    std::deque<unsigned> r{1};
    UNROLL(C_num_quantisations - 1)
    for (unsigned i = 0; i < ModelService::C_num_quantisations - 1; ++i) r.emplace_back(r.back() + std::max<unsigned>(1, r.back() / C_divisor)); // r.emplace_back(r.back() + 1);
    return r;
}();

const unsigned ModelService::C_max_quantisation = *std::max_element(C_default_exec_policy, C_quantisations.cbegin(), C_quantisations.cend());


ModelService::ModelService(dao::ModelDAO &model_dao) : model_dao(model_dao)
{}


unsigned int ModelService::to_level_ix(const unsigned int model_ix, const unsigned int level_ct) noexcept
{
#ifdef VMD_ONLY
    return model_ix >= MIN_LEVEL_COUNT ? model_ix / 2 : model_ix;
#elif defined(EMD_ONLY)
    return model_ix;
#else
    return model_ix >= MIN_LEVEL_COUNT ? ((model_ix >= (level_ct / 4)) ? (model_ix + 1) : model_ix) * 2 : 0;
#endif
}

unsigned int ModelService::to_level_ct(const unsigned model_ct) noexcept
{
#ifdef VMD_ONLY
    return model_ct * 2;
#elif defined(EMD_ONLY)
    return model_ct;
#else
    return model_ct >= MIN_LEVEL_COUNT / 2 - 1 ? model_ct * 2 + 2 : 1;
#endif
}

unsigned int ModelService::to_model_ct(const unsigned int level_ct) noexcept
{
#ifdef VMD_ONLY
    return level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 : 1;
#elif defined(EMD_ONLY)
    return level_ct;
#else
    return level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 - 1 : 1;
#endif
}

unsigned int ModelService::to_model_ix(const unsigned int level_ix, const unsigned int level_ct)
{
    if (level_ct < MIN_LEVEL_COUNT) return 0;
    const auto trans_levix = SVRParametersService::get_trans_levix(level_ct);
    if (level_ix == trans_levix) LOG4_THROW("Illegal level index " << level_ix << ", of count " << level_ct);
    return level_ix / 2 - (level_ix > trans_levix ? 1 : 0);
}

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
ModelService::validate(
        const unsigned int start_ix,
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
            ensemble, model, predict_features, dataset.get_input_queue()->get_resolution(), mx, batch_predicted),
                      "Batch predict of " << num_preds << " rows, level " << level << ", step " << model.get_step());
    if (batch_predicted.size() != num_preds)
        LOG4_THROW("Predicted size " << batch_predicted.size() << " not sane " << arma::size(*predict_features.p_features));

    LOG4_DEBUG("Batch predicted " << batch_predicted.size() << " values, parameters " << *params);
    const auto stepping = model.get_gradient()->get_dataset()->get_multistep();
    arma::vec predicted(num_preds), predicted_online(num_preds),
            actual = arma::mean(labels.rows(start_ix, ix_fini), 1),
            lastknown = arma::mean(last_knowns.rows(start_ix, ix_fini), 1);
#ifdef EMO_DIFF
    OMP_FOR(actual.n_cols)
    for (size_t i = 0; i < actual.n_cols; ++i) actual.col(i) = actual.col(i) + lastknown; // common::sexp<double>(actual.col(i)) + lastknown;
#endif
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    double batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (size_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted[ix] = stepping * batch_predicted[ix]->at(level);
        const double cur_absdiff_lk = std::abs(lastknown[ix] - actual[ix]);
        const double cur_absdiff_batch = std::abs(predicted[ix] - actual[ix]);
        const double cur_alpha_pct_batch = common::alpha(cur_absdiff_lk, cur_absdiff_batch);
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
                       << common::mape(sum_absdiff_batch, sum_abs_labels) << "pc, MAPE last-known " << common::mape(sum_absdiff_lk, sum_abs_labels) << "pc, batch alpha "
                       << common::alpha(sum_absdiff_lk, sum_absdiff_batch) << "pc, current batch alpha " << cur_alpha_pct_batch << "pc, batch correct predictions "
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
        if (row_report.str().size()) LOG4_DEBUG(row_report.str());
    }
    const auto mape_lk = 100. * sum_absdiff_lk / sum_abs_labels;
    if (online)
        return {sum_absdiff_online / double(num_preds), common::mape(sum_absdiff_online, sum_abs_labels), predicted_online, actual, mape_lk, lastknown};
    else
        return {sum_absdiff_batch / double(num_preds), common::mape(sum_absdiff_batch, sum_abs_labels), predicted, actual, mape_lk, lastknown};
}


std::deque<::std::shared_ptr<datamodel::DataRow>>::const_iterator
ModelService::get_start(
        const datamodel::DataRow::container &cont,
        const unsigned int decremental_offset,
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
ModelService::find(const std::deque<datamodel::Model_ptr> &models, const unsigned int levix, const unsigned int stepix)
{
    const auto res = std::find_if(C_default_exec_policy, models.cbegin(), models.cend(),
                                  [levix, stepix](const auto &p_model) { return p_model->get_decon_level() == levix && p_model->get_step() == stepix; });
    if (res != models.cend()) return *res;
    LOG4_WARN("Model for level " << levix << ", step " << stepix << " not found among " << models.size() << " models.");
    return nullptr;
}

void ModelService::configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model)
{
    if (!check(model.get_gradients(), model.get_gradient_count()) && model.get_id())
        model.set_gradients(model_dao.get_svr_by_model_id(model.get_id()), false);

    model.set_max_chunk_size(p_dataset->get_max_chunk_size());
    std::deque<datamodel::SVRParameters_ptr> paramset;
    if (p_dataset->get_id())
        paramset = APP.svr_parameters_service.get_by_dataset_column_level(p_dataset->get_id(), ensemble.get_column_name(), model.get_decon_level(), model.get_step());

    const unsigned default_model_num_chunks =
            paramset.empty() || std::none_of(C_default_exec_policy, paramset.cbegin(), paramset.cend(), [](const auto p) { return p->is_manifold(); }) ?
                1 : datamodel::OnlineMIMOSVR::get_num_chunks(paramset.empty() ? datamodel::C_default_svrparam_decrement_distance :
                                        (**paramset.cbegin()).get_svr_decremental_distance(), model.get_max_chunk_size());

    const unsigned default_adjacent_ct = paramset.empty() ? datamodel::C_default_svrparam_adjacent_levels_ratio * p_dataset->get_spectral_levels() :
                                         paramset.front()->get_adjacent_levels().size();
    datamodel::dq_scaling_factor_container_t all_model_scaling_factors;
    if (model.get_id()) all_model_scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model.get_id());
#pragma omp parallel num_threads(adj_threads(p_dataset->get_gradient_count() * default_model_num_chunks * p_dataset->get_spectral_levels() * default_adjacent_ct))
#pragma omp single
    {
        t_omp_lock gradients_l;
#pragma omp taskloop grainsize(1) default(shared) mergeable untied firstprivate(default_model_num_chunks)
        for (unsigned gix = 0; gix < p_dataset->get_gradient_count(); ++gix) {
            gradients_l.set();
            auto p_svr_model = model.get_gradient(gix);
            gradients_l.unset();

            bool set_params = false;
            unsigned grad_num_chunks = 0;
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
#pragma omp taskloop simd grainsize(1) default(shared) mergeable untied firstprivate(gix)
            for (unsigned chix = 0; chix < grad_num_chunks; ++chix)
                if (SVRParametersService::slice(grad_params, chix, gix).empty()) {
                    const auto level_grad_param_set = SVRParametersService::slice(paramset, chix, gix);
                    const auto p_params = level_grad_param_set.size() ?
                                          *level_grad_param_set.cbegin() :
                                          ptr<datamodel::SVRParameters>(
                                                  0, p_dataset->get_id(), p_dataset->get_input_queue()->get_table_name(), ensemble.get_column_name(),
                                                  p_dataset->get_spectral_levels(), model.get_decon_level(), model.get_step(), chix, gix);
                    grad_params_l.set();
                    grad_params.emplace(p_params);
                    set_params = true;
                    grad_params_l.unset();
                }

            if (!p_svr_model) {
                gradients_l.set();
                p_svr_model = model.get_gradients().emplace_back(ptr<datamodel::OnlineMIMOSVR>(0, model.get_id(), grad_params, p_dataset));
                gradients_l.unset();
            } else {
                if (set_params) p_svr_model->set_param_set(grad_params);
                if (!p_svr_model->get_dataset()) p_svr_model->set_dataset(p_dataset);
            }

            const auto adjacent_ct = (**grad_params.cbegin()).get_adjacent_levels().size();
            if (p_svr_model->get_scaling_factors().size() != p_svr_model->get_num_chunks()) {
#pragma omp taskloop grainsize(1) default(shared) mergeable untied firstprivate(grad_num_chunks)
                for (unsigned chix = 0; chix < grad_num_chunks; ++chix) {
                    datamodel::DQScalingFactor_ptr p_sf;
                    if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(), model.get_step(),
                                                      model.get_decon_level(), false, true)
                        &&
                        (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(), p_svr_model->get_step(),
                                                             model.get_decon_level(), false, true)))
                        p_svr_model->set_scaling_factor(p_sf);
#pragma omp taskloop simd grainsize(1) default(shared) mergeable untied firstprivate(adjacent_ct)
                    for (unsigned levix = 0; levix < adjacent_ct; ++levix) {
                        if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                          p_svr_model->get_step(), levix, true, false)
                            && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                                    p_svr_model->get_step(), levix, true, false)))
                            p_svr_model->set_scaling_factor(p_sf);
                    }
                }
            }
        }
    }
}


int ModelService::save(const datamodel::Model_ptr &p_model)
{
    common::reject_nullptr(p_model);
    if (!p_model->get_id()) p_model->set_id(model_dao.get_next_id());
    return model_dao.save(p_model);
}

bool ModelService::exists(const datamodel::Model &model)
{
    return model_dao.exists(model.get_id());
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

datamodel::Model_ptr ModelService::get_model(const bigint ensemble_id, const unsigned int decon_level)
{
    return model_dao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

bool ModelService::check(const std::deque<datamodel::Model_ptr> &models, const unsigned int model_ct)
{
    const auto level_ct = to_level_ct(model_ct);
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [level_ct](const auto &p_model) { return p_model->get_decon_level() < level_ct; });
}

bool ModelService::check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const size_t grad_ct)
{
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [grad_ct](const auto p_model) { return p_model->get_gradient_level() < grad_ct; });
}

arma::rowvec ModelService::prepare_special_features(const data_row_container::const_iterator &last_known_it, const bpt::time_duration &resolution, const unsigned len)
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

    arma::rowvec row(len, arma::fill::none);
    for (unsigned i = 0; i < spec_features.size(); ++i) row.subvec(i * step, i == spec_features.size() - 1 ? row.n_cols - 1 : (i + 1) * step - 1).fill(spec_features[i]);
    return row;
}

// Takes a decon queue and prepares feature vectors, using lag_count number of autoregressive features (and other misc features).
std::tuple<mat_ptr, mat_ptr, vec_ptr, times_ptr>
ModelService::get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, unsigned int dataset_rows)
{
    LOG4_BEGIN();

    const auto level = model.get_decon_level();
    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    auto p_params = model.get_head_params();
    if (!dataset_rows) dataset_rows = p_params->get_svr_decremental_distance() + C_test_len;
    const auto main_resolution = dataset.get_input_queue()->get_resolution();
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? main_resolution : dataset.get_aux_input_queue()->get_resolution();

    const datamodel::datarow_crange labels_range{ModelService::get_start( // Main labels are used for timing
            label_decon, dataset_rows, model.get_last_modeled_value_time(), main_resolution), label_decon.get_data().cend(), label_decon};
    const auto [p_labels, p_last_knowns, p_label_times] = dataset.get_calc_cache().get_cached_labels(
            model.get_step(), labels_range, labels_aux, dataset.get_max_lookback_time_gap(), level, aux_resolution,
            model.get_last_modeled_value_time(), main_resolution, dataset.get_multistep());
    auto p_features = ptr<arma::mat>();
    if (p_params->get_feature_mechanics().needs_tuning()) { PROFILE_EXEC_TIME(
                tune_features(*p_features, *p_labels, *p_params, *p_label_times, ensemble.get_aux_decon_queues(), dataset.get_max_lookback_time_gap(), aux_resolution,
                              main_resolution), "Tune features " << *p_params);
    } else PROFILE_EXEC_TIME(
            prepare_features(*p_features, *p_label_times, ensemble.get_aux_decon_queues(), *p_params, dataset.get_max_lookback_time_gap(), aux_resolution,
                             main_resolution), "Prepare features " << *p_params);

    if (p_labels->n_rows != p_features->n_rows) LOG4_THROW("Labels size " << arma::size(*p_labels) << ", features size " << arma::size(*p_features) << " do not match.");

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
    const auto half_main_res = .5 * main_queue_resolution;
    const auto horizon_duration = main_queue_resolution * PROPS.get_prediction_horizon();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    const unsigned label_len = main_queue_resolution / aux_queue_res;
    const unsigned horizon_len = horizon_duration / aux_queue_res;
    //LOG4_TRACE("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.front()->get_value_time() <<
    //                              " until " << main_data.back()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
    auto harvest_rows = [&](const datamodel::datarow_crange &harvest_range) {
        tbb::concurrent_unordered_set<arma::uword> shedded_rows;
        const auto expected_rows = harvest_range.distance();
        arma::mat labels(expected_rows, multistep);
        arma::vec last_knowns(expected_rows);
        tbb::concurrent_set<bpt::ptime> times;
        //LOG4_TRACE("Processing range " << (**harvest_range.begin()).get_value_time() << " to " << (**harvest_range.crbegin()).get_value_time() <<
        //                               ", expected " << expected_rows << " rows.");

#define SHED_ROW(msg) { LOG4_DEBUG(msg); shedded_rows.emplace(rowix); continue; }

        OMP_FOR_(expected_rows, simd firstprivate(main_to_aux_period_ratio, expected_rows, level))
        for (dtype(expected_rows) rowix = 0; rowix < expected_rows; ++rowix) {
            const auto label_start_time = harvest_range[rowix]->get_value_time();
            if (label_start_time <= last_modeled_value_time) SHED_ROW("Skipping already modeled row with value time " << label_start_time);
            const auto label_half_time = label_start_time + half_main_res;
            LOG4_TRACE("Adding row to training matrix with value time " << label_start_time);
            auto label_aux_start_iter = lower_bound(labels_aux.get_container(), labels_aux(rowix * main_to_aux_period_ratio), label_start_time);
            if (label_aux_start_iter == labels_aux.contend()) SHED_ROW("Can't find aux labels start " << label_start_time)
            else if ((**label_aux_start_iter).get_value_time() >= label_half_time) SHED_ROW(
                    "label aux start iter value time > label start time " << (**label_aux_start_iter).get_value_time() << " > " << label_half_time);
            // while (label_aux_start_iter != labels_aux.contend() && (**label_aux_start_iter).get_value_time() < label_half_time) ++label_aux_start_iter; // TODO experiment
            const auto label_end_time = label_start_time + main_queue_resolution;
            const auto label_aux_end_iter = lower_bound(
                    label_aux_start_iter, std::min(label_aux_start_iter + label_len + 1, labels_aux.cend()), label_end_time);
            const auto anchor_aux_iter = lower_bound_before(label_aux_start_iter - horizon_len - 1, label_aux_start_iter, label_start_time - horizon_duration);
            if (anchor_aux_iter == labels_aux.contend() || anchor_aux_iter == labels_aux.contbegin()) SHED_ROW("Can't find aux labels start " << label_start_time)
            auto labels_row = labels.row(rowix);
            generate_twap(--label_aux_start_iter, label_aux_end_iter, label_start_time, label_end_time, aux_queue_res, level, labels_row);

            if (labels_row.has_nonfinite()) SHED_ROW(
                    "Sanity check of row at " << label_start_time << " failed, size " << arma::size(labels_row) << ", content " << labels_row)

            const auto p_anchor_row = *anchor_aux_iter;
            last_knowns[rowix] = (*p_anchor_row)[level];
            times.emplace(label_start_time);
            if (ssize_t(rowix) >= harvest_range.distance() - 1)
                LOG4_TRACE(
                        "Added last data row " << rowix << ", value time " << label_start_time << ", label aux prev start time " << (**label_aux_start_iter).get_value_time()
                                               << ", last known time " << p_anchor_row->get_value_time() << ", last last-known value " << last_knowns[rowix] << ", label "
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

    if (all_labels.n_cols != multistep || all_labels.n_rows != 0) all_labels.set_size(0, multistep);

UNROLL()
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
                (**(main_data.begin() + main_data.distance() - 1)).get_value_time() + main_queue_resolution + main_queue_resolution * (1. - PROPS.get_prediction_horizon()));
        if (label_aux_start_iter == labels_aux.contend()) --label_aux_start_iter;
        const bpt::ptime label_start_time = (**label_aux_start_iter).get_value_time();
        arma::rowvec labels_row(multiout);
        labels_row.fill((**label_aux_start_iter)[level]);
        const auto &anchor_row = **std::prev(lower_bound_back(labels_aux.get_container(), label_aux_start_iter, label_start_time - main_queue_resolution * PROPS.get_prediction_horizon()));
        all_labels = arma::join_cols(all_labels, labels_row);
        all_last_knowns = arma::join_cols(all_last_knowns, arma::rowvec(anchor_row[level]));
        all_times.emplace_back(label_start_time);
        LOG4_DEBUG("Temporary data last row, time " << label_start_time << " anchor time " << anchor_row.get_value_time());
    }
#endif

#ifdef EMO_DIFF
    OMP_FOR_i(all_labels.n_cols)
        all_labels.col(i) = all_labels.col(i) - all_last_knowns; // common::slog<double>(all_labels.col(i) - all_last_knowns);
#endif

    if (all_labels.empty() or all_last_knowns.empty())
        LOG4_WARN("No new data to prepare for training, labels " << arma::size(all_labels) << ", last-knowns " << arma::size(all_last_knowns));
    else
        LOG4_TRACE("Prepared level " << level << ", labels " << common::present(all_labels) << ", last-knowns " << common::present(all_last_knowns));
}

void ModelService::tune_features(arma::mat &out_features, const arma::mat &labels, datamodel::SVRParameters &params, const std::deque<bpt::ptime> &label_times,
                                 const std::deque<datamodel::DeconQueue_ptr> &feat_queues, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res,
                                 const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    assert(labels.n_rows == label_times.size());
    const unsigned n_rows = labels.n_rows;
    const unsigned lag = params.get_lag_count();
    const auto adjacent_levels = params.get_adjacent_levels();
    const unsigned coef_lag = datamodel::C_features_superset_coef * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
    const auto coef_lag_ = coef_lag;
#endif
    const unsigned levels = adjacent_levels.size();
    const auto n_queues = feat_queues.size();
    const unsigned levels_queues = levels * n_queues;
    arma::vec best_score(levels, arma::fill::value(std::numeric_limits<double>::infinity()));
    datamodel::t_feature_mechanics fm{{levels_queues,                arma::fill::none},
                                      {levels_queues * lag,          arma::fill::none},
                                      std::deque<arma::uvec>(levels_queues),
                                      {levels_queues * lag,          arma::fill::none},
                                      {levels_queues * lag,          arma::fill::none}};

    const auto horizon_period = main_queue_resolution * PROPS.get_prediction_horizon();
    // const unsigned horizon_len = horizon_period / aux_queue_res;
    const size_t align_features_size = n_rows * coef_lag * sizeof(double) + n_rows * sizeof(double) + coef_lag * sizeof(float) + coef_lag * sizeof(double) +
            coef_lag * sizeof(unsigned) + n_rows * sizeof(unsigned);
    const unsigned n_chunks_align = cdiv(align_features_size, common::gpu_handler_hid::get().get_max_gpu_data_chunk_size());
    const unsigned chunk_len_align = cdiv(coef_lag, n_chunks_align);
    const auto stripe_period = aux_queue_res * coef_lag_;
    arma::vec mean_L = arma::mean(labels, 1);

    std::deque<unsigned> chunk_len_quantise(n_queues), n_feat_rows(n_queues);
    std::deque<arma::mat> decon(n_queues);
    std::deque<arma::u32_vec> times_F(n_queues);
    std::deque<std::vector<t_feat_params>> feat_params(n_queues);

#pragma omp parallel num_threads(C_n_cpu)
#pragma omp single
    {
#pragma omp taskloop grainsize(1) firstprivate(n_rows, levels) default(shared) mergeable // untied
        for (unsigned qix = 0; qix < n_queues; ++qix) {
            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues have different amount of samples, fix assumption they are same!
            const auto earliest_label_time = *std::min_element(label_times.cbegin(), label_times.cend());
            const auto earliest_time = earliest_label_time - horizon_period - stripe_period * C_max_quantisation;
            const auto latest_label_time = *std::max_element(label_times.cbegin(), label_times.cend());
            const auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_time - horizon_period);
            const auto start_iter = lower_bound_or_before_back(*p_queue, last_iter, earliest_time);
            if ((**start_iter).get_value_time() > earliest_time) LOG4_THROW("Start offset " << (**start_iter).get_value_time() << " is after " << earliest_time);
            const unsigned start_offset = start_iter - p_queue->cbegin();
            n_feat_rows[qix] = last_iter - start_iter;
            const size_t quantise_features_size = n_rows * coef_lag_ * sizeof(double) + n_feat_rows[qix] * sizeof(double) + n_rows * sizeof(unsigned) + n_rows *
                    sizeof(unsigned) + n_feat_rows[qix] * sizeof(unsigned);
            const unsigned n_chunks_quantise = cdiv(quantise_features_size, common::gpu_handler_hid::get().get_max_gpu_data_chunk_size());
            chunk_len_quantise[qix] = cdiv(n_rows, n_chunks_quantise);
            decon[qix] = arma::mat(n_feat_rows[qix], levels, arma::fill::none);
            feat_params[qix].resize(n_rows);
            times_F[qix].set_size(n_feat_rows[qix]);

#pragma omp taskloop simd NGRAIN(n_feat_rows[qix]) firstprivate(levels, start_offset) default(shared) mergeable untied
            for (unsigned r = 0; r < n_feat_rows[qix]; ++r) {
                const auto p_row = p_queue->at(start_offset + r);
                times_F[qix][r] = boost::posix_time::to_time_t(p_row->get_value_time());
                for (unsigned l = 0; l < levels; ++l) decon[qix](r, l) = p_row->at(adjacent_levels ^ l);
            }

#pragma omp taskloop simd NGRAIN(n_rows) firstprivate(n_rows) default(shared) mergeable untied
            for (unsigned r = 0; r < n_rows; ++r) {
                feat_params[qix][r].end_time = boost::posix_time::to_time_t(label_times[r] - horizon_period);
                feat_params[qix][r].ix_end = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), feat_params[qix][r].end_time) - times_F[qix].cbegin();
            }

#pragma omp taskloop grainsize(1) firstprivate(levels, qix) default(shared) mergeable // untied
            for (unsigned adj_ix = 0; adj_ix < levels; ++adj_ix) {
                t_omp_lock ins_l;
                const auto adj_ix_q = adj_ix + qix * levels;
                const auto adj_ix_q_1 = adj_ix_q + 1;

#pragma omp taskloop grainsize(1) firstprivate(n_rows, lag, coef_lag, adj_ix_q, adj_ix_q_1, qix) default(shared) mergeable // untied
                for (const auto quantise: C_quantisations) {
                    auto feat_params_qix_qt = feat_params[qix];
#pragma omp taskloop simd NGRAIN(n_rows) firstprivate(quantise) default(shared) mergeable untied
                    for (auto &f: feat_params_qix_qt) {
                        f.time_start = f.end_time - (stripe_period * quantise).total_seconds();
                        f.ix_start = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), f.time_start) - times_F[qix].cbegin();
                    }

                    arma::mat features(n_rows, coef_lag, arma::fill::none);
#pragma omp taskloop grainsize(1) firstprivate(n_rows, adj_ix, quantise, coef_lag_, coef_lag) default(shared) mergeable
                    for (size_t i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_EXEC_TIME(quantise_features(
                            decon[qix].mem, times_F[qix].mem, feat_params_qix_qt.data(), i,
                            std::min<unsigned>(i + chunk_len_quantise[qix], n_rows) - i, n_rows, n_feat_rows[qix], adj_ix,
                            coef_lag_, coef_lag, quantise, features.memptr()), "Quantise features " << chunk_len_quantise[qix] << ", quantise " << quantise);
                    arma::vec scores(coef_lag, arma::fill::none);
                    arma::fvec stretches(coef_lag, arma::fill::none), skips(coef_lag, arma::fill::none);
                    arma::u32_vec shifts(coef_lag, arma::fill::none);

#pragma omp taskloop grainsize(1) firstprivate(coef_lag, chunk_len_align, n_rows, quantise) default(shared) mergeable
                    for (unsigned i = 0; i < coef_lag; i += chunk_len_align) PROFILE_EXEC_TIME(align_features(
                            features.colptr(i), mean_L.mem, scores.memptr() + i, stretches.memptr() + i, shifts.memptr() + i, skips.memptr() + i,
                            n_rows, std::min<unsigned>(i + chunk_len_align, coef_lag) - i), "Align features " << n_rows << "x" << chunk_len_align << ", quantize " << quantise);
                    const arma::uvec trims = arma::uvec(arma::stable_sort_index(scores)).tail(coef_lag - lag);
                    scores.shed_rows(trims);
                    const double score = arma::accu(scores);
                    ins_l.set();
                    if (score < best_score[adj_ix_q]) {
                        LOG4_DEBUG("New best score " << score << ", previous best score " << best_score[adj_ix_q] << ", improvement " <<
                            common::imprv(score, best_score[adj_ix_q]) << "pc, quantise " << quantise << ", aux queue " << qix << ", level " << adj_ix << ", lag " << lag <<
                            ", coef lag " << coef_lag);
                        best_score[adj_ix_q] = score;
                        fm.quantization[adj_ix_q] = quantise;
                        stretches.shed_rows(trims);
                        shifts.shed_rows(trims);
                        skips.shed_rows(trims);
                        fm.stretches.rows(adj_ix_q * lag, adj_ix_q_1 * lag - 1) = stretches;
                        fm.shifts.rows(adj_ix_q * lag, adj_ix_q_1 * lag - 1) = shifts;
                        fm.skips.rows(adj_ix_q * lag, adj_ix_q_1 * lag - 1) = skips;
                        fm.trims[adj_ix_q] = trims;
                    }
                    ins_l.unset();
                }
            }
        }
    }
    params.set_feature_mechanics(fm);

    const unsigned levels_lag = levels * lag;
    const unsigned feature_cols = levels_lag * n_queues;
    const auto &quant = params.get_feature_mechanics().quantization;
    const auto &trims = params.get_feature_mechanics().trims;
    if (out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);

#pragma omp parallel num_threads(C_n_cpu)
#pragma omp single
    {
#pragma omp taskloop grainsize(1) default(shared) mergeable firstprivate(levels_lag)
        for (unsigned qix = 0; qix < n_queues; ++qix) {
            const auto queue_start = qix * levels_lag;
#pragma omp taskloop grainsize(1) default(shared) mergeable firstprivate(lag, coef_lag, qix, levels)
            for (unsigned adj_ix = 0; adj_ix < levels; ++adj_ix) {
                const unsigned adj_ix_q = queue_start + adj_ix;
                auto feat_params_qix_qt = feat_params[qix];
#pragma omp taskloop simd NGRAIN(n_rows) default(shared) mergeable untied
                for (auto &f: feat_params_qix_qt) {
                    f.time_start = f.end_time - (stripe_period * quant[adj_ix_q]).total_seconds();
                    f.ix_start = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), f.time_start) - times_F[qix].cbegin();
                }

                arma::mat level_features(n_rows, coef_lag, arma::fill::none);
#pragma omp taskloop simd grainsize(1) default(shared) mergeable firstprivate(n_rows, adj_ix, coef_lag_, coef_lag)
                for (unsigned i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_EXEC_TIME(quantise_features(
                        decon[qix].mem, times_F[qix].mem, feat_params_qix_qt.data(), i, std::min<unsigned>(i + chunk_len_quantise[qix], n_rows) - i,
                        n_rows, n_feat_rows[qix], adj_ix, coef_lag_, coef_lag, quant[adj_ix_q], level_features.memptr()),
                             "Prepare quantised features " << chunk_len_quantise[qix]);
                level_features.shed_cols(trims[adj_ix_q]);
                out_features.cols(queue_start + adj_ix * lag, queue_start + (adj_ix + 1) * lag - 1) = level_features;
            }
        }
    }

    LOG4_END();
}

void // TODO Test
ModelService::prepare_features(arma::mat &out_features, const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &feat_queues,
                               const datamodel::SVRParameters &params, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res,
                               const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();

    const unsigned n_rows = label_times.size();
    const unsigned lag = params.get_lag_count();
    const auto adjacent_levels = params.get_adjacent_levels();
    const unsigned coef_lag = datamodel::C_features_superset_coef * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
    const auto coef_lag_ = coef_lag;
#endif
    const unsigned levels = adjacent_levels.size();
    const auto n_queues = feat_queues.size();
    arma::vec best_score(levels, arma::fill::value(std::numeric_limits<double>::infinity()));

    const auto horizon_period = main_queue_resolution * PROPS.get_prediction_horizon();
    const auto stripe_period = aux_queue_res * coef_lag_;

    std::deque<unsigned> chunk_len_quantise(n_queues), n_feat_rows(n_queues);
    std::deque<arma::mat> decon(n_queues);
    std::deque<arma::u32_vec> times_F(n_queues);
    std::deque<std::vector<t_feat_params>> feat_params(n_queues);

    const unsigned levels_lag = levels * lag;
    const unsigned feature_cols = levels_lag * n_queues;
    const auto &quant = params.get_feature_mechanics().quantization;
    const auto &trims = params.get_feature_mechanics().trims;
    if (out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);

#pragma omp parallel num_threads(C_n_cpu)
#pragma omp single
    {
#pragma omp taskloop grainsize(1) default(shared) mergeable firstprivate(levels_lag)
        for (unsigned qix = 0; qix < n_queues; ++qix) {
            const auto queue_start = qix * levels_lag;

            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues have different amount of samples, fix assumption they are same!
            const auto earliest_label_time = *std::min_element(label_times.cbegin(), label_times.cend());
            const auto earliest_time = earliest_label_time - horizon_period - stripe_period * C_max_quantisation;
            const auto latest_label_time = *std::max_element(label_times.cbegin(), label_times.cend());
            const auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_time - horizon_period);
            const auto start_iter = lower_bound_or_before_back(*p_queue, last_iter, earliest_time);
            if ((**start_iter).get_value_time() > earliest_time) LOG4_THROW("Start offset " << (**start_iter).get_value_time() << " is after " << earliest_time);
            const unsigned start_offset = start_iter - p_queue->cbegin();
            n_feat_rows[qix] = last_iter - start_iter;
            const size_t quantise_features_size = n_rows * coef_lag_ * sizeof(double) + n_feat_rows[qix] * sizeof(double) + n_rows * sizeof(unsigned) + n_rows *
                    sizeof(unsigned) + n_feat_rows[qix] * sizeof(unsigned);
            const unsigned n_chunks_quantise = cdiv(quantise_features_size, common::gpu_handler_hid::get().get_max_gpu_data_chunk_size());
            chunk_len_quantise[qix] = cdiv(n_rows, n_chunks_quantise);
            decon[qix] = arma::mat(n_feat_rows[qix], levels, arma::fill::none);
            feat_params[qix].resize(n_rows);
            times_F[qix].set_size(n_feat_rows[qix]);

#pragma omp taskloop simd NGRAIN(n_feat_rows[qix]) firstprivate(levels, start_offset) default(shared) mergeable untied
            for (unsigned r = 0; r < n_feat_rows[qix]; ++r) {
                const auto p_row = p_queue->at(start_offset + r);
                times_F[qix][r] = boost::posix_time::to_time_t(p_row->get_value_time());
                for (unsigned l = 0; l < levels; ++l) decon[qix](r, l) = p_row->at(adjacent_levels ^ l);
            }

#pragma omp taskloop simd NGRAIN(n_rows) firstprivate(n_rows) default(shared) mergeable untied
            for (unsigned r = 0; r < n_rows; ++r) {
                feat_params[qix][r].end_time = boost::posix_time::to_time_t(label_times[r] - horizon_period);
                feat_params[qix][r].ix_end = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), feat_params[qix][r].end_time) - times_F[qix].cbegin();
            }

#pragma omp taskloop grainsize(1) default(shared) mergeable firstprivate(lag, coef_lag, qix, levels)
            for (unsigned adj_ix = 0; adj_ix < levels; ++adj_ix) {
                const unsigned adj_ix_q = queue_start + adj_ix;
                auto feat_params_qix_qt = feat_params[qix];
#pragma omp taskloop simd NGRAIN(n_rows) default(shared) mergeable untied
                for (auto &f: feat_params_qix_qt) {
                    f.time_start = f.end_time - (stripe_period * quant[adj_ix_q]).total_seconds();
                    f.ix_start = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), f.time_start) - times_F[qix].cbegin();
                }

                arma::mat level_features(n_rows, coef_lag, arma::fill::none);
#pragma omp taskloop simd grainsize(1) default(shared) mergeable firstprivate(n_rows, adj_ix, coef_lag_, coef_lag)
                for (unsigned i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_EXEC_TIME(quantise_features(
                        decon[qix].mem, times_F[qix].mem, feat_params_qix_qt.data(), i, std::min<unsigned>(i + chunk_len_quantise[qix], n_rows) - i,
                        n_rows, n_feat_rows[qix], adj_ix, coef_lag_, coef_lag, quant[adj_ix_q], level_features.memptr()),
                             "Prepare quantised features " << chunk_len_quantise[qix]);
                level_features.shed_cols(trims[adj_ix_q]);
                out_features.cols(queue_start + adj_ix * lag, queue_start + (adj_ix + 1) * lag - 1) = level_features;
            }
        }
    }

    if (out_features.empty()) LOG4_WARN("No new data to prepare for training, features " << arma::size(out_features));
}


void
ModelService::train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model)
{
    const auto [p_features, p_labels, p_last_knowns, p_times] = get_training_data(dataset, ensemble, model);
    const auto last_value_time = p_times->back();
    if (last_value_time < model.get_last_modeled_value_time()) {
        LOG4_ERROR("Data is older " << last_value_time << " than last modeled time " << model.get_last_modeled_value_time());
        return;
    }
    if (model.get_last_modeled_value_time() == bpt::min_date_time)
        train_batch(model, p_features, p_labels, p_last_knowns, last_value_time);
    else
        train_online(model, *p_features, *p_labels, *p_last_knowns, last_value_time);
    model.set_last_modeled_value_time(last_value_time);
    model.set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO("Finished training model " << model);
}


void
ModelService::train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::vec &last_knowns, const bpt::ptime &last_value_time)
{
    arma::mat residuals, learn_labels = labels;
UNROLL()
    for (unsigned g = 0; g < model.get_gradient_count(); ++g) {
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
UNROLL()
    for (unsigned gix = 0; gix < model.get_gradient_count(); ++gix) {
        const auto p_gradient = model.get_gradient(gix);
        if (!p_gradient) LOG4_THROW("SVR model for gradient " << gix << " not initialized " << model);
        p_gradient->batch_train(gradient_data.p_features, gradient_data.p_labels, gradient_data.p_last_knowns, last_value_time);
        if (model.get_gradient_count() < 2 || gix == model.get_gradient_count() - 1) continue;
        gradient_data = model.get_gradient(gix)->produce_residuals();
UNROLL()
        for (auto &p: model.get_gradient(gix + 1)->get_param_set())
            p->set_svr_decremental_distance(gradient_data.p_features->n_rows - C_test_len);
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble &ensemble, const unsigned int level, const std::deque<bpt::ptime> &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size());
    const auto p_aux_decon = ensemble.get_label_aux_decon();
    const auto lastknown_offset = PROPS.get_prediction_horizon() * resolution;
    OMP_FOR_i_(res.size(), simd firstprivate(level))
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
    OMP_FOR(model.get_gradient_count())
    for (const auto &p_svr: model.get_gradients()) {
        const auto this_prediction = p_svr->predict(*predict_features.p_features, predict_features.times.back());
        predict_lock.set();
        prediction += this_prediction;
        predict_lock.unset();
    }
#ifdef EMO_DIFF
    const auto lk = get_last_knowns(ensemble, model.get_decon_level(), predict_features.times, resolution);
    OMP_FOR_i(prediction.n_cols) prediction.col(i) = prediction.col(i) + lk; // common::sexp<double>(prediction.col(i)) + lk;
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
ModelService::init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble &ensemble)
{
    if (!check(ensemble.get_models(), p_dataset->get_model_count()) && ensemble.get_id())
        ensemble.set_models(model_dao.get_all_ensemble_models(ensemble.get_id()), false);
    t_omp_lock init_models_l;
    OMP_FOR_(p_dataset->get_model_count() * p_dataset->get_multistep(), simd collapse(2))
    for (unsigned levix = 0; levix < p_dataset->get_spectral_levels(); levix += LEVEL_STEP)
        for (unsigned stepix = 0; stepix < p_dataset->get_multistep(); ++stepix)
            if (levix != p_dataset->get_trans_levix()) {
                init_models_l.set();
                auto p_model = ensemble.get_model(levix, stepix);
                init_models_l.unset();
                if (!p_model) {
                    p_model = ptr<datamodel::Model>(0, ensemble.get_id(), levix, stepix, PROPS.get_multiout(), p_dataset->get_gradient_count(),
                                                    p_dataset->get_max_chunk_size());
                    init_models_l.set();
                    ensemble.get_models().emplace_back(p_model);
                    init_models_l.unset();
                }
                configure(p_dataset, ensemble, *p_model);
            }
}

} // business
} // svr
