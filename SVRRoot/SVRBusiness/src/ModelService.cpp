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

const std::deque<uint32_t> ModelService::C_quantisations = [] {
    constexpr uint32_t C_divisor = 30; // Meaning, quantisations up to C_divisor * 2 are incremented by 1
    std::deque<uint32_t> r{1};
    UNROLL(C_num_quantisations - 1)
    for (DTYPE(ModelService::C_num_quantisations) i = 0; i < ModelService::C_num_quantisations - 1; ++i)
        r.emplace_back(r.back() + std::max<uint32_t>(1, r.back() / C_divisor)); // r.emplace_back(r.back() + 1);
    return r;
}();

const uint32_t ModelService::C_max_quantisation = *std::max_element(C_default_exec_policy, C_quantisations.cbegin(), C_quantisations.cend());

uint32_t ModelService::get_max_row_len()
{
    return C_max_quantisation * (1 + datamodel::C_default_svrparam_lag_count * datamodel::C_features_superset_coef);
}

ModelService::ModelService(dao::ModelDAO &model_dao) : model_dao(model_dao)
{}


uint16_t ModelService::to_level_ix(const uint16_t model_ix, const uint16_t level_ct) noexcept
{
#ifdef VMD_ONLY
    return model_ix >= MIN_LEVEL_COUNT ? model_ix / 2 : model_ix;
#elif defined(EMD_ONLY)
    return model_ix;
#else
    return model_ix >= MIN_LEVEL_COUNT ? ((model_ix >= (level_ct / 4)) ? (model_ix + 1) : model_ix) * 2 : 0;
#endif
}

uint16_t ModelService::to_level_ct(const uint16_t model_ct) noexcept
{
#ifdef VMD_ONLY
    return model_ct * 2;
#elif defined(EMD_ONLY)
    return model_ct;
#else
    return model_ct >= MIN_LEVEL_COUNT / 2 - 1 ? model_ct * 2 + 2 : 1;
#endif
}

uint16_t ModelService::to_model_ct(const uint16_t level_ct) noexcept
{
#ifdef VMD_ONLY
    return level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 : 1;
#elif defined(EMD_ONLY)
    return level_ct;
#else
    return level_ct >= MIN_LEVEL_COUNT ? level_ct / 2 - 1 : 1;
#endif
}

uint16_t ModelService::to_model_ix(const uint16_t level_ix, const uint16_t level_ct)
{
    if (level_ct < MIN_LEVEL_COUNT) return 0;
    const auto trans_levix = SVRParametersService::get_trans_levix(level_ct);
    if (level_ix == trans_levix) LOG4_THROW("Illegal level index " << level_ix << ", of count " << level_ct);
    return level_ix / 2 - (level_ix > trans_levix ? 1 : 0);
}

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
ModelService::validate(const uint32_t start_ix, const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model, const arma::mat &features,
                       const arma::mat &labels, const arma::vec &last_knowns, const arma::mat &weights, const data_row_container &times, const bool online, const bool verbose)
{
    LOG4_BEGIN();
    if (labels.n_rows <= start_ix) {
        LOG4_WARN("Calling future validate at the end of labels array. MAE is 1000");
        return {common::C_bad_validation, common::C_bad_validation, {}, {}, 0., {}};
    }

    const uint32_t ix_fini = labels.n_rows - 1;
    const uint32_t num_preds = labels.n_rows - start_ix;
    const auto params = model.get_head_params();
    const auto level = params->get_decon_level();

    datamodel::t_level_predict_features predict_features{{times.cbegin() + start_ix, times.cend()}, otr<arma::mat>(features.rows(start_ix, ix_fini))};
    LOG4_TRACE("Predicting features " << common::present<double>(*predict_features.p_features));
    data_row_container batch_predicted, cont_predicted_online;
    tbb::mutex mx;
    PROFILE_EXEC_TIME(ModelService::predict(ensemble, model, predict_features, dataset.get_input_queue()->get_resolution(), mx, batch_predicted),
                      "Batch predict of " << num_preds << " rows, level " << level << ", step " << model.get_step());
    if (batch_predicted.size() != num_preds)
        LOG4_THROW("Predicted size " << batch_predicted.size() << " not sane " << arma::size(*predict_features.p_features));

    LOG4_DEBUG("Batch predicted " << batch_predicted.size() << " values, parameters " << *params);
    const auto stepping = model.get_gradient()->get_dataset()->get_multistep();
    arma::vec predicted_batch(num_preds), predicted_online(num_preds), actual = arma::mean(labels.rows(start_ix, ix_fini), 1), lastknown = last_knowns.rows(start_ix, ix_fini);
#ifdef EMO_DIFF
    OMP_FOR(actual.n_cols)
    for (DTYPE(actual.n_cols) i = 0; i < actual.n_cols; ++i) actual.col(i) = actual.col(i) + lastknown; // common::sexp<double>(actual.col(i)) + lastknown;
#endif
    double sum_absdiff_batch = 0, sum_absdiff_lk = 0, sum_abs_labels = 0, sum_absdiff_online = 0;
    double batch_correct_directions = 0, batch_correct_predictions = 0, online_correct_directions = 0, online_correct_predictions = 0;
    for (uint32_t ix_future = start_ix; ix_future <= ix_fini; ++ix_future) {
        const auto ix = ix_future - start_ix;
        predicted_batch[ix] = stepping * batch_predicted[ix]->at(level);
        const double cur_absdiff_lk = std::abs(lastknown[ix] - actual[ix]);
        const double cur_absdiff_batch = std::abs(predicted_batch[ix] - actual[ix]);
        const double cur_alpha_pct_batch = common::alpha(cur_absdiff_lk, cur_absdiff_batch);
        sum_abs_labels += std::abs(actual[ix]);
        sum_absdiff_batch += cur_absdiff_batch;
        sum_absdiff_lk += std::abs(actual[ix] - lastknown[ix]);
        batch_correct_predictions += cur_absdiff_batch < cur_absdiff_lk;
        batch_correct_directions += std::signbit(predicted_batch[ix] - lastknown[ix]) == std::signbit(actual[ix] - lastknown[ix]);

        const auto ix_div = ix + 1.;
        const bool print_line = verbose || ix_future == ix_fini || ix % 115 == 0;
        std::stringstream row_report;
        if (print_line)
            row_report << "Position " << ix << ", level " << level << ", step " << model.get_step() << ", actual " << actual[ix] << ", batch predicted " << predicted_batch[ix]
                       << ", last known " << lastknown[ix] << " batch MAE " << sum_absdiff_batch / ix_div << ", MAE last-known " << sum_absdiff_lk / ix_div
                       << ", batch MAPE "
                       << common::mape(sum_absdiff_batch, sum_abs_labels) << "pc, MAPE last-known " << common::mape(sum_absdiff_lk, sum_abs_labels) << "pc, batch alpha "
                       << common::alpha(sum_absdiff_lk, sum_absdiff_batch) << "pc, current batch alpha " << cur_alpha_pct_batch << "pc, batch correct predictions "
                       << 100. * batch_correct_predictions / ix_div << "pc, batch correct directions " << 100. * batch_correct_directions / ix_div << "pc";

        if (online) {
            PROFILE_EXEC_TIME(
                    ModelService::predict(
                            ensemble, model,
                            datamodel::t_level_predict_features{
                                    {times[ix]},
                                    ptr<arma::mat>(features.row(ix_future))},
                            dataset.get_input_queue()->get_resolution(), mx, cont_predicted_online),
                    "Online predict " << ix << " of 1 row, " << features.n_cols << " feature columns, " << labels.n_cols << " labels per row, level " << level <<
                                      ", step " << model.get_step() << " at " << times[ix_future]);
            PROFILE_EXEC_TIME(
                    ModelService::train_online(model, features.row(ix_future), labels.row(ix_future), last_knowns.row(ix_future), weights.row(ix_future),
                                               times[ix_future]->get_value_time()),
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
    const auto &sum_absdiff = online ? sum_absdiff_online : sum_absdiff_batch;
    const auto &predicted = online ? predicted_online : predicted_batch;
    LOG4_INFO("Parameters " << params << ", predictions start " << start_ix << ", last index " << ix_fini << ", concession " << common::present<double>(actual - predicted));
    return {sum_absdiff / double(num_preds), common::mape(sum_absdiff, sum_abs_labels), predicted, actual, mape_lk, lastknown};
}


std::deque<::std::shared_ptr<datamodel::DataRow>>::const_iterator
ModelService::get_start(
        const datamodel::DataRow::container &cont,
        const uint32_t decremental_offset,
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
ModelService::find(const std::deque<datamodel::Model_ptr> &models, const uint16_t levix, const uint16_t stepix)
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

    const uint16_t default_model_num_chunks =
            paramset.empty() || std::none_of(C_default_exec_policy, paramset.cbegin(), paramset.cend(), [](const auto p) { return p->is_manifold(); }) ?
            1 : datamodel::OnlineMIMOSVR::get_num_chunks(paramset.empty() ? datamodel::C_default_svrparam_decrement_distance :
                                                         (**paramset.cbegin()).get_svr_decremental_distance(), model.get_max_chunk_size());

    const uint16_t default_adjacent_ct = paramset.empty() ? datamodel::C_default_svrparam_adjacent_levels_ratio * p_dataset->get_spectral_levels() :
                                         paramset.front()->get_adjacent_levels().size();
    datamodel::dq_scaling_factor_container_t all_model_scaling_factors;
    if (model.get_id()) all_model_scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model.get_id());
#pragma omp parallel num_threads(adj_threads(p_dataset->get_gradient_count() * default_model_num_chunks * p_dataset->get_spectral_levels() * default_adjacent_ct))
#pragma omp single
    {
        t_omp_lock gradients_l;
#pragma omp taskloop grainsize(1) default(shared) mergeable untied firstprivate(default_model_num_chunks)
        for (uint16_t gix = 0; gix < p_dataset->get_gradient_count(); ++gix) {
            gradients_l.set();
            auto p_svr_model = model.get_gradient(gix);
            gradients_l.unset();

            bool set_params = false;
            uint16_t grad_num_chunks = 0;
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
            for (uint16_t chix = 0; chix < grad_num_chunks; ++chix)
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
                for (DTYPE(grad_num_chunks) chix = 0; chix < grad_num_chunks; ++chix) {
                    datamodel::DQScalingFactor_ptr p_sf;
                    if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(), model.get_step(),
                                                      model.get_decon_level(), false, true)
                        &&
                        (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(), p_svr_model->get_step(),
                                                             model.get_decon_level(), false, true))) {
                        gradients_l.set();
                        p_svr_model->set_scaling_factor(p_sf);
                        gradients_l.unset();
                    }
#pragma omp taskloop simd grainsize(1) default(shared) mergeable untied firstprivate(adjacent_ct)
                    for (DTYPE(adjacent_ct) levix = 0; levix < adjacent_ct; ++levix) {
                        if (!DQScalingFactorService::find(p_svr_model->get_scaling_factors(), model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                          p_svr_model->get_step(), levix, true, false)
                            && (p_sf = DQScalingFactorService::find(all_model_scaling_factors, model.get_id(), chix, p_svr_model->get_gradient_level(),
                                                                    p_svr_model->get_step(), levix, true, false))) {
                            gradients_l.set();
                            p_svr_model->set_scaling_factor(p_sf);
                            gradients_l.unset();
                        }
                    }
                }
            }
        }
    }
}


int ModelService::save(const datamodel::Model_ptr &p_model)
{
    REJECT_NULLPTR(p_model);
    if (!p_model->get_id()) p_model->set_id(model_dao.get_next_id());
    return model_dao.save(p_model);
}

bool ModelService::exists(const datamodel::Model &model)
{
    return model_dao.exists(model.get_id());
}

int ModelService::remove(const datamodel::Model_ptr &model)
{
    REJECT_NULLPTR(model);
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

datamodel::Model_ptr ModelService::get_model(const bigint ensemble_id, const uint16_t decon_level)
{
    return model_dao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

bool ModelService::check(const std::deque<datamodel::Model_ptr> &models, const uint16_t model_ct)
{
    const auto level_ct = to_level_ct(model_ct);
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [level_ct](const auto &p_model) { return p_model->get_decon_level() < level_ct; });
}

bool ModelService::check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const uint16_t grad_ct)
{
    return std::count_if(C_default_exec_policy, models.cbegin(), models.cend(),
                         [grad_ct](const auto p_model) { return p_model->get_gradient_level() < grad_ct; });
}

arma::rowvec ModelService::prepare_special_features(const data_row_container::const_iterator &last_known_it, const bpt::time_duration &resolution, const uint32_t len)
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
    for (uint32_t i = 0; i < spec_features.size(); ++i) row.subvec(i * step, i == spec_features.size() - 1 ? row.n_cols - 1 : (i + 1) * step - 1).fill(spec_features[i]);
    return row;
}

void ModelService::prepare_weights(
        arma::mat &weights, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const uint16_t steps,
        const bpt::time_duration &resolution_main)
{
    LOG4_BEGIN();

    const auto num_rows = times.size();
    if (num_rows < 1) LOG4_THROW("No times to prepare weights for.");
    if (weights.n_rows != num_rows || weights.n_cols != steps) weights.set_size(num_rows, steps);
    weights.ones();
    const auto s_duration = resolution_main / steps;
#pragma omp parallel ADJ_THREADS(num_rows * aux_inputs.size() * steps)
#pragma omp single
    {
        OMP_TASKLOOP_(num_rows,)
        for (DTYPE(num_rows) i = 0; i < num_rows; ++i) {
            const auto &t = times[i];
            for (const auto &q: aux_inputs)
                    OMP_TASKLOOP_1()
                for (uint16_t s = 0; s < steps; ++s) {
                    const auto s_start = t->get_value_time() + s * s_duration;
                    for (auto it = lower_bound(std::as_const(q->get_data()), s_start); it != q->cend() && (*it)->get_value_time() < s_start + s_duration; ++it)
                        weights(i, s) += (**it).get_tick_volume();
                }
        }
    }

    LOG4_END();
}

// Takes a decon queue and prepares feature vectors, using lag_count number of autoregressive features (and other misc features).
std::tuple<mat_ptr, mat_ptr, vec_ptr, mat_ptr, data_row_container_ptr>
ModelService::get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, uint32_t dataset_rows)
{
    LOG4_BEGIN();

    const auto level = model.get_decon_level();
    const auto &label_decon = *ensemble.get_decon_queue();
    const auto &labels_aux = *ensemble.get_label_aux_decon();
    auto p_params = model.get_head_params();
    if (!dataset_rows) dataset_rows = p_params->get_svr_decremental_distance();
    const auto main_resolution = dataset.get_input_queue()->get_resolution();
    const auto aux_resolution = dataset.get_aux_input_queues().empty() ? main_resolution : dataset.get_aux_input_queue()->get_resolution();
    const datamodel::datarow_crange labels_range{ModelService::get_start( // Main labels are used for timing
            label_decon, dataset_rows, model.get_last_modeled_value_time(), main_resolution), label_decon.get_data().cend(), label_decon};
    const auto [p_labels, p_last_knowns, p_label_times] = dataset.get_calc_cache().get_labels(
            p_params->get_input_queue_column_name(), model.get_step(), labels_range, labels_aux, dataset.get_max_lookback_time_gap(), level, aux_resolution,
            model.get_last_modeled_value_time(), main_resolution, dataset.get_multistep(), model.get_head_params()->get_lag_count());
    auto p_features = ptr<arma::mat>();
    if (p_params->get_feature_mechanics().needs_tuning()) {
        PROFILE_EXEC_TIME(
                tune_features(*p_features, *p_labels, *p_params, *p_label_times, ensemble.get_aux_decon_queues(), dataset.get_max_lookback_time_gap(), aux_resolution,
                              main_resolution),
                "Tune features " << *p_params);
    } else PROFILE_EXEC_TIME(
            prepare_features(*p_features, *p_label_times, ensemble.get_aux_decon_queues(), *p_params, dataset.get_max_lookback_time_gap(), aux_resolution, main_resolution),
            "Prepare features " << *p_params);
    assert(p_labels->n_rows == p_features->n_rows);
#ifdef INSTANCE_WEIGHTS
    const auto p_weights = dataset.get_calc_cache().get_weights(
            dataset.get_id(), *p_label_times, dataset.get_aux_input_queues(), model.get_step(), dataset.get_multistep(), main_resolution);
    assert(p_labels->n_rows == p_weights->n_rows);
#else
    const auto p_weights = nullptr;
#endif

    return {p_features, p_labels, p_last_knowns, p_weights, p_label_times};
}


void
ModelService::prepare_labels(arma::mat &all_labels, arma::vec &all_last_knowns, data_row_container &all_times, const datamodel::datarow_crange &main_data,
                             const datamodel::datarow_crange &aux_data, const bpt::time_duration &max_gap, const uint16_t level,
                             const bpt::time_duration &resolution_aux, const bpt::ptime &last_modeled_value_time, const bpt::time_duration &resolution_main,
                             const uint16_t multistep, const uint32_t lag)
{
    LOG4_BEGIN();
    const auto req_rows = main_data.distance();
    const uint32_t coef_lag = datamodel::C_features_superset_coef * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
    const auto coef_lag_ = coef_lag;
#endif
    // LOG4_TRACE("Preparing level " << level << ", training " << req_rows << " rows, main range from " << main_data.front()->get_value_time() <<
    //                              " until " << main_data.back()->get_value_time() << ", main to aux period ratio " << main_to_aux_period_ratio);
#define label_duration resolution_main
    const auto horizon_duration = resolution_main * PROPS.get_prediction_horizon();
    if (req_rows < 1 or main_data.get_container().empty()) LOG4_THROW("Main data level " << level << " is empty!");
    const uint32_t label_len = resolution_main / resolution_aux;

    std::vector<t_label_ix> label_ixs;
    label_ixs.reserve(req_rows);
    std::vector<uint32_t> ix_F_end;
    ix_F_end.reserve(req_rows);
    std::vector<double> last_knowns;
    last_knowns.reserve(req_rows);
    const uint32_t horizon_len_1 = label_len * PROPS.get_prediction_horizon() * 2;
    const auto label_len_1 = label_len + 1;
    const uint32_t avail_rows = main_data.cend() - main_data.contcbegin();
    const auto stripe_period = resolution_aux * coef_lag_;
    const auto first_time = aux_data.front()->get_value_time();
    const auto max_row_duration = horizon_duration + stripe_period * C_max_quantisation;
    OMP_FOR_(avail_rows, ordered)
    for (auto it_main_time = main_data.contcbegin(); it_main_time != main_data.cend(); ++it_main_time) {
        const auto L_start_time = (**it_main_time).get_value_time();
        if (L_start_time - max_row_duration < first_time || L_start_time < last_modeled_value_time) {
            LOG4_TRACE("Skipping time " << L_start_time << " as it is before last modeled time " << last_modeled_value_time << " or before max row duration " << max_row_duration);
            continue;
        }
        const auto L_start_it = lower_bound_before(aux_data.cbegin(), aux_data.cend(), L_start_time);
        if (L_start_it == aux_data.cend() || (**L_start_it).get_value_time() > L_start_time) {
            LOG4_TRACE("No aux data for time " << L_start_time);
            continue;
        }

        const auto L_end_time = L_start_time + label_duration;
        const auto L_end_it = lower_bound(L_start_it, aux_data.cend() - L_start_it > label_len_1 ? L_start_it + label_len_1 : aux_data.cend(), L_end_time);
        if (L_end_it - L_start_it < 1) {
            LOG4_TRACE("No aux data for time " << L_start_time << " label ending at " << L_end_time);
            continue;
        }
        const auto F_end_time = L_start_time - horizon_duration;
        auto F_end_it = lower_bound(L_start_it - aux_data.cbegin() > horizon_len_1 ? L_start_it - horizon_len_1 : aux_data.cbegin(), L_start_it, F_end_time);
        if (F_end_it == aux_data.cend() || F_end_it == aux_data.cbegin()) {
            LOG4_TRACE("No feature data for time " << F_end_time << " label at " << L_start_time);
            continue;
        }
        const uint32_t F_end_ix = F_end_it - aux_data.cbegin();
        if ((**F_end_it).get_value_time() >= F_end_time) --F_end_it; // Do not use F_end_it after this line

        t_label_ix *L_ins;
#pragma omp ordered
        {
            L_ins = &label_ixs.emplace_back(t_label_ix{.n_ixs = label_len});
            ix_F_end.emplace_back(F_end_ix);
            all_times.emplace_back(*it_main_time);
            last_knowns.emplace_back((**F_end_it)[level]);
        };
        generate_twap_indexes(aux_data.cbegin(), L_start_it, L_end_it, L_start_time, L_end_time, resolution_aux, label_len, L_ins->label_ixs);
        LOG4_TRACE("Added row " << label_ixs.size() << " for time " << L_start_time << " label at " << *L_ins->label_ixs << " feature at " << F_end_time << " with " <<
                                F_end_ix << " index, of length " << label_len);
    }

#ifdef LAST_KNOWN_LABEL // TODO Implement for online learn!
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

    const auto labels_size = label_ixs.size();
    assert(labels_size == ix_F_end.size());
    assert(labels_size == all_times.size());
    assert(labels_size == last_knowns.size());
    if (CAST2(req_rows) labels_size > req_rows) {
        const auto offshoot = labels_size - req_rows;
        label_ixs.erase(label_ixs.begin(), label_ixs.begin() + offshoot);
        ix_F_end.erase(ix_F_end.begin(), ix_F_end.begin() + offshoot);
        all_times.erase(all_times.begin(), all_times.begin() + offshoot);
        last_knowns.erase(last_knowns.begin(), last_knowns.begin() + offshoot);
    } else if (CAST2(req_rows) labels_size < req_rows)
        LOG4_THROW("Label indexes size " << labels_size << " less than required " << req_rows);
    assert(CAST2(req_rows) label_ixs.size() == req_rows);

    all_labels.set_size(req_rows, multistep);
    all_last_knowns.set_size(req_rows);
    memcpy(all_last_knowns.memptr(), last_knowns.data(), req_rows * sizeof(double));
    std::vector<double> labels_aux_in(aux_data.distance());
    OMP_FOR_i(aux_data.distance()) labels_aux_in[i] = aux_data[i]->at(level);
    quantise_labels(label_len, labels_aux_in, label_ixs, ix_F_end, all_labels.memptr(), multistep);

    assert(!all_labels.has_nonfinite() && !all_last_knowns.has_nonfinite());
    if (all_labels.empty() or all_last_knowns.empty())
        LOG4_WARN("No new data to prepare for training, labels " << arma::size(all_labels) << ", last-knowns " << arma::size(all_last_knowns));
    else
        LOG4_TRACE("Prepared level " << level << ", labels " << common::present(all_labels) << ", last-knowns " << common::present(all_last_knowns));
}

#if 0
void generate_labels_indexes(data_row_container &all_times, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &aux_data,
                                           const uint32_t level, const bpt::time_duration &resolution_aux, const bpt::ptime &last_modeled_value_time,
                                           const bpt::time_duration &resolution_main, const boost::posix_time::time_duration &horizon_duration, const uint32_t label_len,
                                           std::vector<t_label_ix> &label_ixs, std::vector<uint32_t> &ix_F_end, const std::vector<double> &last_knowns,
                                           const uint32_t horizon_len_1, const uint32_t label_len_1, const bpt::ptime &first_time,
                                           const boost::posix_time::time_duration &max_row_duration) noexcept
{
    for (auto it_main_time = main_data.contcbegin(); it_main_time != main_data.cend(); ++it_main_time) {
        const auto it_time = (**it_main_time).get_value_time();
        if (it_time < last_modeled_value_time || it_time - max_row_duration < first_time) {
            LOG4_TRACE("Skipping time " << it_time << " as it is before last modeled time " << last_modeled_value_time << " or before max row duration " << max_row_duration);
            continue;
        }
        const auto L_start_it = lower_bound_before(aux_data.cbegin(), aux_data.cend(), it_time);
        if (L_start_it == aux_data.cend() || (**L_start_it).get_value_time() > it_time) {
            LOG4_TRACE("No aux data for time " << it_time);
            continue;
        }

        const auto L_end_time = it_time + label_duration;
        const auto L_end_it = lower_bound(L_start_it, std::min(L_start_it + label_len_1, aux_data.cend()), L_end_time);
        if (L_end_it == L_start_it) {
            LOG4_TRACE("No aux data for time " << it_time << " label ending at " << L_end_time);
            continue;
        }
        const auto F_time = it_time - horizon_duration;
        auto F_end_it = lower_bound(std::max(L_start_it - horizon_len_1, aux_data.cbegin()), L_start_it, F_time);
        if (F_end_it == aux_data.cend() || F_end_it == aux_data.cbegin()) {
            LOG4_TRACE("No feature data for time " << F_time << " label at " << it_time);
            continue;
        }
        const uint32_t F_end_ix = F_end_it - aux_data.cbegin();

        DTYPE(label_ixs)::iterator L_ins;
#pragma omp ordered
        {
            L_ins = label_ixs.insert(label_ixs.cend(), t_label_ix{.n_ixs = label_len});
            ix_F_end.emplace_back(F_end_ix);
            all_times.emplace_back(*it_main_time);
            last_knowns.emplace_back((**--F_end_it)[level]); // Do not use F_end_it after this line
        };
        generate_twap_indexes(aux_data.cbegin(), L_start_it, L_end_it, it_time, L_end_time, resolution_aux, label_len, L_ins->label_ixs);
        LOG4_TRACE(
                "Added row " << label_ixs.size() << " for time " << it_time << " label at " << *L_ins->label_ixs << " feature at " << F_time << " with " << F_end_ix << " index.");
    }
}
#endif

uint32_t time_to_watch_ix = 0xDeadBeef; // TODO Remove after testing
const bpt::ptime time_to_watch = bpt::time_from_string("2025-01-06 04:00:00");

void ModelService::tune_features(arma::mat &out_features, const arma::mat &labels, datamodel::SVRParameters &params, const data_row_container &label_times,
                                 const std::deque<datamodel::DeconQueue_ptr> &feat_queues, const bpt::time_duration &max_gap, const bpt::time_duration &resolution_aux,
                                 const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();
    assert(labels.n_rows == label_times.size());
    const uint32_t n_rows = labels.n_rows;
    const uint32_t lag = params.get_lag_count();
    const auto adjacent_levels = params.get_adjacent_levels();
    const uint32_t coef_lag = datamodel::C_features_superset_coef * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
    const auto coef_lag_ = coef_lag;
#endif
    const uint16_t levels = adjacent_levels.size();
    const uint16_t n_queues = feat_queues.size();
    const uint16_t levels_queues = levels * n_queues;
    arma::vec best_score(levels_queues, arma::fill::value(std::numeric_limits<double>::infinity()));
    datamodel::t_feature_mechanics fm{{levels_queues, arma::fill::none},
                                      {levels_queues * lag, arma::fill::none},
                                      std::deque<arma::uvec>(levels_queues),
                                      {levels_queues * lag, arma::fill::none},
                                      {levels_queues * lag, arma::fill::none}};

    const auto horizon_duration = main_queue_resolution * PROPS.get_prediction_horizon();
    const size_t align_features_size = n_rows * coef_lag * sizeof(double) + n_rows * sizeof(double) + coef_lag * sizeof(float) + coef_lag * sizeof(double) +
                                       coef_lag * sizeof(uint32_t) + n_rows * sizeof(uint32_t);
    const auto &gpu_handler_4 = common::gpu_handler_4::get();
    const auto max_gpu_chunk_size = gpu_handler_4.get_max_gpu_data_chunk_size();
    const uint16_t n_chunks_align = cdiv(align_features_size, max_gpu_chunk_size);
    const uint32_t chunk_len_align = cdiv(coef_lag, n_chunks_align);
    const auto stripe_period = resolution_aux * coef_lag_;
    const arma::vec mean_L = arma::mean(labels, 1);
    const auto earliest_label_time = label_times.front()->get_value_time();
    const auto latest_label_time = label_times.back()->get_value_time();

    std::deque<uint32_t> chunk_len_quantise(n_queues), in_rows(n_queues);
    std::deque<arma::mat> decon(n_queues);
    std::deque<arma::u32_vec> times_F(n_queues);
    std::deque<std::vector<t_feat_params>> feat_params(n_queues);

#pragma omp parallel default(shared) num_threads(gpu_handler_4.get_max_gpu_threads())
#pragma omp single
    {
        OMP_TASKLOOP_1(firstprivate(n_rows, levels))
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues have different amount of samples, fix assumption they are same!
            const auto earliest_time = earliest_label_time - horizon_duration - stripe_period * C_max_quantisation;
            const auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_time - horizon_duration);
            const auto start_iter = lower_bound_or_before_back(*p_queue, last_iter, earliest_time);
            if ((**start_iter).get_value_time() > earliest_time) LOG4_THROW("Start offset " << (**start_iter).get_value_time() << " is after " << earliest_time);
            const uint32_t start_offset = start_iter - p_queue->cbegin();
            in_rows[qix] = last_iter - start_iter;
            const size_t quantise_features_size = n_rows * coef_lag_ * sizeof(double) + in_rows[qix] * sizeof(double) + 2 * n_rows * sizeof(uint32_t) +
                                                  in_rows[qix] * sizeof(uint32_t);
            const uint16_t n_chunks_quantise = cdiv(quantise_features_size, max_gpu_chunk_size);
            chunk_len_quantise[qix] = cdiv(n_rows, n_chunks_quantise);
            decon[qix] = arma::mat(in_rows[qix], levels, arma::fill::none);
            feat_params[qix].resize(n_rows);
            times_F[qix].set_size(in_rows[qix]);

#pragma omp taskloop simd NGRAIN(in_rows[qix]) firstprivate(levels, start_offset) default(shared) mergeable untied
            for (uint32_t r = 0; r < in_rows[qix]; ++r) {
                const auto p_row = p_queue->at(start_offset + r);
                times_F[qix][r] = boost::posix_time::to_time_t(p_row->get_value_time());
                for (uint16_t l = 0; l < levels; ++l) decon[qix](r, l) = p_row->at(adjacent_levels ^ l);
            }

#pragma omp taskloop simd NGRAIN(n_rows) firstprivate(n_rows) default(shared) mergeable untied
            for (uint32_t r = 0; r < n_rows; ++r) {
                if (label_times[r]->get_value_time() == time_to_watch) time_to_watch_ix = r; // TODO Remove after testing

                feat_params[qix][r].end_time = boost::posix_time::to_time_t(label_times[r]->get_value_time() - horizon_duration);
                feat_params[qix][r].ix_end = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), feat_params[qix][r].end_time) - times_F[qix].cbegin();
            }

            OMP_TASKLOOP_1(firstprivate(levels, qix))
            for (DTYPE(levels) adj_ix = 0; adj_ix < levels; ++adj_ix) {
                t_omp_lock ins_l;
                const auto adj_ix_q = adj_ix + qix * levels;
                const auto adj_ix_q_1 = adj_ix_q + 1;

                OMP_TASKLOOP_1(firstprivate(n_rows, lag, coef_lag, adj_ix_q, adj_ix_q_1, qix))
                for (const auto quantise: C_quantisations) {
                    auto feat_params_qix_qt = feat_params[qix];
#pragma omp taskloop SSIMD NGRAIN(n_rows) firstprivate(quantise) default(shared) mergeable untied
                    for (auto &f: feat_params_qix_qt) {
                        f.time_start = f.end_time - coef_lag_ * quantise;
                        f.ix_start = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cbegin() + f.ix_end, f.time_start) - times_F[qix].cbegin();
                    }

                    arma::mat features(n_rows, coef_lag, arma::fill::none);
                    OMP_TASKLOOP_1(firstprivate(n_rows, adj_ix, quantise, coef_lag_, coef_lag))
                    for (uint32_t i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_EXEC_TIME(quantise_features(
                            decon[qix].mem, times_F[qix].mem, feat_params_qix_qt.data(), i,
                            std::min<uint32_t>(i + chunk_len_quantise[qix], n_rows) - i, n_rows, in_rows[qix], adj_ix,
                            coef_lag_, coef_lag, quantise, features.memptr()), "Quantise features " << chunk_len_quantise[qix] << ", quantise " << quantise);
                    RELEASE_CONT(feat_params_qix_qt);
                    arma::vec scores(coef_lag, arma::fill::none);
                    arma::fvec stretches(coef_lag, arma::fill::none), skips(coef_lag, arma::fill::none);
                    arma::u32_vec shifts(coef_lag, arma::fill::none);
                    OMP_TASKLOOP_1(firstprivate(coef_lag, chunk_len_align, n_rows, quantise))
                    for (uint32_t i = 0; i < coef_lag; i += chunk_len_align) PROFILE_EXEC_TIME(align_features(
                            features.colptr(i), mean_L.mem, scores.memptr() + i, stretches.memptr() + i, shifts.memptr() + i, skips.memptr() + i,
                            n_rows, std::min<uint32_t>(i + chunk_len_align, coef_lag) - i), "Align features " << n_rows << "x" << chunk_len_align << ", quantize " << quantise);
                    const arma::uvec trims = arma::uvec(arma::stable_sort_index(scores)).tail(coef_lag - lag);
                    scores.shed_rows(trims);
                    const double score = arma::accu(scores);
                    ins_l.set();
                    if (score < best_score[adj_ix_q]) {
                        LOG4_DEBUG("New best score " << score << ", previous best score " << best_score[adj_ix_q] << ", improvement " << common::imprv(score, best_score[adj_ix_q])
                                                     << "pc, quantise " << quantise << ", aux queue " << qix << ", level " << adj_ix << ", lag " << lag << ", coef lag "
                                                     << coef_lag);
                        best_score[adj_ix_q] = score;
                        fm.quantization[adj_ix_q] = quantise;
                        stretches.shed_rows(trims);
                        shifts.shed_rows(trims);
                        skips.shed_rows(trims);
                        const auto adj_ix_q_lag = adj_ix_q * lag;
                        const auto adj_ix_q_1_lag = adj_ix_q_1 * lag - 1;
                        fm.stretches.rows(adj_ix_q_lag, adj_ix_q_1_lag) = stretches;
                        fm.shifts.rows(adj_ix_q_lag, adj_ix_q_1_lag) = shifts;
                        fm.skips.rows(adj_ix_q_lag, adj_ix_q_1_lag) = skips;
                        fm.trims[adj_ix_q] = trims;
                    }
                    ins_l.unset();
                }
            }
        }
    }
    params.set_feature_mechanics(fm);

    do_features(out_features, n_rows, lag, coef_lag, coef_lag_, levels, n_queues, fm, stripe_period, chunk_len_quantise, in_rows, decon, times_F, feat_params);

    LOG4_END();
}


void ModelService::do_features(
        arma::mat &out_features, const uint32_t n_rows, const uint32_t lag, const uint32_t coef_lag, const uint32_t coef_lag_, const uint16_t levels,
        const uint16_t n_queues, const datamodel::t_feature_mechanics &fm, const boost::posix_time::time_duration &stripe_period,
        const std::deque<uint32_t> &chunk_len_quantise, const std::deque<uint32_t> &in_rows, const std::deque<arma::mat> &decon,
        const std::deque<arma::u32_vec> &times_F, const std::deque<std::vector<t_feat_params>> &feat_params)
{
    const auto levels_lag = levels * lag;
    const auto feature_cols = levels_lag * n_queues;
    if (out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);

#pragma omp parallel num_threads(C_n_cpu)
#pragma omp single
    {
        OMP_TASKLOOP_1(firstprivate(levels_lag))
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            OMP_TASKLOOP_1(firstprivate(lag, coef_lag, qix, levels))
            for (DTYPE(levels) adj_ix = 0; adj_ix < levels; ++adj_ix) {
                const auto adj_ix_q = adj_ix + qix * levels;
                const auto adj_ix_q_lag = adj_ix_q * lag;
                auto feat_params_qix_qt = feat_params[qix];
                const auto quantise = fm.quantization[adj_ix_q];
                const auto quantise_period_sec = (stripe_period * quantise).total_seconds();
#pragma omp taskloop SSIMD NGRAIN(n_rows) default(shared) mergeable untied
                for (auto &f: feat_params_qix_qt) {
                    f.time_start = f.end_time - quantise_period_sec;
                    f.ix_start = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), f.time_start) - times_F[qix].cbegin();
                }
                arma::mat level_features(n_rows, coef_lag, arma::fill::none);
                OMP_TASKLOOP_1(firstprivate(n_rows, adj_ix, coef_lag_, coef_lag))
                for (uint32_t i = 0; i < n_rows; i += chunk_len_quantise[qix]) PROFILE_EXEC_TIME(quantise_features(
                        decon[qix].mem, times_F[qix].mem, feat_params_qix_qt.data(), i, std::min<uint32_t>(i + chunk_len_quantise[qix], n_rows) - i,
                        n_rows, in_rows[qix], adj_ix, coef_lag_, coef_lag, quantise, level_features.memptr()),
                                                                                                 "Prepare quantised features " << chunk_len_quantise[qix]);
                level_features.shed_cols(fm.trims[adj_ix_q]);
                out_features.cols(adj_ix_q_lag, adj_ix_q_lag + lag - 1) = level_features;
                LOG4_TRACE("Level " << adj_ix << ", queue " << qix << ", features " << common::present(level_features));
            }
        }
    }
    if (out_features.empty()) LOG4_WARN("No new data to prepare for training, features " << arma::size(out_features));
    if (time_to_watch_ix != 0xDeadBeef) { // TODO Remove after testing
        LOG4_TRACE("Test predict features " << common::present(out_features.row(time_to_watch_ix)));
        time_to_watch_ix = 0xDeadBeef;
    }
    LOG4_TRACE("Out features " << common::present(out_features));
}

void
ModelService::prepare_features(arma::mat &out_features, const data_row_container &label_times, const std::deque<datamodel::DeconQueue_ptr> &feat_queues,
                               const datamodel::SVRParameters &params, const bpt::time_duration &max_gap, const bpt::time_duration &resolution_aux,
                               const bpt::time_duration &main_queue_resolution)
{
    LOG4_BEGIN();

    const auto horizon_duration = main_queue_resolution * PROPS.get_prediction_horizon();
    const auto earliest_label_time = label_times.front()->get_value_time();
    const auto latest_label_time = label_times.back()->get_value_time();

    const uint32_t n_rows = label_times.size();
    const auto lag = params.get_lag_count();
    const auto adjacent_levels = params.get_adjacent_levels();
    const auto coef_lag = datamodel::C_features_superset_coef * lag;
#ifdef EMO_DIFF
    const auto coef_lag_ = coef_lag + 1;
#else
    const auto coef_lag_ = coef_lag;
#endif
    const uint16_t levels = adjacent_levels.size();
    const uint16_t n_queues = feat_queues.size();
    arma::vec best_score(levels, arma::fill::value(std::numeric_limits<double>::infinity()));
    const auto stripe_period = resolution_aux * coef_lag_;
    LOG4_TRACE("Preparing level " << params.get_decon_level() << ", " << n_rows << " rows, main range from " << earliest_label_time << " until " << latest_label_time <<
        ", lag " << lag << ", " << n_queues << " queues, " << levels << " levels.");

    std::deque<uint32_t> chunk_len_quantise(n_queues), in_rows(n_queues);
    std::deque<arma::mat> decon(n_queues);
    std::deque<arma::u32_vec> times_F(n_queues);
    std::deque<std::vector<t_feat_params>> feat_params(n_queues);
    const auto &fm = params.get_feature_mechanics();
    const auto max_gpu_data_chunk_size = common::gpu_handler_4::get().get_max_gpu_data_chunk_size();
#pragma omp parallel default(shared) num_threads(common::gpu_handler_4::get().get_max_gpu_threads())
#pragma omp single
    {
        OMP_TASKLOOP_1(firstprivate(n_rows, levels))
        for (DTYPE(n_queues) qix = 0; qix < n_queues; ++qix) {
            const auto &p_queue = feat_queues[qix]; // TODO Multiple queues have different amount of samples, fix assumption they are same!
            const auto earliest_time = earliest_label_time - horizon_duration - stripe_period * C_max_quantisation;
            const auto last_iter = lower_bound(std::as_const(*p_queue), latest_label_time - horizon_duration);
            const auto start_iter = lower_bound_or_before_back(*p_queue, last_iter, earliest_time);
            if ((**start_iter).get_value_time() > earliest_time || start_iter == p_queue->cend())
                LOG4_THROW("Start offset " << (**start_iter).get_value_time() << " is after " << earliest_time);
            const uint32_t start_offset = start_iter - p_queue->cbegin();
            in_rows[qix] = last_iter - start_iter;
            const size_t quantise_features_size = n_rows * coef_lag_ * sizeof(double) + in_rows[qix] * sizeof(double) + (2 * n_rows + in_rows[qix]) * sizeof(uint32_t);
            const uint16_t n_chunks_quantise = cdiv(quantise_features_size, max_gpu_data_chunk_size);
            chunk_len_quantise[qix] = cdiv(n_rows, n_chunks_quantise);
            decon[qix] = arma::mat(in_rows[qix], levels, arma::fill::none);
            feat_params[qix].resize(n_rows);
            times_F[qix].set_size(in_rows[qix]);
            LOG4_TRACE("Queue " << qix << ", start offset " << start_offset << ", in rows " << in_rows[qix] << ", quantise features size " << quantise_features_size <<
                ", earliest time " << earliest_time << ", chunks " << n_chunks_quantise << ", chunk rows " << chunk_len_quantise[qix]);

#pragma omp taskloop simd NGRAIN(in_rows[qix]) firstprivate(levels, start_offset) default(shared) mergeable untied
            for (uint32_t r = 0; r < in_rows[qix]; ++r) {
                const auto p_row = p_queue->at(start_offset + r);
                times_F[qix][r] = boost::posix_time::to_time_t(p_row->get_value_time());
                for (uint16_t l = 0; l < levels; ++l) decon[qix](r, l) = p_row->at(adjacent_levels ^ l);
            }

#pragma omp taskloop simd NGRAIN(n_rows) firstprivate(n_rows) default(shared) mergeable untied
            for (uint32_t r = 0; r < n_rows; ++r) {
                if (label_times[r]->get_value_time() == time_to_watch) time_to_watch_ix = r; // TODO Remove after testing

                feat_params[qix][r].end_time = boost::posix_time::to_time_t(label_times[r]->get_value_time() - horizon_duration);
                feat_params[qix][r].ix_end = std::lower_bound(times_F[qix].cbegin(), times_F[qix].cend(), feat_params[qix][r].end_time) - times_F[qix].cbegin();
            }
        }
    }

    const auto levels_lag = levels * lag;
    const auto feature_cols = levels_lag * n_queues;
    if (out_features.n_rows != n_rows || out_features.n_cols != feature_cols) out_features.set_size(n_rows, feature_cols);

    do_features(out_features, n_rows, lag, coef_lag, coef_lag_, levels, n_queues, fm, stripe_period, chunk_len_quantise, in_rows, decon, times_F, feat_params);

    LOG4_END();
}


void
ModelService::train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model)
{
    const auto [p_features, p_labels, p_last_knowns, p_weights, p_times] = get_training_data(dataset, ensemble, model);
    const auto last_value_time = p_times->back()->get_value_time();
    if (model.get_last_modeled_value_time() >= last_value_time) {
        LOG4_DEBUG("No new data to train model " << model << ", last modeled time " << model.get_last_modeled_value_time() << ", last value time " << last_value_time);
        return;
    }
    if (last_value_time < model.get_last_modeled_value_time()) {
        LOG4_ERROR("Data is older " << last_value_time << " than last modeled time " << model.get_last_modeled_value_time());
        return;
    }
    if (model.get_last_modeled_value_time() == bpt::min_date_time)
        train_batch(model, p_features, p_labels, p_last_knowns, p_weights, last_value_time);
    else
        train_online(model, *p_features, *p_labels, *p_last_knowns, *p_weights, last_value_time);
    model.set_last_modeled_value_time(last_value_time);
    model.set_last_modified(bpt::second_clock::local_time());
    LOG4_INFO("Finished training model " << model);
}


void
ModelService::train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::vec &last_knowns, const arma::mat &weights,
                           const bpt::ptime &last_value_time)
{
    arma::mat residuals, learn_labels = labels;
    UNROLL()
    for (uint16_t g = 0; g < model.get_gradient_count(); ++g) {
        const bool is_gradient = g < model.get_gradient_count() - 1;
        const auto &m = model.get_gradient(g);
        if (is_gradient) residuals = learn_labels - m->predict(features, last_value_time);
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
        PROFILE_EXEC_TIME(m->learn(features, learn_labels, last_knowns, weights, last_value_time), "Online SVM train gradient " << g);
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
        const mat_ptr &p_weights,
        const bpt::ptime &last_value_time)
{
    LOG4_BEGIN();

    datamodel::t_gradient_data gradient_data(p_features, p_labels, p_last_knowns);
    UNROLL()
    for (uint16_t gix = 0; gix < model.get_gradient_count(); ++gix) {
        const auto p_gradient = model.get_gradient(gix);
        if (!p_gradient) LOG4_THROW("SVR model for gradient " << gix << " not initialized " << model);
        p_gradient->batch_train(gradient_data.p_features, gradient_data.p_labels, gradient_data.p_last_knowns, p_weights, last_value_time);

        if (model.get_gradient_count() < 2 || gix == model.get_gradient_count() - 1) continue;
        gradient_data = model.get_gradient(gix)->produce_residuals();
        for (auto &p: model.get_gradient(gix + 1)->get_param_set())
            p->set_svr_decremental_distance(gradient_data.p_features->n_rows - C_test_len);
    }

    LOG4_END();
}

arma::vec
ModelService::get_last_knowns(const datamodel::Ensemble &ensemble, const uint16_t level, const data_row_container &times, const bpt::time_duration &resolution)
{
    arma::vec res(times.size());
    const auto p_aux_decon = ensemble.get_label_aux_decon();
    if (!p_aux_decon || p_aux_decon->empty()) LOG4_THROW("No label auxiliary data for ensemble " << ensemble);
    const auto lastknown_offset = PROPS.get_prediction_horizon() * resolution;
    OMP_FOR_i_(res.size(), simd firstprivate(level)) res[i] = (**lower_bound_back_before(*p_aux_decon, times[i]->get_value_time() - lastknown_offset))[level];
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
        const auto this_prediction = p_svr->predict(*predict_features.p_features, predict_features.times.back()->get_value_time());
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
    for (uint16_t levix = 0; levix < p_dataset->get_spectral_levels(); levix += LEVEL_STEP)
        for (uint16_t stepix = 0; stepix < p_dataset->get_multistep(); ++stepix)
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
