#include <boost/date_time/posix_time/ptime.hpp>
#include <armadillo>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>

#include "DQScalingFactorService.hpp"
#include "EnsembleService.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "onlinesvr.hpp"
#include "util/ValidationUtils.hpp"
#include "appcontext.hpp"

#include "DAO/DatasetDAO.hpp"
#include "model/User.hpp"

#include "common/rtp_thread_pool.hpp"
#include "common/gpu_handler.hpp"

#include "SVRParametersService.hpp"


using namespace svr::common;
using namespace svr::datamodel;
using namespace svr::context;


namespace svr {
namespace business {


void DatasetService::load(const datamodel::Dataset_ptr &p_dataset)
{
    if (!p_dataset) LOG4_THROW("Dataset is missing.");

    APP.input_queue_service.load(p_dataset->get_input_queue());
    bpt::time_duration aux_res(boost::date_time::not_a_date_time);
    for (auto &iq: p_dataset->get_aux_input_queues()) {
        APP.input_queue_service.load(iq);
        if (aux_res.is_special()) aux_res = iq->get_resolution();
        else if (aux_res != iq->get_resolution())
            LOG4_THROW("Auxilliary input queue resolution " << iq->get_resolution() << " does not equal " << aux_res);
    }

    const auto id = p_dataset->get_id();
    const size_t levct = p_dataset->get_transformation_levels();
    LOG4_DEBUG("Loading dataset with id " << id << " from database, levels count " << levct);
    p_dataset->set_ensembles(APP.ensemble_service.get_all_by_dataset_id(id));

#pragma omp parallel for
    for (datamodel::Ensemble_ptr &e: p_dataset->get_ensembles())
        APP.ensemble_service.load(p_dataset, e, true);
}


datamodel::Dataset_ptr DatasetService::get(const bigint dataset_id, const bool doload)
{
    datamodel::Dataset_ptr p_dataset = dataset_dao.get_by_id(dataset_id);
    if (doload) load(p_dataset);
    return p_dataset;
}


std::deque<datamodel::Dataset_ptr> DatasetService::find_all_user_datasets(std::string username)
{
    return dataset_dao.find_all_user_datasets(username);
}


bool DatasetService::save(datamodel::Dataset_ptr &p_dataset)
{
    if (!p_dataset) {
        LOG4_ERROR("Dataset is null! Aborting.");
        return false;
    }

    dataset_dao.save(p_dataset);

    auto existing_svr_parameters = svr_parameters_service.get_all_by_dataset_id(p_dataset->get_id());
    svr_parameters_service.remove_by_dataset(p_dataset->get_id());
#pragma omp parallel for
    for (const auto &e: p_dataset->get_ensembles())
        for (const auto &m: e->get_models())
            for (const auto &p: m->get_param_set())
                svr_parameters_service.save(p);

    // save ensembles
    if (!p_dataset->get_ensembles().empty()) {
        ensemble_service.remove_by_dataset_id(p_dataset->get_id());
        if (!p_dataset->get_ensembles().empty()
            && !ensemble_service.save_ensembles(p_dataset->get_ensembles(), true))
            return false;
    }
    return true;
}


bool DatasetService::exists(datamodel::Dataset_ptr const &dataset)
{
    reject_nullptr(dataset);
    return dataset_dao.exists(dataset->get_id());
}


bool DatasetService::exists(int dataset_id)
{
    return dataset_dao.exists(dataset_id);
}


int DatasetService::remove(datamodel::Dataset_ptr const &dataset)
{
    reject_nullptr(dataset);
    svr_parameters_service.remove_by_dataset(dataset->get_id());
#pragma omp parallel for
    for (auto e: dataset->get_ensembles())
        ensemble_service.remove(e);
    return dataset_dao.remove(dataset);
}

int DatasetService::remove(const datamodel::SVRParameters_ptr &svr_parameter)
{
    return svr_parameters_service.remove(svr_parameter);
}


bool DatasetService::link_user_to_dataset(User_ptr const &user, datamodel::Dataset_ptr const &dataset)
{
    return dataset_dao.link_user_to_dataset(user->get_user_name(), dataset);
}


bool DatasetService::unlink_user_from_dataset(User_ptr const &user, datamodel::Dataset_ptr const &dataset)
{
    return dataset_dao.unlink_user_from_dataset(user->get_user_name(), dataset);
}

DatasetService::DatasetUsers::DatasetUsers(datamodel::Dataset_ptr const &dataset, std::deque<User_ptr> &&users)
        : dataset(dataset), users(std::forward<std::deque<User_ptr>>(users))
{}

namespace {
bool is_dataset_element_of(const datamodel::Dataset_ptr &p_dataset, const svr::dao::DatasetDAO::UserDatasetPairs &active_datasets)
{
    for (auto &dataset_users: active_datasets)
        if (p_dataset->get_id() == dataset_users.second->get_id())
            return true;
    LOG4_DEBUG("Dataset " << p_dataset->get_id() << " is not element of active datasets.");
    return false;
}
}


void DatasetService::update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs)
{
    svr::dao::DatasetDAO::UserDatasetPairs active_datasets = dataset_dao.get_active_datasets();

    // Remove inactive datasets
    auto new_end = std::remove_if(
            processed_user_dataset_pairs.begin(),
            processed_user_dataset_pairs.end(),
            [&active_datasets](DatasetUsers &processed_pair) -> bool { return !is_dataset_element_of(processed_pair.dataset, active_datasets); }
    );
    if (new_end != processed_user_dataset_pairs.end()) {
        LOG4_DEBUG("Erasing from " << new_end->dataset->dataset_name_ << " until end.");
        processed_user_dataset_pairs.erase(new_end, processed_user_dataset_pairs.end());
    }

    for (auto &pudp: processed_user_dataset_pairs)
        pudp.users.clear();

    for (auto &dataset_pair: active_datasets) {
        auto compare_by_dataset_id = [&dataset_pair](UserDatasetPairs::value_type const &other) {
            if (not dataset_pair.second or not other.dataset) LOG4_THROW("Null ptr value passed");
            return dataset_pair.second->get_id() == other.dataset->get_id();
        };

        auto iter = std::find_if(processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
                                 compare_by_dataset_id);
        if (iter == processed_user_dataset_pairs.end()) {
            load(dataset_pair.second);
            processed_user_dataset_pairs.emplace_back(
                    UserDatasetPairs::value_type(
                            dataset_pair.second,
                            {APP.user_service.get_user_by_user_name(dataset_pair.first)}
                    ));
        } else
            iter->users.emplace_back(APP.user_service.get_user_by_user_name(dataset_pair.first));
    }

    auto compare_user_dataset_priority = [](UserDatasetPairs::value_type const &lhs,
                                            UserDatasetPairs::value_type const &rhs) {
        auto i_lhs = lhs.users.begin(), i_rhs = rhs.users.begin(), end_lhs = lhs.users.end(), end_rhs = rhs.users.end();

        for (; i_lhs != end_lhs && i_rhs != end_rhs; ++i_lhs, ++i_rhs) {
            if ((*i_lhs)->get_priority() > (*i_rhs)->get_priority())
                return true;
            if ((*i_lhs)->get_priority() < (*i_rhs)->get_priority())
                return false;
        }

        if (lhs.dataset->get_priority() > rhs.dataset->get_priority())
            return true;
        if (lhs.dataset->get_priority() < rhs.dataset->get_priority())
            return false;

        return false;
    };

    std::sort(processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(), compare_user_dataset_priority);
}


bool DatasetService::exists(const std::string &user_name, const std::string &dataset_name)
{
    return dataset_dao.exists(user_name, dataset_name);
}


datamodel::Dataset_ptr DatasetService::get_user_dataset(const std::string &user_name, const std::string &dataset_name)
{
    datamodel::Dataset_ptr p_dataset = dataset_dao.get_by_name(user_name, dataset_name);
    if (p_dataset) {
        p_dataset->set_input_queue(AppContext::get_instance().input_queue_service.get_queue_metadata(
                p_dataset->get_input_queue()->get_table_name()));
        load(p_dataset);
    }
    return p_dataset;
}


bool
DatasetService::prepare_training_data(
        datamodel::Dataset_ptr &p_dataset,
        datamodel::Ensemble_ptr &p_ensemble,
        tbb::concurrent_map<size_t, matrix_ptr> &features,
        tbb::concurrent_map<size_t, matrix_ptr> &labels,
        tbb::concurrent_map<size_t, bpt::ptime> &last_row_time)
{
    LOG4_BEGIN();

    const auto levct = p_dataset->get_transformation_levels();
    const size_t half_levct = levct / 2;
    const auto p_label_decon = p_ensemble->get_decon_queue();
    const auto p_label_aux_decon = DeconQueueService::find_decon_queue(
            p_ensemble->get_aux_decon_queues(), p_ensemble->get_aux_decon_queue()->get_input_queue_table_name(), p_label_decon->get_input_queue_column_name());
    const auto aux_feats_data = [&p_ensemble](){
        std::deque<datarow_range> r;
        for (const auto &dq: p_ensemble->get_aux_decon_queues()) r.emplace_back(dq->get_data());
        return r;
    }();
    const datarow_range aux_label_data(p_label_aux_decon->get_data());

    const auto resolution_factor = double(p_dataset->get_input_queue()->get_resolution().total_microseconds()) / double(p_dataset->get_aux_input_queue()->get_resolution().total_microseconds());
    std::deque<bpt::ptime> label_times;
    predictions_t tune_predictions;
    tbb::concurrent_map<size_t, matrix_ptr> last_knowns;
    tbb::concurrent_map<size_t, double> scale_label, dc_offset;
#pragma omp parallel for num_threads(common::gpu_handler::get().get_max_running_gpu_threads_number()) schedule(static, 1)
    for (size_t levix = 0; levix < levct; levix += 2) {
        if (levix == half_levct) continue;
        const auto p_model = p_ensemble->get_model(levix);
        auto p_head_params = p_model->get_params_ptr();
        features[levix] = std::make_shared<arma::mat>();
        labels[levix] = std::make_shared<arma::mat>();
        last_knowns[levix] = std::make_shared<arma::mat>();
        APP.model_service.get_training_data(
                *features[levix], *labels[levix], *last_knowns[levix], label_times,
                {svr::business::EnsembleService::get_start( // Main labels
                        p_label_decon->get_data(),
                        p_head_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE,
                        0,
                        p_model->get_last_modeled_value_time(),
                        p_dataset->get_input_queue()->get_resolution()),
                 p_label_decon->get_data().end(),
                 p_label_decon->get_data()},
                aux_label_data,
                aux_feats_data,
                p_head_params->get_lag_count(),
                p_model->get_learning_levels(),
                p_dataset->get_max_lookback_time_gap(),
                levix,
                resolution_factor,
                p_model->get_last_modeled_value_time(),
                p_dataset->get_input_queue()->get_resolution());

        last_row_time[levix] = p_label_decon->get_data().back()->get_value_time();
        APP.dq_scaling_factor_service.scale(p_dataset, p_ensemble->get_aux_decon_queues(), p_head_params, *features[levix], *labels[levix], *last_knowns[levix]);
        if (!std::isnormal(p_head_params->get_svr_kernel_param()))
            PROFILE_EXEC_TIME(OnlineMIMOSVR::tune_kernel_params(
                tune_predictions[p_head_params->get_decon_level()], p_model->get_param_set(), *features[levix], *labels[levix], *last_knowns[levix], p_dataset->get_chunk_size()),
                              "Tune kernel params for model " << p_head_params->get_decon_level());
        scale_label[levix] = p_dataset->get_dq_scaling_factor_labels(p_label_aux_decon->get_input_queue_table_name(), p_label_aux_decon->get_input_queue_column_name(), levix);
        dc_offset[levix] = levix ? 0 : p_dataset->get_dq_scaling_factor_labels(p_label_aux_decon->get_input_queue_table_name(), p_label_aux_decon->get_input_queue_column_name(), DC_INDEX);
    }

#pragma omp parallel for
    for (size_t chunk_ix = 0; chunk_ix < tune_predictions.begin()->second.size(); ++chunk_ix)
        PROFILE_EXEC_TIME(recombine_params(tune_predictions, p_ensemble, levct, scale_label, dc_offset, chunk_ix),
                          "Recombine parameters for " << half_levct - 1 << " models, chunk " << chunk_ix);

#ifdef TRIM_DATA // TODO Buggy check why needed data is cleared!
    const auto leftover_len = p_dataset->get_max_lag_count() + p_dataset->get_residuals_length() + 1;
    const auto leftover_len_aux = leftover_len * resolution_factor;
    if (p_dataset->get_input_queue()->get_data().size() > leftover_len)
        p_dataset->get_input_queue()->get_data().erase(p_dataset->get_input_queue()->get_data().begin(), p_dataset->get_input_queue()->get_data().end() - leftover_len);
    if (p_label_decon->get_data().size() > leftover_len)
        p_label_decon->get_data().erase(p_label_decon->get_data().begin(), p_label_decon->get_data().end() - leftover_len);
    p_dataset->get_input_queue()->get_data().shrink_to_fit();
    p_label_decon->get_data().shrink_to_fit();
#pragma omp parallel for
    for (auto p_aux_input: p_dataset->get_aux_input_queues()) {
        if (p_aux_input->get_data().size() > leftover_len_aux) {
            p_aux_input->get_data().erase(p_aux_input->get_data().begin(), std::prev(lower_bound_back(p_aux_input->get_data(), p_dataset->get_input_queue()->get_data().rbegin()->get()->get_value_time()), leftover_len_aux));
            p_aux_input->get_data().shrink_to_fit();
        }
        for (auto p_aux_column: p_dataset->get_input_queue()->get_value_columns()) {
            auto p_aux_decon = p_dataset->get_decon_queue(p_aux_input, p_aux_column);
            if (p_aux_decon->get_data().size() > leftover_len_aux) {
                p_aux_decon->get_data().erase(
                        p_aux_input->get_data().begin(),
                        std::prev(lower_bound_back(
                                p_aux_input->get_data(),
                                p_dataset->get_input_queue()->get_data().rbegin()->get()->get_value_time()), leftover_len_aux));
                p_aux_decon->get_data().shrink_to_fit();
            }
        }
    }
#endif

    LOG4_END();

    return true;
}


void DatasetService::join_features(tbb::concurrent_map<size_t, tbb::concurrent_map<size_t, matrix_ptr>> &features, const size_t levct, const size_t ensct)
{
#pragma omp parallel for
    for (size_t levix = 0; levix < levct; levix += 2) {
        if (levix == levct / 2) continue;
        for (size_t ensix = 1; ensix < ensct; ++ensix) { // Not parallelizable // Features past ensix 0 become invalid
            features[0][levix]->insert_cols(features[0][levix]->n_cols, *features[ensix][levix]);
            features[ensix][levix]->clear();
        }
    }
}

size_t DatasetService::to_levix(const size_t modix)
{
    return (modix >= 16 ? (modix + 1) : modix) * 2;
}

void DatasetService::process_dataset(datamodel::Dataset_ptr &p_dataset)
{
    InputQueueService::prepare_queues(p_dataset);

    const auto levct = p_dataset->get_transformation_levels();
#ifdef NO_ONLINE_LEARN
    if (p_dataset->get_initialized()) {
#pragma omp parallel for
        for (auto &e: p_dataset->get_ensembles()) {
#pragma omp parallel for
            for (auto &m: e->get_models()) {
                if (!m) continue;
                const auto new_last_modeled_time = e->get_decon_queue()->get_data().back()->get_value_time();
                if (m->get_last_modeled_value_time() >= new_last_modeled_time) continue;
                m->set_last_modeled_value_time(new_last_modeled_time);
                m->set_last_modified(bpt::second_clock::local_time());
            }
        }
        return;
    }
#endif

    const auto ensct = p_dataset->get_ensembles().size();
    tbb::concurrent_map<size_t, tbb::concurrent_map<size_t, bpt::ptime>> last_row_time;
    tbb::concurrent_map<size_t, tbb::concurrent_map<size_t, matrix_ptr>> features, labels;

    bool train_ret = true;
#pragma omp parallel for
    for (size_t ensix = 0; ensix < ensct; ++ensix) {
        auto p_ensemble = p_dataset->get_ensemble(ensix);
#ifdef MANIFOLD_TEST
        PROFILE_EXEC_TIME(process_dataset_test_tune(p_dataset, p_ensemble), "Prepare data test and tune");
#else
        bool ret;
        PROFILE_EXEC_TIME(ret = prepare_training_data(p_dataset, p_ensemble, features[ensix], labels[ensix], last_row_time[ensix]), "Prepare data ensemble " << ensix);
#pragma omp critical
        train_ret &= ret;
#endif
    }
#ifdef MANIFOLD_TEST
    exit(0);
#endif
    if (!train_ret) {
        LOG4_ERROR("Aborting training.");
        return;
    }

    join_features(features, ensct, levct); // labels differ but aux decon queues and features are shared among ensembles

#pragma omp parallel for collapse(2)
    for (size_t levix = 0; levix < levct; levix += 2)
        for (size_t ensix = 0; ensix < ensct; ++ensix)
            if (levix != levct / 2)
                ModelService::train(p_dataset->get_ensemble(ensix)->get_model(levix), features[0][levix], labels[ensix][levix], last_row_time[ensix][levix]);

    p_dataset->set_initialized();

    LOG4_END();
}

// Return value contains the needed data without the decomposition tail length
boost::posix_time::time_period
DatasetService::get_training_range(const datamodel::Dataset_ptr &p_dataset)
{
    LOG4_BEGIN();

    if (!p_dataset->get_input_queue()) LOG4_THROW("Input queue not initialized!");
    const auto max_decrement_distance = p_dataset->get_max_decrement();
    const auto newest_row = APP.input_queue_service.find_newest_record(p_dataset->get_input_queue());
    if (!newest_row) LOG4_THROW("Could not find newest record for input queue " << p_dataset->get_input_queue()->get_table_name());
    const auto input_queue_newest_value_time = newest_row->get_value_time();
    boost::posix_time::ptime oldest_start_train_time = boost::posix_time::min_date_time;
    std::set<boost::posix_time::ptime> ensemble_train_start_times;
    /* Determine start and end training time range for every ensemble based on every model parameters. */
#pragma omp parallel for
    for (const auto &e: p_dataset->get_ensembles()) {
        const boost::posix_time::ptime start_at = e->get_models().empty() || e->get_model()->get_last_modeled_value_time() == bpt::min_date_time ?
                                                  APP.input_queue_service.load_nth_last_row(p_dataset->get_input_queue(),
                                                                                            max_decrement_distance + MANIFOLD_TEST_VALIDATION_WINDOW)->get_value_time() :
                                                  e->get_model()->get_last_modeled_value_time();
        if (start_at == input_queue_newest_value_time)
            LOG4_DEBUG("No new data in input queue to process for ensemble " << e->get_id());
#pragma omp critical
        ensemble_train_start_times.insert(start_at);
    }
    if (!ensemble_train_start_times.empty()) oldest_start_train_time = *ensemble_train_start_times.begin();
    if (oldest_start_train_time == bpt::min_date_time)
        LOG4_THROW("Some models were not initialized with train, aborting!");
    if (oldest_start_train_time > input_queue_newest_value_time)
        LOG4_THROW("Ensemble oldest start train time " << oldest_start_train_time << " is newer than input queue last value time " << input_queue_newest_value_time);
    LOG4_DEBUG("Input queue time period for training all ensembles from " << oldest_start_train_time << " to " << input_queue_newest_value_time);
    return {oldest_start_train_time, input_queue_newest_value_time - oldest_start_train_time};
}


struct save_response
{
    const std::string &value_column;
    const datamodel::MultivalRequest_ptr &p_request;

    save_response(const std::string &value_column_, const datamodel::MultivalRequest_ptr &p_request_) : value_column(value_column_), p_request(p_request_)
    {}

    void operator()(const std::shared_ptr<svr::datamodel::DataRow> &p_result_row)
    {
        if (p_result_row->get_value_time() < p_request->value_time_start or p_result_row->get_value_time() > p_request->value_time_end) {
            LOG4_DEBUG("Skipping save p_response " << p_result_row->to_string());
            return;
        }

        auto resp_val = p_result_row->get_value(0);
        const auto resp_time = p_result_row->get_value_time();
#ifndef OFFSET_PRED_MUL
        // Small hack for improving performance consequent p_predictions
        {
            const auto input_colix = InputQueueService::get_value_column_index(p_input_queue, value_column);
            static std::atomic<data_row_container::iterator> aux_hint{p_aux_input_queue->get_data().begin()};
            const auto last_minute_pc = std::prev(
                    lower_bound(p_aux_input_queue->get_data(), aux_hint.load(), resp_time))->second->get_value(input_colix);
            const auto last_hour_pc = std::prev(
                    p_input_queue->get_data().lower_bound(resp_time))->second->get_value(input_colix);

            if (last_minute_pc < last_hour_pc != resp_val > last_hour_pc) resp_val = last_minute_pc;
            aux_hint.store(last_minute_pc);
        }
#endif
        datamodel::MultivalResponse_ptr p_response = std::make_shared<MultivalResponse>(0, p_request->get_id(), resp_time, value_column, resp_val);
        LOG4_TRACE("Saving " << p_response->to_string());
        AppContext::get_instance().request_service.save(p_response); // Requests get marked processed here
    }
};

// By specification, a multival request's time period to predict is
// [start_predict_time, end_predict_time) i.e. right-hand exclusive.
void DatasetService::process_mimo_multival_requests(const User_ptr &p_user, datamodel::Dataset_ptr &p_dataset)
{
    LOG4_DEBUG("Processing " << *p_user << " requests for dataset " << *p_dataset);
    datamodel::InputQueue_ptr p_dataset_input_queue = p_dataset->get_input_queue();
    const auto main_resolution = p_dataset_input_queue->get_resolution();
    const auto main_to_aux_period_ratio = main_resolution / p_dataset->get_aux_input_queue()->get_resolution();
    auto active_requests = AppContext::get_instance().request_service.get_active_multival_requests(*p_user, *p_dataset);
    if (active_requests.empty()) LOG4_DEBUG("No active requests!");
#pragma omp parallel for
    for (const auto &p_active_request: active_requests) {
        if (!p_active_request->sanity_check()) {
            LOG4_ERROR("Request " << *p_active_request << " incorrect!");
            continue;
        }

        LOG4_DEBUG("Processing request " << *p_active_request);

        const auto start_predict_time = p_active_request->value_time_start;
        const auto end_predict_time = p_active_request->value_time_end;
        const auto predict_start_times_grid = [&](){
            std::set<bpt::ptime> r;
            if (start_predict_time == end_predict_time)
                r.insert(start_predict_time);
            else
                for (auto t = start_predict_time; t < end_predict_time; t += main_resolution)
                    r.insert(t);
            return r;
        }();
        std::atomic_bool request_answered {true};
        const auto request_columns = from_sql_array(p_active_request->value_columns);
        tbb::concurrent_map<std::string, datamodel::DeconQueue_ptr> request_decons;

        if (predict_start_times_grid.empty()) {
            LOG4_ERROR("Times grid empty.");
            continue;
        }

        const auto levct = p_dataset->get_transformation_levels();
        tbb::concurrent_map<std::tuple<
                bpt::ptime /* prediction start time */,
                size_t /* ensemble index */,
                size_t /* level index */>,
                arma::rowvec /* features vector */>
                pred_features;
        // Prepare per ensemble features
#pragma omp parallel for collapse(2)
        for (size_t predix = 0; predix < predict_start_times_grid.size(); ++predix) {
            for (size_t ensix = 0; ensix < request_columns.size(); ++ensix) {
                const auto pred_time = predict_start_times_grid ^ predix;
                const auto p_ensemble = p_dataset->get_ensemble(request_columns[ensix]);
                if (!p_ensemble) {
                    LOG4_ERROR("Ensemble for request column " << request_columns[ensix] << " not found.");
                    request_answered.store(false, std::memory_order_relaxed);
                    continue;
                }
#pragma omp parallel for
                for (size_t modix = 0; modix < levct; modix += 2) {
                    if (modix == levct / 2) continue;
                    APP.model_service.get_features_row(
                            pred_time,
                            p_ensemble->get_aux_decon_queues(),
                            p_ensemble->get_model(modix)->get_learning_levels(),
                            modix,
                            p_dataset->get_max_lookback_time_gap(),
                            main_to_aux_period_ratio,
                            p_ensemble->get_model(modix)->get_params().get_lag_count(),
                            p_dataset_input_queue->get_resolution(),
                            pred_features[{pred_time, ensix, modix}]);
                    if (!pred_features[{pred_time, ensix, modix}].empty()) {
                        request_answered.store(false, std::memory_order_relaxed);
                        continue;
                    }
                    APP.dq_scaling_factor_service.scale(
                            p_dataset,
                            p_ensemble->get_aux_decon_queues(),
                            p_ensemble->get_model(modix)->get_params_ptr(),
                            pred_features[{pred_time, ensix, modix}]);
                }
            }
        }
        if (!request_answered.load(std::memory_order_relaxed)) {
            LOG4_ERROR("Skipping request " << p_active_request->get_id());
            continue;
        }

        // Join rows of prediction features for all dataset ensembles, for other (correlated) input queue column predictions
#pragma omp parallel for collapse(2)
        for (size_t predix = 0; predix < predict_start_times_grid.size(); ++predix) {
            for (size_t modix = 0; modix < levct; modix += 2) {
                if (modix == levct / 2) continue;
                const auto pred_time = predict_start_times_grid ^ predix;
                for (size_t ensix = 1; ensix < p_dataset->get_ensembles().size(); ++ensix) { // Features past ensix 0 become invalid // Non-parallelizabille
                    pred_features[{pred_time, 0, modix}].insert_cols(pred_features[{pred_time, 0, modix}].n_cols, pred_features[{pred_time, ensix, modix}]);
                    pred_features[{pred_time, ensix, modix}].clear();
                }
            }
        }

        // Do actual predictions
#pragma omp parallel for
        for (const auto &request_column: request_columns) {
#pragma omp parallel for
            for (size_t predix = 0; predix < predict_start_times_grid.size(); ++predix) {
                const auto pred_time = predict_start_times_grid ^ predix;
                const auto p_ensemble = p_dataset->get_ensemble(request_column);

                if (!p_ensemble) {
                    LOG4_ERROR("Ensemble for request column " << request_column << " not found.");
                    request_answered.store(false, std::memory_order_relaxed);
                    continue;
                }

                auto p_aux_decon = p_ensemble->get_aux_decon_queue(request_column);
                if (!p_aux_decon) {
                    LOG4_ERROR("Decon queue for column " << request_column << " not found.");
                    request_answered.store(false, std::memory_order_relaxed);
                    continue;
                }
                p_aux_decon = p_aux_decon->clone_empty();
                request_decons[request_column] = p_aux_decon;
                arma::mat predictions(PROPS.get_multistep_len(), levct, arma::fill::zeros);
#pragma omp parallel for
                for (size_t levix = 0; levix < levct; levix += 2) {
                    if (levix == levct / 2) continue;
                    const auto aux_dq_scaling_factors = APP.dq_scaling_factor_service.slice(
                            p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), p_aux_decon->get_input_queue_table_name(),
                            p_aux_decon->get_input_queue_column_name(), p_ensemble->get_model(levix)->get_learning_levels());
                    auto level_preds = p_ensemble->get_model(levix)->get_gradient(0)->predict(pred_features[{pred_time, 0, levix}], pred_time);
                    level_preds = arma::mean(level_preds, 1);
                    predictions.col(levix) = DQScalingFactorService::unscale(level_preds, levix, aux_dq_scaling_factors);
                }

#pragma omp critical
                datamodel::DataRow::insert_rows(p_aux_decon->get_data(), predictions, pred_time, main_resolution);
            }

            const auto reconstructed_data = APP.decon_queue_service.reconstruct(
                    datamodel::datarow_range(request_decons[request_column]->get_data()),
                    recon_type_e::ADDITIVE, // TODO Add conversion of decon string to recon type, as M to N mapping
                    p_dataset->get_transformation_levels());

            if (reconstructed_data.empty()) {
                LOG4_ERROR("Empty reconstructed data.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            LOG4_DEBUG("Saving reconstructed p_predictions from " << reconstructed_data.begin()->get()->get_value_time() << " until "
                                                                  << reconstructed_data.rbegin()->get()->get_value_time());
            std::for_each(std::execution::par_unseq,
                          find(reconstructed_data, start_predict_time),
                          reconstructed_data.end(),
                          save_response(request_column, p_active_request));
        }

        if (request_answered)
            APP.request_service.force_finalize(p_active_request);
        else
            LOG4_WARN("Failed finalizing request id " << p_active_request->get_id());
    }
}


} //business
} //svr
