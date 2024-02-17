#include <boost/date_time/posix_time/ptime.hpp>
#include <armadillo>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <memory>
#include <oneapi/tbb/concurrent_map.h>
#include <utility>
#include <algorithm>
#include <vector>

#include "DQScalingFactorService.hpp"
#include "EnsembleService.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "model/DataRow.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"

#include "DAO/DatasetDAO.hpp"
#include "model/User.hpp"

#include "common/rtp_thread_pool.hpp"
#include "common/gpu_handler.hpp"

#include "SVRParametersService.hpp"
#include "ModelService.hpp"


namespace svr {
namespace business {


void DatasetService::load(datamodel::Dataset_ptr &p_dataset)
{
    if (!p_dataset) LOG4_THROW("Dataset is missing.");

    APP.input_queue_service.load(p_dataset->get_input_queue());
    bpt::time_duration aux_res(boost::date_time::not_a_date_time);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_aux_input_queues().size()))
    for (size_t i = 0; i < p_dataset->get_aux_input_queues().size(); ++i) {
        const auto &iq = p_dataset->get_aux_input_queue(i);
        APP.input_queue_service.load(iq);
#pragma omp critical
        {
            if (aux_res.is_special()) aux_res = iq->get_resolution();
            else if (aux_res != iq->get_resolution())
                LOG4_THROW("Auxiliary input queue resolution " << iq->get_resolution() << " does not equal " << aux_res);
        }
    }

    p_dataset->set_ensembles(APP.ensemble_service.get_all_by_dataset_id(p_dataset->get_id()));
    if (p_dataset->get_ensembles().empty()) EnsembleService::init_default_ensembles(p_dataset);

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_ensembles().size()))
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
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (const auto &e: p_dataset->get_ensembles())
#pragma omp parallel for num_threads(adj_threads(e->get_models().size()))
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
    common::reject_nullptr(dataset);
    return dataset_dao.exists(dataset->get_id());
}


bool DatasetService::exists(int dataset_id)
{
    return dataset_dao.exists(dataset_id);
}


int DatasetService::remove(datamodel::Dataset_ptr const &dataset)
{
    common::reject_nullptr(dataset);
    svr_parameters_service.remove_by_dataset(dataset->get_id());
#pragma omp parallel for num_threads(adj_threads(dataset->get_ensembles().size()))
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
        p_dataset->set_input_queue(context::AppContext::get_instance().input_queue_service.get_queue_metadata(
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
        tbb::concurrent_map<size_t, matrix_ptr> &last_knowns,
        tbb::concurrent_map<size_t, bpt::ptime> &last_row_time)
{
    LOG4_BEGIN();

    const auto p_label_decon = p_ensemble->get_label_decon();
    const auto p_label_aux_decon = p_ensemble->get_label_aux_decon();
    const auto aux_feats_data = [&p_ensemble](){
        std::deque<datamodel::datarow_range> r;
        for (const auto &dq: p_ensemble->get_aux_decon_queues())
            r.emplace_back(*dq);
        return r;
    }();

    std::deque<bpt::ptime> label_times;
    tbb::concurrent_map<size_t, double> scale_label, dc_offset;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_ensemble->get_models().size()))
    for (const auto &p_model: p_ensemble->get_models()) {
        const auto levix = p_model->get_decon_level();
        auto p_head_params = p_model->get_params_ptr();
        features[levix] = std::make_shared<arma::mat>();
        labels[levix] = std::make_shared<arma::mat>();
        last_knowns[levix] = std::make_shared<arma::mat>();
        APP.model_service.get_training_data(
                *features[levix], *labels[levix], *last_knowns[levix], label_times,
                {svr::business::EnsembleService::get_start( // Main labels are used for timing
                        p_label_decon->get_data(),
                        p_head_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE,
                        p_model->get_last_modeled_value_time(),
                        p_dataset->get_input_queue()->get_resolution()),
                 p_label_decon->end(),
                 p_label_decon->get_data()},
                *p_label_aux_decon,
                aux_feats_data,
                p_head_params->get_lag_count(),
                common::get_adjacent_indexes(levix, p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels()),
                p_dataset->get_max_lookback_time_gap(),
                levix,
                p_dataset->get_aux_input_queues().empty() ? p_dataset->get_input_queue()->get_resolution() : p_dataset->get_aux_input_queue()->get_resolution(),
                p_model->get_last_modeled_value_time(),
                p_dataset->get_input_queue()->get_resolution(),
                p_model->get_multiout());

        last_row_time[levix] = p_label_decon->back()->get_value_time();
        APP.dq_scaling_factor_service.scale(p_dataset, p_ensemble->get_aux_decon_queues(), p_head_params, *features[levix], *labels[levix], *last_knowns[levix]);
    }

    LOG4_END();

    return true;
}


tbb::concurrent_map<size_t, matrix_ptr>
DatasetService::join_features(
        t_enscon<matrix_ptr> &features,
        const size_t levct,
        const std::deque<datamodel::Ensemble_ptr> &ensembles)
{
    tbb::concurrent_map<size_t, matrix_ptr> result = features[ensembles.begin()->get()->get_column_name()];
    if (ensembles.size() < 2) goto __bail;
#pragma omp parallel for num_threads(adj_threads(levct))
    for (size_t levix = 0; levix < levct; levix += 2) {
        if (levix == levct / 2) continue;
        for (auto ens_it = std::next(ensembles.begin()); ens_it != ensembles.end(); ++ens_it) { // Non-parallelizabile
            const std::string column_name = ens_it->get()->get_column_name();
            result[levix]->insert_cols(result[levix]->n_cols, *features[column_name][levix]);
            features[column_name][levix]->clear();
        }
    }
__bail:
    return result;
}

#ifdef MANIFOLD_TEST

void DatasetService::process(datamodel::Dataset_ptr &p_dataset)
{
    InputQueueService::prepare_queues(p_dataset);
    t_enscon<bpt::ptime> last_row_time;
    t_enscon<matrix_ptr> features, labels;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (size_t ensix = 0; ensix < p_dataset->get_ensembles().size(); ++ensix) {
        auto p_ensemble = p_dataset->get_ensemble(ensix);
        PROFILE_EXEC_TIME(process_dataset_test_tune(p_dataset, p_ensemble), "Prepare data test and tune");
    }
    exit(0);
}

#else

void DatasetService::process(datamodel::Dataset_ptr &p_dataset)
{
    InputQueueService::prepare_queues(p_dataset);

    const auto levct = p_dataset->get_transformation_levels();
    t_enscon<bpt::ptime> last_row_time;
    t_enscon<matrix_ptr> features, labels, last_knowns;

    bool train_ret = true;
#pragma omp parallel for reduction(&:train_ret) num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (auto p_ensemble: p_dataset->get_ensembles())
        PROFILE_EXEC_TIME(
                train_ret &= prepare_training_data(
                        p_dataset, p_ensemble,
                        features[p_ensemble->get_column_name()],
                        labels[p_ensemble->get_column_name()],
                        last_knowns[p_ensemble->get_column_name()],
                        last_row_time[p_ensemble->get_column_name()]),
                          "Prepare data for ensemble " << p_ensemble->get_column_name());
    if (!train_ret) {
        LOG4_ERROR("Prepare data failed, aborting training.");
        return;
    }

    const auto dataset_features = join_features(features, levct, p_dataset->get_ensembles()); // labels differ but aux decon queues and features are shared among ensembles

#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (auto p_ensemble: p_dataset->get_ensembles())
        EnsembleService::train(
                p_dataset, p_ensemble, dataset_features,
                labels[p_ensemble->get_column_name()], last_knowns[p_ensemble->get_column_name()], last_row_time[p_ensemble->get_column_name()]);

    LOG4_END();
}

#endif

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
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
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
            static std::atomic<data_row_container::iterator> aux_hint{p_aux_input_queue->begin()};
            const auto last_minute_pc = std::prev(
                    lower_bound(p_aux_input_queue->get_data(), aux_hint.load(), resp_time))->second->get_value(input_colix);
            const auto last_hour_pc = std::prev(
                    p_input_queue->get_data().lower_bound(resp_time))->second->get_value(input_colix);

            if (last_minute_pc < last_hour_pc != resp_val > last_hour_pc) resp_val = last_minute_pc;
            aux_hint.store(last_minute_pc);
        }
#endif
        datamodel::MultivalResponse_ptr p_response = std::make_shared<datamodel::MultivalResponse>(0, p_request->get_id(), resp_time, value_column, resp_val);
        LOG4_TRACE("Saving " << p_response->to_string());
        context::AppContext::get_instance().request_service.save(p_response); // Requests get marked processed here
    }
};

auto DatasetService::prepare_request_features(const datamodel::Dataset_ptr &p_dataset, const std::set<bpt::ptime> &predict_times)
{
    tbb::concurrent_map<size_t /* level */, std::pair<std::set<bpt::ptime /* feature row times */>, arma::mat /* features matrix */>> res;
    std::atomic<bool> fail = false;
#pragma omp parallel for ordered schedule(static, 1) num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (const auto &p_ensemble: p_dataset->get_ensembles()) {
#pragma omp parallel for ordered schedule(static, 1) num_threads(adj_threads(p_ensemble->get_models().size()))
        for (const auto &p_model: p_ensemble->get_models()) {

            arma::mat level_features(predict_times.size(), 0);
            const auto p_head_params = p_model->get_params_ptr();

            for (size_t predix = 0; predix < predict_times.size(); ++predix) {
                const auto pred_time = predict_times ^ predix;
                arma::rowvec features;
                try {
                    APP.model_service.get_features_row(p_dataset, p_ensemble, p_model, pred_time, features);
                } catch (const std::exception &ex) {
                    LOG4_ERROR("Failed preparing features for " << pred_time << ", level " << p_model->get_decon_level() << ", column " << p_ensemble->get_column_name() << ", " << ex.what());
                    fail.store(true, std::memory_order_relaxed);
                }
                if (level_features.n_cols != features.n_cols)
                    level_features.set_size(level_features.n_rows, features.n_cols);
                level_features.row(predix) = features;
            }

            const std::set<size_t> feat_levels = common::get_adjacent_indexes(
                    p_head_params->get_decon_level(), p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
            APP.dq_scaling_factor_service.scale_features(p_dataset, p_ensemble->get_aux_decon_queues(), p_head_params, level_features, p_dataset->get_dq_scaling_factors(), feat_levels);
#pragma omp ordered
            {
                decltype(res)::iterator it_res;
                if ((it_res = res.find(p_model->get_decon_level())) == res.end())
                    res.emplace(p_model->get_decon_level(), std::pair{predict_times, level_features});
                else
                    it_res->second.second = arma::join_rows(it_res->second.second, level_features);
            }
        }
    }
    if (fail) res.clear();
    return res;
}


// By specification, a multival request's time period to predict is [start_predict_time, end_predict_time) i.e. right-hand exclusive.
void DatasetService::process_requests(const User_ptr &p_user, datamodel::Dataset_ptr &p_dataset)
{
    LOG4_DEBUG("Processing " << *p_user << " requests for dataset " << *p_dataset);

    const auto requests = context::AppContext::get_instance().request_service.get_active_multival_requests(*p_user, *p_dataset);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(requests.size()))
    for (const auto &p_request: requests) {
        std::atomic<bool> request_answered = true;
        if (!p_request->sanity_check()) {
            LOG4_ERROR("Request " << *p_request << " incorrect!");
            request_answered.store(false, std::memory_order_relaxed);
            continue;
        }

        LOG4_DEBUG("Processing request " << *p_request);
        std::set<bpt::ptime> predict_times;
        if (p_request->value_time_start == p_request->value_time_end)
            predict_times.emplace(p_request->value_time_start);
        else for (auto t = p_request->value_time_start; t < p_request->value_time_end; t += p_dataset->get_input_queue()->get_resolution())
                predict_times.emplace(t);

        if (predict_times.empty()) {
            LOG4_ERROR("Times grid empty.");
            request_answered.store(false, std::memory_order_relaxed);
            continue;
        }

        const auto pred_features = prepare_request_features(p_dataset, predict_times);
        if (pred_features.empty()) {
            LOG4_ERROR("Failed preparing prediction features.");
            request_answered.store(false, std::memory_order_relaxed);
            continue;
        }

        // Do actual predictions
        const auto columns = common::from_sql_array(p_request->value_columns);
#pragma omp parallel for num_threads(adj_threads(columns.size()))
        for (const auto &request_column: columns) {
            const auto p_ensemble = p_dataset->get_ensemble(request_column);
            if (!p_ensemble) {
                LOG4_ERROR("Ensemble for request column " << request_column << " not found.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            const auto predicted_decon = EnsembleService::predict_noexcept(p_dataset, p_ensemble, pred_features);
            if (!predicted_decon) {
                LOG4_ERROR("Predicted decon empty.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }

            const auto predicted_recon = APP.decon_queue_service.reconstruct(
                    datamodel::datarow_range(predicted_decon->get_data()),
                    recon_type_e::ADDITIVE);

            if (predicted_recon.empty()) {
                LOG4_ERROR("Empty reconstructed data.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            LOG4_DEBUG("Saving reconstructed p_predictions from " << predicted_recon.begin()->get()->get_value_time()
                            << " until " << predicted_recon.rbegin()->get()->get_value_time());
            std::for_each(std::execution::par_unseq, predicted_recon.begin(), predicted_recon.end(), save_response(request_column, p_request));
        }

        if (request_answered)
            APP.request_service.force_finalize(p_request);
        else
            LOG4_WARN("Failed finalizing request id " << p_request->get_id());
    }

    LOG4_END();
}


} //business
} //svr
