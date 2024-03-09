#include <boost/date_time/posix_time/ptime.hpp>
#include <armadillo>
#include <cstdlib>
#include <execution>
#include <limits>
#include <iterator>
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>

#include "DQScalingFactorService.hpp"
#include "EnsembleService.hpp"
#include "InputQueueService.hpp"
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


DatasetService::DatasetService(
        dao::DatasetDAO &datasetDao,
        EnsembleService &ensemble_service,
        SVRParametersService &svr_parameters_service) :
        dataset_dao(datasetDao),
        ensemble_service(ensemble_service),
        svr_parameters_service(svr_parameters_service)
{
}

datamodel::Dataset_ptr DatasetService::load(const bigint dataset_id, const bool load_dependencies)
{
    datamodel::Dataset_ptr p_dataset = dataset_dao.get_by_id(dataset_id);
    if (!p_dataset) LOG4_ERROR("Couldn't find dataset " << dataset_id << " in database.");
    else if (load_dependencies) load(p_dataset);
    return p_dataset;
}

void DatasetService::load(datamodel::Dataset_ptr &p_dataset)
{
    APP.input_queue_service.load(*p_dataset->get_input_queue());
    bpt::time_duration aux_res(boost::date_time::not_a_date_time);
    OMP_LOCK(DatasetService_load)
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_aux_input_queues().size()))
    for (size_t i = 0; i < p_dataset->get_aux_input_queues().size(); ++i) {
        auto &iq = *p_dataset->get_aux_input_queue(i);
        APP.input_queue_service.load(iq);
        omp_set_lock(&DatasetService_load);
        if (aux_res.is_special())
            aux_res = iq.get_resolution();
        else if (aux_res != iq.get_resolution())
            LOG4_THROW("Auxiliary input queue resolution " << iq.get_resolution() << " does not equal " << aux_res);
        omp_unset_lock(&DatasetService_load);
    }

    if (!EnsembleService::check(p_dataset->get_ensembles(), p_dataset->get_input_queue()->get_value_columns()))
        EnsembleService::init_ensembles(p_dataset, true);
}


datamodel::Dataset_ptr DatasetService::get_user_dataset(const std::string &user_name, const std::string &dataset_name)
{
    datamodel::Dataset_ptr p_dataset = dataset_dao.get_by_name(user_name, dataset_name);
    if (p_dataset != nullptr) {
        p_dataset->set_input_queue(APP.input_queue_service.get_queue_metadata(p_dataset->get_input_queue()->get_table_name()));
        load(p_dataset);
    }
    return p_dataset;
}

std::deque<datamodel::Dataset_ptr> DatasetService::find_all_user_datasets(const std::string &username)
{
    return dataset_dao.find_all_user_datasets(username);
}


bool DatasetService::save(const datamodel::Dataset_ptr &p_dataset)
{
    if (!p_dataset) {
        LOG4_ERROR("Dataset is null! Aborting.");
        return false;
    }

    dataset_dao.save(p_dataset);

    // save parameters
    auto existing_svr_parameters = svr_parameters_service.get_all_by_dataset_id(p_dataset->get_id());
    svr_parameters_service.remove_by_dataset(p_dataset->get_id());
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (const auto &e: p_dataset->get_ensembles())
#pragma omp parallel for num_threads(adj_threads(e->get_models().size()))
        for (const auto &m: e->get_models()) {
            APP.model_service.save(m);
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    svr_parameters_service.save(p);
        }

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

size_t DatasetService::get_level_count(const bigint dataset_id)
{
    return dataset_dao.get_level_count(dataset_id);
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
bool is_dataset_element_of(const datamodel::Dataset_ptr &p_dataset, const dao::DatasetDAO::UserDatasetPairs &active_datasets)
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
    dao::DatasetDAO::UserDatasetPairs active_datasets = dataset_dao.get_active_datasets();

    // Remove inactive datasets
    const auto new_end = std::remove_if(
            std::execution::par_unseq, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
            [&active_datasets](const auto &processed_pair) { return !is_dataset_element_of(processed_pair.dataset, active_datasets); });
    if (new_end != processed_user_dataset_pairs.end()) {
        LOG4_DEBUG("Erasing from " << new_end->dataset->dataset_name_ << " until end.");
        processed_user_dataset_pairs.erase(new_end, processed_user_dataset_pairs.end());
    }

#pragma omp parallel for num_threads(adj_threads(processed_user_dataset_pairs.size())) schedule(static, 1)
    for (auto &pudp: processed_user_dataset_pairs) pudp.users.clear();

    OMP_LOCK(emplace_dataset_t)
    OMP_LOCK(emplace_user_t)
#pragma omp parallel for num_threads(adj_threads(active_datasets.size())) schedule(static, 1)
    for (auto &dataset_pair: active_datasets) {
        auto iter = std::find_if(std::execution::par_unseq, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
                                 [&dataset_pair](const auto &other) {
                                     if (!dataset_pair.second || !other.dataset) LOG4_THROW("Null ptr value passed");
                                     return dataset_pair.second->get_id() == other.dataset->get_id();
                                 });

        if (iter == processed_user_dataset_pairs.end()) {
            load(dataset_pair.second);
            omp_set_lock(&emplace_dataset_t);
            processed_user_dataset_pairs.emplace_back(dataset_pair.second, std::deque{APP.user_service.get_user_by_user_name(dataset_pair.first)});
            omp_unset_lock(&emplace_dataset_t);
        } else {
            omp_set_lock(&emplace_user_t);
            iter->users.emplace_back(APP.user_service.get_user_by_user_name(dataset_pair.first));
            omp_unset_lock(&emplace_user_t);
        }
    }

    std::sort(std::execution::par_unseq, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
              [](UserDatasetPairs::value_type const &lhs, UserDatasetPairs::value_type const &rhs) {
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
              });
}


bool DatasetService::exists(const std::string &user_name, const std::string &dataset_name)
{
    return dataset_dao.exists(user_name, dataset_name);
}


bool
DatasetService::prepare_training_data(
        datamodel::Dataset &dataset,
        datamodel::Ensemble &ensemble,
        t_training_data &train_data)
{
    LOG4_BEGIN();

    bool features_scaled = false;
    std::deque<bpt::ptime> label_times;
    static arma::mat empty_mat;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_transformation_levels()))
    for (const auto &p_model: ensemble.get_models()) {
        const auto levix = p_model->get_decon_level();
        auto [p_features, p_labels, p_last_knowns] = ModelService::get_training_data(label_times, dataset, ensemble, *p_model);
#pragma omp critical
        {
            APP.dq_scaling_factor_service.scale(dataset, ensemble.get_aux_decon_queues(), *p_model->get_head_params(), features_scaled ? empty_mat : *p_features, *p_labels, *p_last_knowns);
            features_scaled = true;
        }
        train_data.features[levix] = p_features;
        train_data.labels[levix] = p_labels;
        train_data.last_knowns[levix] = p_last_knowns;
    }
    train_data.last_row_time = label_times.back();

    LOG4_END();

    return true;
}


std::unordered_map<size_t, matrix_ptr>
DatasetService::join_features(std::unordered_map<std::string, t_training_data> &train_data, const size_t levct, const std::deque<datamodel::Ensemble_ptr> &ensembles)
{
    auto result = train_data[(**ensembles.begin()).get_column_name()].features;
    train_data[(**ensembles.begin()).get_column_name()].features.clear();
    if (ensembles.size() < 2) goto __bail;
#pragma omp parallel for num_threads(adj_threads(levct))
    for (size_t levix = 0; levix < levct; levix += 2) {
        if (levix == levct / 2) continue;
        for (auto ens_it = std::next(ensembles.begin()); ens_it != ensembles.end(); ++ens_it) { // Non-parallelizabile
            const std::string column_name = (**ens_it).get_column_name();
            result[levix]->insert_cols(result[levix]->n_cols, *train_data.at(column_name).features.at(levix));
            train_data.at(column_name).features.at(levix)->clear();
        }
    }
__bail:
    return result;
}


void DatasetService::process(datamodel::Dataset_ptr &p_dataset)
{
    InputQueueService::prepare_queues(p_dataset);

    std::unordered_map<std::string /* column name */, t_training_data> ensemble_training_data;

    bool train_ret = true;
#pragma omp parallel for reduction(&:train_ret) num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (auto p_ensemble: p_dataset->get_ensembles()) PROFILE_EXEC_TIME(
            train_ret &= prepare_training_data(*p_dataset, *p_ensemble, ensemble_training_data[p_ensemble->get_column_name()]),
            "Prepare data for ensemble " << p_ensemble->get_column_name());
    if (!train_ret) {
        LOG4_ERROR("Prepare data failed, aborting training.");
        return;
    }

    const auto dataset_features = join_features(    // Labels differ but aux decon queues and features are shared among ensembles of the dataset
            ensemble_training_data, p_dataset->get_transformation_levels(), p_dataset->get_ensembles());

    p_dataset->get_calc_cache().clear();
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (auto p_ensemble: p_dataset->get_ensembles())
        EnsembleService::train(*p_ensemble, dataset_features, ensemble_training_data[p_ensemble->get_column_name()]);

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
    OMP_LOCK(insert_times_l)
    /* Determine start and end training time range for every ensemble based on every model parameters. */
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (const auto &e: p_dataset->get_ensembles()) {
        const boost::posix_time::ptime start_at = e->get_models().empty() || e->get_model()->get_last_modeled_value_time() == bpt::min_date_time ?
                                                  APP.input_queue_service.load_nth_last_row(
                                                          p_dataset->get_input_queue(), max_decrement_distance + INTEGRATION_TEST_VALIDATION_WINDOW)->get_value_time() :
                                                  e->get_model()->get_last_modeled_value_time();
        if (start_at == input_queue_newest_value_time)
            LOG4_DEBUG("No new data in input queue to process for ensemble " << e->get_id());
        omp_set_lock(&insert_times_l);
        ensemble_train_start_times.insert(start_at);
        omp_unset_lock(&insert_times_l);
    }
    if (!ensemble_train_start_times.empty()) oldest_start_train_time = *ensemble_train_start_times.begin();
    if (oldest_start_train_time == bpt::min_date_time)
        LOG4_THROW("Some models were not initialized with train, aborting!");
    if (oldest_start_train_time > input_queue_newest_value_time)
        LOG4_THROW("Ensemble oldest start train time " << oldest_start_train_time << " is newer than input queue last value time " << input_queue_newest_value_time);
    LOG4_DEBUG("Input queue time period for training all ensembles from " << oldest_start_train_time << " to " << input_queue_newest_value_time);
    return {oldest_start_train_time, input_queue_newest_value_time - oldest_start_train_time};
}


auto DatasetService::prepare_request_features(const datamodel::Dataset_ptr &p_dataset, const std::set<bpt::ptime> &predict_times)
{
    std::unordered_map<size_t /* level */, std::pair<std::set<bpt::ptime /* feature row times */>, arma::mat /* features matrix */>> res;
    std::atomic<bool> fail = false;
    const auto ensemble_column = p_dataset->get_input_queue()->get_value_columns();
#pragma omp parallel for ordered schedule(static, 1) num_threads(adj_threads(ensemble_column.size())) collapse(2)
    for (size_t e = 0; e < ensemble_column.size(); ++e) {
        for (size_t m = 0; m < p_dataset->get_model_count(); ++m) {
            if (fail.load(std::memory_order_relaxed)) continue;
            const auto p_ensemble = p_dataset->get_ensemble(ensemble_column[e]);
            const auto p_model = p_ensemble->get_model(m);
            arma::mat level_features;
            const auto p_head_params = p_model->get_head_params();
            const auto feat_levels = common::get_adjacent_indexes(
                    p_model->get_decon_level(), p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
            OMP_LOCK(level_features_l)
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(predict_times.size()))
            for (size_t predix = 0; predix < predict_times.size(); ++predix) {
                const auto pred_time = predict_times ^ predix;
                arma::rowvec features;
                try {
                    APP.model_service.get_features_row(p_dataset, p_ensemble, p_model, pred_time, feat_levels, features);
                } catch (const std::exception &ex) {
                    LOG4_ERROR("Failed preparing features for " << pred_time << ", level " << p_model->get_decon_level() << ", column " << p_ensemble->get_column_name()
                                                                << ", " << ex.what());
                    fail.store(true, std::memory_order_relaxed);
                }
                omp_set_lock(&level_features_l);
                if (level_features.empty()) level_features = features;
                else level_features.row(predix) = features;
                omp_unset_lock(&level_features_l);
            }
            if (fail.load(std::memory_order_relaxed)) continue;
            APP.dq_scaling_factor_service.scale_features(
                    *p_dataset, p_ensemble->get_aux_decon_queues(), *p_head_params, level_features, p_dataset->get_dq_scaling_factors(), feat_levels);
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
        else
            for (auto t = p_request->value_time_start; t < p_request->value_time_end; t += p_dataset->get_input_queue()->get_resolution())
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
        const auto columns = p_request->get_value_columns();
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
            const auto unscaler = IQScalingFactorService::get_unscaler(
                    *p_dataset, p_ensemble->get_label_aux_decon()->get_input_queue_table_name(), p_ensemble->get_label_aux_decon()->get_input_queue_column_name());
            const auto predicted_recon = APP.decon_queue_service.reconstruct(
                    datamodel::datarow_range(predicted_decon->get_data()),
                    recon_type_e::ADDITIVE,
                    unscaler);

            if (predicted_recon.empty()) {
                LOG4_ERROR("Empty reconstructed data.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            LOG4_DEBUG("Saving reconstructed p_predictions from " << predicted_recon.begin()->get()->get_value_time()
                                                                  << " until " << predicted_recon.rbegin()->get()->get_value_time());
            std::for_each(std::execution::par_unseq, predicted_recon.begin(), predicted_recon.end(),
                          [&request_column, &p_request] (const auto &p_result_row) {
                              if (p_result_row->get_value_time() < p_request->value_time_start or p_result_row->get_value_time() > p_request->value_time_end) {
                                  LOG4_DEBUG("Skipping save response " << *p_result_row);
                                  return;
                              }
                              const auto p_response = ptr<datamodel::MultivalResponse>(
                                      0, p_request->get_id(), p_result_row->get_value_time(), request_column, p_result_row->get_value(0));
                              LOG4_TRACE("Saving " << *p_response);
                              context::AppContext::get_instance().request_service.save(p_response); // Requests get marked processed here
                          });
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
