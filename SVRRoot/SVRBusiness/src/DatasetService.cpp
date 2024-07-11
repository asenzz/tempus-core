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
#include "DeconQueueService.hpp"
#include "EnsembleService.hpp"
#include "IQScalingFactorService.hpp"
#include "InputQueueService.hpp"
#include "common/logging.hpp"
#include "common/parallelism.hpp"
#include "common/compatibility.hpp"
#include "model/DataRow.hpp"
#include "appcontext.hpp"
#include "DAO/DatasetDAO.hpp"
#include "model/User.hpp"
#include "common/rtp_thread_pool.hpp"
#include "SVRParametersService.hpp"
#include "ModelService.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

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

bool DatasetService::exists(const std::string &user_name, const std::string &dataset_name)
{
    return dataset_dao.exists(user_name, dataset_name);
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
    t_omp_lock DatasetService_load;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_aux_input_queues().size()))
    for (size_t i = 0; i < p_dataset->get_aux_input_queues().size(); ++i) {
        auto &iq = *p_dataset->get_aux_input_queue(i);
        APP.input_queue_service.load(iq);
        DatasetService_load.set();
        if (aux_res.is_special())
            aux_res = iq.get_resolution();
        else if (aux_res != iq.get_resolution())
            LOG4_THROW("Auxiliary input queue " << iq.get_table_name() << " resolution " << iq.get_resolution() << " does not equal " << aux_res);
        DatasetService_load.unset();
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
        : p_dataset(dataset), users(std::forward<std::deque<User_ptr>>(users))
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
            [&active_datasets](const auto &processed_pair) { return !is_dataset_element_of(processed_pair.p_dataset, active_datasets); });
    if (new_end != processed_user_dataset_pairs.end()) {
        LOG4_DEBUG("Erasing from " << new_end->p_dataset->dataset_name_ << " until end.");
        processed_user_dataset_pairs.erase(new_end, processed_user_dataset_pairs.end());
    }

#pragma omp parallel for num_threads(adj_threads(processed_user_dataset_pairs.size())) schedule(static, 1)
    for (auto &pudp: processed_user_dataset_pairs) pudp.users.clear();

    t_omp_lock emplace_dataset_t;
    t_omp_lock emplace_user_t;
#pragma omp parallel for num_threads(adj_threads(active_datasets.size())) schedule(static, 1)
    for (auto &dataset_pair: active_datasets) {
        auto iter = std::find_if(std::execution::par_unseq, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
                                 [&dataset_pair](const auto &other) {
                                     if (!dataset_pair.second || !other.p_dataset) LOG4_THROW("Null ptr value passed");
                                     return dataset_pair.second->get_id() == other.p_dataset->get_id();
                                 });

        if (iter == processed_user_dataset_pairs.end()) {
            load(dataset_pair.second);
            emplace_dataset_t.set();
            processed_user_dataset_pairs.emplace_back(dataset_pair.second, std::deque{APP.user_service.get_user_by_user_name(dataset_pair.first)});
            emplace_dataset_t.unset();
        } else {
            emplace_user_t.set();
            iter->users.emplace_back(APP.user_service.get_user_by_user_name(dataset_pair.first));
            emplace_user_t.unset();
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

                  if (lhs.p_dataset->get_priority() > rhs.p_dataset->get_priority())
                      return true;
                  if (lhs.p_dataset->get_priority() < rhs.p_dataset->get_priority())
                      return false;

                  return false;
              });
}

datamodel::t_predict_features
DatasetService::prepare_prediction_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const std::deque<bpt::ptime> &predict_times)
{
    LOG4_BEGIN();

    datamodel::t_predict_features res;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_multistep() * dataset.get_transformation_levels())) ordered collapse(2)
    for (size_t levix = 0; levix < dataset.get_transformation_levels(); levix += 2)
        for (size_t stepix = 0; stepix < dataset.get_multistep(); ++stepix) {
            const auto p_model = ensemble.get_model(levix, stepix);
            auto p_features = ptr<arma::mat>();
            ModelService::prepare_features(
                    *p_features, predict_times, ensemble.get_aux_decon_queues(), *p_model->get_head_params(),
                    dataset.get_max_lookback_time_gap(),
                    dataset.get_aux_input_queues().empty() ? dataset.get_input_queue()->get_resolution() : dataset.get_aux_input_queue()->get_resolution(),
                    dataset.get_input_queue()->get_resolution());
#pragma omp ordered
            res.emplace(std::tuple{levix, stepix}, datamodel::t_level_predict_features{predict_times, p_features});
        }

    LOG4_BEGIN();

    return res;
}


void DatasetService::process(datamodel::Dataset &dataset)
{
    dataset.get_calc_cache().clear();
    InputQueueService::prepare_queues(dataset);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_ensembles().size()))
    for (auto p_ensemble: dataset.get_ensembles())
        PROFILE_EXEC_TIME(EnsembleService::train(dataset, *p_ensemble), "Ensemble " << p_ensemble->get_column_name() << " train");
    LOG4_END();
}

// By specification, a multival request's time period to predict is [start_predict_time, end_predict_time) i.e. right-hand exclusive.
void DatasetService::process_requests(const datamodel::User &user, datamodel::Dataset &dataset)
{
    LOG4_DEBUG("Processing " << user << " requests for dataset " << dataset);

    const auto requests = context::AppContext::get_instance().request_service.get_active_multival_requests(user, dataset);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(requests.size()))
    for (const auto &p_request: requests) {
        std::atomic<bool> request_answered = true;
        if (!p_request->sanity_check()) {
            LOG4_ERROR("Request " << *p_request << " incorrect!");
            request_answered.store(false, std::memory_order_relaxed);
            continue;
        }

        LOG4_DEBUG("Processing request " << *p_request);
        std::deque<bpt::ptime> predict_times;
        if (p_request->value_time_start == p_request->value_time_end)
            predict_times.emplace_back(p_request->value_time_start);
        else
            for (auto t = p_request->value_time_start; t < p_request->value_time_end; t += dataset.get_input_queue()->get_resolution())
                predict_times.emplace_back(t);

        if (predict_times.empty()) {
            LOG4_ERROR("Times grid empty.");
            request_answered.store(false, std::memory_order_relaxed);
            continue;
        }

        // Do actual predictions
        const auto columns = p_request->get_value_columns();
#pragma omp parallel for num_threads(adj_threads(columns.size())) schedule(static, 1)
        for (const auto &request_column: columns) {
            const auto p_ensemble = dataset.get_ensemble(request_column);
            if (!p_ensemble) {
                LOG4_ERROR("Ensemble for request column " << request_column << " not found.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }

            const auto pred_features = prepare_prediction_data(dataset, *p_ensemble, predict_times);
            if (pred_features.empty()) {
                LOG4_ERROR("Failed preparing prediction features.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }

            const auto predicted_decon = EnsembleService::predict_noexcept(dataset, *p_ensemble, pred_features);
            if (!predicted_decon) {
                LOG4_ERROR("Predicted decon empty.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            const auto unscaler = IQScalingFactorService::get_unscaler(
                    dataset, p_ensemble->get_label_aux_decon()->get_input_queue_table_name(), p_ensemble->get_label_aux_decon()->get_input_queue_column_name());
            const auto predicted_recon = APP.decon_queue_service.reconstruct(datamodel::datarow_range(predicted_decon->get_data()), recon_type_e::ADDITIVE, unscaler);

            if (predicted_recon.empty()) {
                LOG4_ERROR("Empty reconstructed data.");
                request_answered.store(false, std::memory_order_relaxed);
                continue;
            }
            LOG4_DEBUG("Saving reconstructed p_predictions from " << (**predicted_recon.cbegin()).get_value_time()
                                                                  << " until " << (**predicted_recon.crbegin()).get_value_time());
            std::for_each(std::execution::par_unseq, predicted_recon.cbegin(), predicted_recon.cend(),
                          [&request_column, &p_request](const auto &p_result_row) {
                              if (p_result_row->get_value_time() < p_request->value_time_start or p_result_row->get_value_time() > p_request->value_time_end) {
                                  LOG4_DEBUG("Skipping save response " << *p_result_row);
                                  return;
                              }
                              const auto p_response = ptr<datamodel::MultivalResponse>(
                                      0, p_request->get_id(), p_result_row->get_value_time(), request_column, p_result_row->get_value(0));
                              LOG4_TRACE("Saving " << *p_response);
                              APP.request_service.save(p_response); // Requests get marked processed here
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