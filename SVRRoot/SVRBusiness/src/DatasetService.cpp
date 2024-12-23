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
    auto p_dataset = dataset_dao.get_by_id(dataset_id);
    if (!p_dataset) LOG4_ERROR("Couldn't find dataset " << dataset_id << " in database.");
    else if (load_dependencies) load(p_dataset);
    return p_dataset;
}

void DatasetService::load(datamodel::Dataset_ptr &p_dataset)
{
    APP.input_queue_service.load(*p_dataset->get_input_queue());
    bpt::time_duration aux_res(boost::date_time::not_a_date_time);
    t_omp_lock load_l;
    OMP_FOR(p_dataset->get_aux_input_queues().size())
    for (const auto &p_input_queue: p_dataset->get_aux_input_queues()) {
        APP.input_queue_service.load(*p_input_queue);
        load_l.set();
        if (aux_res.is_special())
            aux_res = p_input_queue->get_resolution();
        else if (aux_res != p_input_queue->get_resolution())
            LOG4_THROW("Auxiliary input queue " << p_input_queue->get_table_name() << " resolution " << p_input_queue->get_resolution() << " does not equal " << aux_res);
        load_l.unset();
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

    OMP_FOR_(p_dataset->get_ensembles().size(),)
    for (const auto &e: p_dataset->get_ensembles()) {
        OMP_FOR_(e->get_models().size(),)
        for (const auto &m: e->get_models()) {
            APP.model_service.save(m);
            OMP_FOR(m->get_gradients().size())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    svr_parameters_service.save(p);
        }
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
    REJECT_NULLPTR(dataset);
    return dataset_dao.exists(dataset->get_id());
}


bool DatasetService::exists(int dataset_id)
{
    return dataset_dao.exists(dataset_id);
}


int DatasetService::remove(datamodel::Dataset_ptr const &dataset)
{
    REJECT_NULLPTR(dataset);
    svr_parameters_service.remove_by_dataset(dataset->get_id());
    OMP_FOR(dataset->get_ensembles().size())
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
    return std::find_if(C_default_exec_policy, active_datasets.cbegin(), active_datasets.cend(), [&p_dataset](const auto &dataset_users) {
        return p_dataset->get_id() == dataset_users.second->get_id();
    }) != active_datasets.cend();
}

}


void DatasetService::update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs)
{
    dao::DatasetDAO::UserDatasetPairs active_datasets = dataset_dao.get_active_datasets();

    // Remove inactive datasets
    const auto new_end = std::remove_if(
            C_default_exec_policy, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
            [&active_datasets](const auto &processed_pair) { return !is_dataset_element_of(processed_pair.p_dataset, active_datasets); });
    if (new_end != processed_user_dataset_pairs.cend()) {
        LOG4_DEBUG("Erasing from " << new_end->p_dataset->dataset_name_ << " until end.");
        processed_user_dataset_pairs.erase(new_end, processed_user_dataset_pairs.cend());
    }

    std::for_each(C_default_exec_policy, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
                  [](auto &pudp) { pudp.users.clear(); });

    t_omp_lock emplace_dataset_t;
    t_omp_lock emplace_user_t;
    OMP_FOR(active_datasets.size())
    for (auto &dataset_pair: active_datasets) {
        auto iter = std::find_if(C_default_exec_policy, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
                                 [&dataset_pair](const auto &other) { return dataset_pair.second->get_id() == other.p_dataset->get_id(); });

        if (iter == processed_user_dataset_pairs.cend()) {
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

    std::sort(C_default_exec_policy, processed_user_dataset_pairs.begin(), processed_user_dataset_pairs.end(),
              [](const auto &lhs, const auto &rhs) {
                  auto i_lhs = lhs.users.cbegin(), i_rhs = rhs.users.cbegin();
                  const auto end_lhs = lhs.users.cend(), end_rhs = rhs.users.cend();

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


void DatasetService::process(datamodel::Dataset &dataset)
{
    dataset.get_calc_cache().clear();
    InputQueueService::prepare_queues(dataset);
    std::for_each(C_default_exec_policy, dataset.get_ensembles().begin(), dataset.get_ensembles().end(), [&dataset](auto &p_ensemble) {
        PROFILE_EXEC_TIME(EnsembleService::train(dataset, *p_ensemble), "Ensemble " << p_ensemble->get_column_name() << " train");
    });
    LOG4_END();
}

// By specification, a multival request's time period to predict is [start_predict_time, end_predict_time) i.e. right-hand exclusive.
void DatasetService::process_requests(const datamodel::User &user, datamodel::Dataset &dataset)
{
    LOG4_DEBUG("Processing " << user << " requests for dataset " << dataset);

    const auto requests = context::AppContext::get_instance().request_service.get_active_multival_requests(user, dataset);
    const auto timenow = bpt::second_clock::local_time();
#pragma omp parallel ADJ_THREADS(2 * requests.size())
#pragma omp single
    {
        OMP_TASKLOOP_(requests.size(),)
        for (const auto &p_request: requests) {
            std::atomic<bool> request_answered = true;

#define REQCHK(COND, MSG) if (COND) { LOG4_ERROR(MSG); request_answered.store(false, std::memory_order_relaxed); continue; }

            REQCHK(!p_request->sanity_check(), "Request " << *p_request << " incorrect!")

            LOG4_DEBUG("Processing request " << *p_request);
            data_row_container predict_times;
            if (p_request->value_time_start == p_request->value_time_end)
                predict_times.emplace_back(otr<datamodel::DataRow>(p_request->value_time_start, timenow, 0, 0));
            else
                for (auto t = p_request->value_time_start; t < p_request->value_time_end; t += dataset.get_input_queue()->get_resolution())
                    predict_times.emplace_back(otr<datamodel::DataRow>(t, timenow, 0, 0));

            REQCHK(predict_times.empty(), "Times grid empty.")

            // Do actual predictions
            const auto &columns = p_request->get_value_columns();
            OMP_TASKLOOP_(columns.size(),)
            for (const auto &request_column: columns)
                if (request_answered) {
                    const auto p_ensemble = dataset.get_ensemble(request_column);
                    REQCHK(!p_ensemble, "Ensemble for request column " << request_column << " not found.")

                    const auto predicted_decon = EnsembleService::predict_noexcept(dataset, *p_ensemble, predict_times);
                    REQCHK(!predicted_decon, "Predicted decon empty.")

                    const auto unscaler = IQScalingFactorService::get_unscaler(
                            dataset, p_ensemble->get_label_aux_decon()->get_input_queue_table_name(), request_column);
                    const auto predicted_recon = APP.decon_queue_service.reconstruct(
                            datamodel::datarow_range(predicted_decon->get_data()), recon_type_e::ADDITIVE, unscaler);

                    REQCHK(predicted_recon.empty(), "Empty reconstructed data.")

                    LOG4_DEBUG("Saving reconstructed predictions from " << (**predicted_recon.cbegin()).get_value_time()
                                                                        << " until " << (**predicted_recon.crbegin()).get_value_time());
                    OMP_TASKLOOP_(predicted_recon.size(),)
                    for (const auto &p_result_row: predicted_recon) {
                        if (p_result_row->get_value_time() < p_request->value_time_start or p_result_row->get_value_time() > p_request->value_time_end) {
                            LOG4_DEBUG("Skipping save response " << *p_result_row);
                            continue;
                        }
                        const auto p_response = ptr<datamodel::MultivalResponse>(
                                0, p_request->get_id(), p_result_row->get_value_time(), request_column, p_result_row->get_value(0));
                        LOG4_TRACE("Saving " << *p_response);
                        APP.request_service.save(p_response); // Requests get marked processed here
                    }
                }

            if (request_answered)
                APP.request_service.force_finalize(p_request);
            else
                LOG4_WARN("Failed finalizing request " << *p_request);
        }
    }
    LOG4_END();
}


} //business
} //svr