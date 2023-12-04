#include <boost/date_time/posix_time/ptime.hpp>
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <algorithm>
#include <vector>

#include "DQScalingFactorService.hpp"
#include "EnsembleService.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "common/defines.h"
#include "model/DataRow.hpp"
#include "onlinesvr.hpp"
#include "util/ValidationUtils.hpp"
#include "appcontext.hpp"

#include "DAO/DatasetDAO.hpp"
#include "model/User.hpp"

#include "common/rtp_thread_pool.hpp"
#include "common/thread_pool.hpp"

#include "SVRParametersService.hpp"
#include "spectral_transform.hpp"


using namespace svr::common;
using namespace svr::datamodel;
using namespace svr::context;


namespace svr {
namespace business {


namespace {


void
load_dataset(SVRParametersService &svr_parameters_service, EnsembleService &ensemble_service, Dataset_ptr &p_dataset)
{
    if (!p_dataset) LOG4_THROW("Dataset is missing.");
    const bigint id = p_dataset->get_id();
    const size_t lev_ct = p_dataset->get_transformation_levels();
    LOG4_DEBUG("Loading dataset with id " << id << " from database, levels count " << lev_ct);
    p_dataset->set_ensembles(ensemble_service.get_all_by_dataset_id(id));
    auto dataset_svr_parameters = svr_parameters_service.get_all_by_dataset_id(id);
    /* TODO Implement loading/saving of models
      auto dataset_models = model_service.get_all_by_dataset_id(id);
      dataset.set_models(dataset_models);
     */

    for (const auto &p_ensemble: p_dataset->get_ensembles()) {
        auto it_ensemble_parameters = dataset_svr_parameters.find(p_ensemble->get_key_pair());
        while (it_ensemble_parameters == dataset_svr_parameters.end()) {
            auto res = dataset_svr_parameters.emplace(p_ensemble->get_key_pair(), std::vector<SVRParameters_ptr>(lev_ct));
            if (!res.second) LOG4_THROW("Creating SVR parameter set for " << p_ensemble->to_string() << " failed.");
            else it_ensemble_parameters = res.first;
        }
        if (it_ensemble_parameters->second.size() != lev_ct) {
            LOG4_WARN("Corrupt number of parameters " << dataset_svr_parameters.size() << " should be " << lev_ct);
            if (it_ensemble_parameters->second.size() > lev_ct)
                it_ensemble_parameters->second.erase(std::prev(it_ensemble_parameters->second.end(), it_ensemble_parameters->second.size() - lev_ct), it_ensemble_parameters->second.end());
            else if (it_ensemble_parameters->second.size() < lev_ct)
                it_ensemble_parameters->second.resize(lev_ct);
        }
#pragma omp parallel for
        for (size_t level_ix = 0; level_ix < lev_ct; level_ix += 2) {
            if (level_ix == lev_ct / 2 || it_ensemble_parameters->second[level_ix]) continue;
            LOG4_DEBUG("Creating default parameters for " << id << ", " << p_ensemble->get_decon_queue()->get_input_queue_table_name() << ", " << p_ensemble->get_decon_queue()->get_input_queue_column_name() << ", " << level_ix);
            it_ensemble_parameters->second[level_ix] = std::make_shared<SVRParameters>(
                0, id, p_ensemble->get_decon_queue()->get_input_queue_table_name(), p_ensemble->get_decon_queue()->get_input_queue_column_name(), level_ix, 0, 0);
        }

        LOG4_DEBUG("Added parameters " << it_ensemble_parameters->second.size() << " for ensemble " << p_ensemble->get_id());
    }
    p_dataset->set_ensemble_svr_parameters(dataset_svr_parameters);
}

}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wembedded-directive"

bool DatasetService::check_ensembles_svr_parameters(const Dataset_ptr &p_dataset)
{
    std::vector<Ensemble_ptr> &ensembles = p_dataset->get_ensembles();
    return !(ensembles.size() != 0 && p_dataset->get_ensemble_svr_parameters().size() != ensembles.size());
}


Dataset_ptr DatasetService::get(const bigint dataset_id, const bool load)
{
    Dataset_ptr dataset = dataset_dao.get_by_id(dataset_id);
    if (load) load_dataset(svr_parameters_service, ensemble_service, dataset);
    return dataset;
}


std::vector<Dataset_ptr> DatasetService::find_all_user_datasets(std::string username)
{
    return dataset_dao.find_all_user_datasets(username);
}


namespace {


bool
svr_params_equal(std::vector<SVRParameters_ptr> lhs, std::vector<SVRParameters_ptr> rhs)
{
    if (lhs.size() != rhs.size())
        return false;

    for (auto il = lhs.begin(), ir = rhs.begin(); il != lhs.end(); ++il, ++ir)
        if ((*il).get() != (*ir).get())
            return false;

    return true;
}


bool
svr_params_equal(svr::datamodel::ensemble_svr_parameters_t const &lhs, svr::datamodel::ensemble_svr_parameters_t const &rhs)
{
    if (lhs.size() != rhs.size()) return false;

    using svrp = svr::datamodel::ensemble_svr_parameters_t;
    for (svrp::const_iterator i1 = lhs.begin(), i2 = rhs.begin(); i1 != lhs.end(); ++i1, ++i2) {
        if (i1->first != i2->first)
            return false;
        if (!svr_params_equal(i1->second, i2->second))
            return false;
    }

    return true;
}

}

bool DatasetService::save(Dataset_ptr &p_dataset)
{
    if (!p_dataset) {
        LOG4_ERROR("Dataset is null! Aborting.");
        return false;
    }

    dataset_dao.save(p_dataset);

    auto existing_svr_parameters = svr_parameters_service.get_all_by_dataset_id(p_dataset->get_id())
    , new_svr_parameters = p_dataset->get_ensemble_svr_parameters();

    if (!svr_params_equal(existing_svr_parameters, new_svr_parameters)) {
        svr_parameters_service.remove_by_dataset(p_dataset->get_id());

        for (auto &pair_vec_svr_parameters: new_svr_parameters)
            for (auto &svr_parameters: pair_vec_svr_parameters.second)
                svr_parameters_service.save(svr_parameters);
    }

    // save ensembles
    if (!p_dataset->get_ensembles().empty()) {
        ensemble_service.remove_by_dataset_id(p_dataset->get_id());
        // check
        if (!check_ensembles_svr_parameters(p_dataset)) {
            LOG4_ERROR("SVRParameters in ensembles and in dataset are not equal");
        }
        if (!p_dataset->get_ensembles().empty()
            && !ensemble_service.save_ensembles(p_dataset->get_ensembles(), true))
            return false;
    }
    return true;
}


bool DatasetService::exists(Dataset_ptr const &dataset)
{
    reject_nullptr(dataset);
    return dataset_dao.exists(dataset->get_id());
}


bool DatasetService::exists(int dataset_id)
{
    return dataset_dao.exists(dataset_id);
}


int DatasetService::remove(Dataset_ptr const &dataset)
{
    reject_nullptr(dataset);
    svr_parameters_service.remove_by_dataset(dataset->get_id());
    for (Ensemble_ptr ensemble: dataset->get_ensembles()) {
        ensemble_service.remove(ensemble);
    }
    return dataset_dao.remove(dataset);
}

int DatasetService::remove(const SVRParameters_ptr &svr_parameter)
{
    return svr_parameters_service.remove(svr_parameter);
}


bool DatasetService::link_user_to_dataset(User_ptr const &user, Dataset_ptr const &dataset)
{
    return dataset_dao.link_user_to_dataset(user->get_user_name(), dataset);
}


bool DatasetService::unlink_user_from_dataset(User_ptr const &user, Dataset_ptr const &dataset)
{
    return dataset_dao.unlink_user_from_dataset(user->get_user_name(), dataset);
}

DatasetService::DatasetUsers::DatasetUsers(Dataset_ptr const &dataset, std::vector<User_ptr> &&users)
        : dataset(dataset), users(std::forward<std::vector<User_ptr>>(users))
{}

namespace {
bool is_dataset_element_of(const Dataset_ptr &p_dataset, const svr::dao::DatasetDAO::UserDatasetPairs &active_datasets)
{
    /* TODO Implement 'is functionally equivalent' Dataset class operator
     * in order to avoid removing of dataset when only runtime modifiable parts have changed
     */
    for (auto &dataset_users: active_datasets)
        if (*p_dataset == *dataset_users.second)
            return true;
    LOG4_DEBUG("Dataset " << p_dataset->get_id() << " is not element of active datasets.");
    return false;
}
}

// TODO Review and optimize
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
            load_dataset(svr_parameters_service, ensemble_service, dataset_pair.second);
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


Dataset_ptr DatasetService::get_user_dataset(const std::string &user_name, const std::string &dataset_name)
{
    Dataset_ptr p_dataset = dataset_dao.get_by_name(user_name, dataset_name);
    if (p_dataset != nullptr) {
        p_dataset->set_input_queue(AppContext::get_instance().input_queue_service.get_queue_metadata(
                p_dataset->get_input_queue()->get_table_name()));
        load_dataset(svr_parameters_service, ensemble_service, p_dataset);
    }
    return p_dataset;
}


bool DatasetService::prepare_dataset(Dataset_ptr &p_dataset)
{
    LOG4_DEBUG("Processing dataset " << p_dataset->get_id() << " number of ensembles " << p_dataset->get_ensembles().size());
    if (p_dataset->get_ensembles().empty()) LOG4_THROW("No ensembles for dataset " << p_dataset->get_id());

    svr::business::InputQueueService::prepare_queues(p_dataset);

    return true;
}


bool
DatasetService::prepare_models(Ensemble_ptr &p_ensemble, std::vector<SVRParameters_ptr> &ensemble_params, const size_t lev_ct)
{
    if (p_ensemble->get_models().size() == lev_ct) {
        LOG4_DEBUG("Models already initialized.");
        return true;
    }
    LOG4_DEBUG("Ensemble " << p_ensemble->get_decon_queue()->get_table_name() << " models are empty. Initializing " << lev_ct << " models with default values.");
    if (ensemble_params.size() != lev_ct) {
        LOG4_ERROR("Params size " << ensemble_params.size() << " does not match " << lev_ct);
        return false;
    }

    std::vector<Model_ptr> models(lev_ct);
    __tbb_spfor(lev_ix, 0, lev_ct, 2,
        if (lev_ix == lev_ct / 2) continue;
        if (!ensemble_params[lev_ix]) ensemble_params[lev_ix] = std::make_shared<SVRParameters>(
                0, p_ensemble->get_dataset_id(),
                p_ensemble->get_decon_queue()->get_input_queue_table_name(),
                p_ensemble->get_decon_queue()->get_input_queue_column_name(),
                lev_ix, 0, 0);
        models[lev_ix] = std::make_shared<Model>(
             0, p_ensemble->get_id(), lev_ix,
             get_adjacent_indexes(
                     lev_ix,
                     ensemble_params[lev_ix]->get_svr_adjacent_levels_ratio(),
                     lev_ct),
             nullptr, bpt::min_date_time, bpt::min_date_time)
    )
    p_ensemble->set_models(models);
    return true;
}


// TODO Optimize
bool
DatasetService::prepare_data(
        Dataset_ptr &p_dataset,
        Ensemble_ptr &p_ensemble,
        std::vector<SVRParameters_ptr> &ensemble_params,
        std::vector<std::vector<matrix_ptr>> &features,
        std::vector<std::vector<matrix_ptr>> &labels,
        std::vector<std::vector<bpt::ptime>> &last_row_time,
        const size_t lev_ct,
        const size_t ens_ix)
{
    LOG4_BEGIN();

    bool train_ret = true;
    if (p_ensemble->get_aux_decon_queues().size() > 1) LOG4_ERROR("More than one aux decon queue is not supported!");
    const auto p_decon = p_ensemble->get_decon_queue();
    const auto p_aux_decon = p_ensemble->get_aux_decon_queue(0);
#ifdef CACHED_FEATURE_ITER
    APP.model_service.aux_decon_hint = p_aux_decon->get_data().begin(); // lower_bound(p_aux_decon->get_data(), p_decon->get_data()[std::max<size_t>(0, p_decon->get_data().size() - p_dataset->get_max_lag_count() - p_dataset->get_max_decrement() - EMO_TUNE_VALIDATION_WINDOW - MANIFOLD_TEST_VALIDATION_WINDOW - DATA_LUFTA)]->get_value_time());
#endif
    const auto resolution_factor = double(p_dataset->get_input_queue()->get_resolution().total_microseconds()) / double(p_dataset->get_aux_input_queue(0)->get_resolution().total_microseconds());
    std::vector<bpt::ptime> label_times;
    predictions_t tune_predictions(lev_ct);
    std::vector<std::vector<matrix_ptr>> last_knowns(p_dataset->get_ensembles().size(), std::vector<matrix_ptr>(lev_ct));
    const size_t half_levct = lev_ct / 2;
    std::vector<double> scale_label(lev_ct), dc_offset(lev_ct);
    __tbb_spfor(lev_ix, 0, lev_ct, 2,
        if (lev_ix == half_levct) continue;
        const auto p_model = p_ensemble->get_model(lev_ix);
        auto p_params = ensemble_params[lev_ix];
        features[ens_ix][lev_ix] = std::make_shared<arma::mat>();
        labels[ens_ix][lev_ix] = std::make_shared<arma::mat>();
        last_knowns[ens_ix][lev_ix] = std::make_shared<arma::mat>();
        train_ret = APP.model_service.get_training_data(
               *features[ens_ix][lev_ix], *labels[ens_ix][lev_ix], *last_knowns[ens_ix][lev_ix], label_times,
               {svr::business::EnsembleService::get_start(
                           p_decon->get_data(),
                           p_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE,
                           0,
                           p_model->get_last_modeled_value_time(),
                           p_dataset->get_input_queue()->get_resolution()),
                    p_decon->get_data().end(),
                    p_decon->get_data()},
               p_aux_decon->get_data(),
               p_params->get_lag_count(),
               p_model->get_learning_levels(),
               p_dataset->get_max_lookback_time_gap(),
               lev_ix,
               resolution_factor,
               p_model->get_last_modeled_value_time(),
               p_dataset->get_input_queue()->get_resolution());
        last_row_time[ens_ix][lev_ix] = p_decon->get_data().back()->get_value_time();
        APP.dq_scaling_factor_service.scale(p_dataset, p_aux_decon, p_params, p_model->get_learning_levels(), *features[ens_ix][lev_ix], *labels[ens_ix][lev_ix], *last_knowns[ens_ix][lev_ix]);
        if (!p_params->get_svr_kernel_param())
            PROFILE_EXEC_TIME(OnlineMIMOSVR::tune_kernel_params(
                    tune_predictions[p_params->get_decon_level()], p_params, *features[ens_ix][lev_ix], *labels[ens_ix][lev_ix],
                    *last_knowns[ens_ix][lev_ix]), "Tune kernel params for model " << p_params->get_decon_level());
        scale_label[lev_ix] = p_dataset->get_dq_scaling_factor_labels(p_aux_decon->get_input_queue_table_name(), p_aux_decon->get_input_queue_column_name(), lev_ix);
        dc_offset[lev_ix] = lev_ix ? 0 : p_dataset->get_dq_scaling_factor_labels(p_aux_decon->get_input_queue_table_name(), p_aux_decon->get_input_queue_column_name(), DC_DQ_SCALING_FACTOR);
    )

    PROFILE_EXEC_TIME(recombine_params(tune_predictions, ensemble_params, half_levct, scale_label, dc_offset), "Recombine parameters for " << half_levct - 1 << " models.");

#ifdef TRIM_DATA // TODO Buggy check why needed data is cleared!
    const auto leftover_len = p_dataset->get_max_lag_count() + p_dataset->get_residuals_count() + 1;
    const auto leftover_len_aux = leftover_len * resolution_factor;
    if (p_dataset->get_input_queue()->get_data().size() > leftover_len)
        p_dataset->get_input_queue()->get_data().erase(p_dataset->get_input_queue()->get_data().begin(), p_dataset->get_input_queue()->get_data().end() - leftover_len);
    if (p_decon->get_data().size() > leftover_len)
        p_decon->get_data().erase(p_decon->get_data().begin(), p_decon->get_data().end() - leftover_len);
    p_dataset->get_input_queue()->get_data().shrink_to_fit();
    p_decon->get_data().shrink_to_fit();
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

    return train_ret;
}


void DatasetService::join_features(std::vector<std::vector<matrix_ptr>> &features, const size_t lev_ct, const size_t ens_ct)
{
#pragma omp parallel for
    for (size_t lev_ix = 0; lev_ix < lev_ct; lev_ix += 2) {
        if (lev_ix == lev_ct / 2) continue;
        for (size_t ens_ix = 1; ens_ix < ens_ct; ++ens_ix) { // Not parallelizable // Features past ens_ix 0 become invalid
            features[0][lev_ix]->insert_cols(features[0][lev_ix]->n_cols, *features[ens_ix][lev_ix]);
            features[ens_ix][lev_ix]->clear();
        }
    }
}

void DatasetService::process_dataset(Dataset_ptr &p_dataset)
{
    if (not prepare_dataset(p_dataset)) return;

    const auto lev_ct = p_dataset->get_transformation_levels();
#ifdef NO_ONLINE_LEARN
    if (p_dataset->get_ensemble(0)->get_models().size() == lev_ct) {
        for (auto &p_model: p_dataset->get_ensemble(0)->get_models()) {
            if (!p_model) continue;
            const auto new_last_modeled_time = p_dataset->get_ensemble()->get_decon_queue()->get_data().back()->get_value_time();
            if (p_model->get_last_modeled_value_time() >= new_last_modeled_time) continue;
            p_model->set_last_modeled_value_time(new_last_modeled_time);
            p_model->set_last_modified(bpt::second_clock::local_time());
        }
        return;
    }
#endif

    const auto ens_ct = p_dataset->get_ensembles().size();
    std::vector<std::vector<bpt::ptime>> last_row_time(ens_ct);
    std::vector<std::vector<matrix_ptr>> features(ens_ct), labels(ens_ct);
    __omp_pfor_i(0, ens_ct, features[i].resize(lev_ct); labels[i].resize(lev_ct); last_row_time[i].resize(lev_ct))

    bool train_ret = true;
    std::mutex mx;
    __omp_pfor(ens_ix, 0, ens_ct,
        auto p_ensemble = p_dataset->get_ensemble(ens_ix);
        std::vector<SVRParameters_ptr> ensemble_params = p_dataset->get_ensemble_svr_parameters()[p_ensemble->get_key_pair()];
        bool train_ret_ens = true;
        PROFILE_EXEC_TIME(train_ret_ens &= prepare_models(p_ensemble, ensemble_params, lev_ct), "Prepare models");
#ifdef MANIFOLD_TEST
        PROFILE_EXEC_TIME(process_dataset_test_tune(p_dataset, p_ensemble, ensemble_params, features, labels, last_row_time, lev_ct, ens_ix), "Prepare data test and tune");
#else
        PROFILE_EXEC_TIME(train_ret_ens &= prepare_data(p_dataset, p_ensemble, ensemble_params, features, labels, last_row_time, lev_ct, ens_ix), "Prepare data");
#endif
        std::scoped_lock l(mx);
        train_ret &= train_ret_ens;
    )
#ifdef MANIFOLD_TEST
    exit(0);
#endif
    if (not train_ret) {
        LOG4_ERROR("Aborting training.");
        return;
    }

    join_features(features, ens_ct, lev_ct);
    
    __tbb_spfor(lev_ix, 0, lev_ct, 2,
        if (lev_ix == lev_ct / 2) continue;
        __pxt_pfor(ens_ix, 0, ens_ct,
            const auto p_ensemble = p_dataset->get_ensemble(ens_ix);
            auto ensemble_params = p_dataset->get_ensemble_svr_parameters()[p_ensemble->get_key_pair()];
            ModelService::train(ensemble_params[lev_ix], p_ensemble->get_model(lev_ix), features[0][lev_ix], labels[ens_ix][lev_ix], last_row_time[ens_ix][lev_ix]);
        )
    )

    LOG4_END();
}

// Return value contains the needed data without the decomposition tail length
boost::posix_time::time_period
DatasetService::get_training_range(const Dataset_ptr &p_dataset)
{
    LOG4_BEGIN();

    if (!p_dataset->get_input_queue()) LOG4_THROW("Input queue not initialized!");
    const auto max_decrement_distance = p_dataset->get_max_decrement();
    const auto newest_row = APP.input_queue_service.find_newest_record(p_dataset->get_input_queue());
    if (not newest_row) LOG4_THROW("Could not find newest record for input queue " << p_dataset->get_input_queue()->get_table_name());
    const auto input_queue_newest_value_time = newest_row->get_value_time();
    boost::posix_time::ptime oldest_start_train_time = boost::posix_time::min_date_time;
    std::set<boost::posix_time::ptime> ensemble_train_start_times;
    std::mutex mx;
    /* Determine start and end training time range for every ensemble based on every model parameters. */
    __omp_pfor_i(0, p_dataset->get_ensembles().size(),
         const auto p_ensemble = p_dataset->get_ensemble(i);
         boost::posix_time::ptime start_train_time;
         if (p_ensemble->get_models().empty() or p_ensemble->get_model(0)->get_last_modeled_value_time() == bpt::min_date_time) {
             start_train_time = APP.input_queue_service.get_nth_last_row(p_dataset->get_input_queue(), max_decrement_distance + MANIFOLD_TEST_VALIDATION_WINDOW)->get_value_time();
         } else {
             start_train_time = p_ensemble->get_model(0)->get_last_modeled_value_time();
             if (start_train_time == input_queue_newest_value_time)
                 LOG4_DEBUG("No new data in input queue to process for ensemble " << p_ensemble->get_id());
         }
         std::scoped_lock l(mx);
         ensemble_train_start_times.insert(start_train_time);
    )
    if (not ensemble_train_start_times.empty()) oldest_start_train_time = *ensemble_train_start_times.begin();
    if (oldest_start_train_time == bpt::min_date_time)
        LOG4_THROW("Some models were not initialized with train, aborting!");
    if (oldest_start_train_time > input_queue_newest_value_time)
        LOG4_THROW("Ensemble oldest start train time " << oldest_start_train_time << " is newer than input queue last value time " << input_queue_newest_value_time);
    LOG4_DEBUG("Input queue time period for training all ensembles from " << oldest_start_train_time << " to " << input_queue_newest_value_time);
    return {oldest_start_train_time, input_queue_newest_value_time - oldest_start_train_time};
}

#pragma GCC diagnostic pop


} //business
} //svr
