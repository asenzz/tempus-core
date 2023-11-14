#if 0 // Paramtune is deprecated, models are tuned on the fly
#include "paramtune.h"
#include "appcontext.hpp"
#include "smo_exception.h"

#include <common/rtp_thread_pool.hpp>
#include <common/thread_pool.hpp>
#include "spectral_transform.hpp"

using namespace svr::common;
using namespace svr::datamodel;
using namespace svr::business;
using namespace svr::context;
using namespace bpt;

namespace svr {
namespace paramtune {

// TODO: make a proper component for all these...
static std::atomic<size_t> total_loss_calls(0);
static std::atomic<size_t> current_loss_call_number__(0);

static const bpt::ptime time_first_call(bpt::second_clock::local_time());

double
loss(
        const std::vector<double> &parameters,
        Dataset_ptr &p_original_dataset,
        validation_parameters_t &validation_parameters,
        const score_metric_e score_metric,
        const DeconQueue_ptr &p_decon_queue,
        const std::vector<DeconQueue_ptr> &aux_decon_queues)
{
    svr::common::memory_manager::instance().wait();
    LOG4_BEGIN();

    svr::datamodel::Dataset fresh_dataset(*p_original_dataset);
    Dataset_ptr p_dataset = std::make_shared<svr::datamodel::Dataset>(fresh_dataset);

    ResourceMeasure loss_resource_measure;
    loss_resource_measure.set_start_time();
    ++current_loss_call_number__;
    const size_t loss_call_number = current_loss_call_number__;
    if (total_loss_calls)
        LOG4_INFO("Estimated time left "
                          << (((bpt::second_clock::local_time() - time_first_call) / loss_call_number) *
                              (total_loss_calls - loss_call_number)));


    LOG4_INFO(
            "Calculating loss count " << loss_call_number << " on column " << validation_parameters.column_name_
                                      <<
                                      " level " << validation_parameters.model_number_);

    try {
        // TODO Set parameters to model
    } catch (const std::range_error &ex) {
        LOG4_ERROR("Setting parameters failed: " << ex.what() << " Returning score of " << LOSS_WORSE_SCORE);
        return LOSS_WORSE_SCORE;
    }

    const SVRParameters_ptr p_svr_parameters = p_dataset->get_svr_parameters(
            p_dataset->get_input_queue()->get_table_name(), validation_parameters.column_name_,
            validation_parameters.model_number_);

    double result = 0.;
    size_t score_calls_ct = 0;
    const auto &api = AppContext::get_instance();
    auto training_window = validation_parameters.training_range_;
    auto validation_window = validation_parameters.validation_range_;
    std::vector<std::future<int>> futures;
    // TODO Parallelize loop
    for (size_t ix_pos = 0; ix_pos < validation_parameters.validation_slide_count_; ++ix_pos) {
        // TODO Parallelize slide calls
        //    futures.push_back(svr::common::async_task([ix_pos, p_dataset, &p_svr_parameters, &result, &score_calls_ct, &mx,
        //            &training_window, &validation_window, &validation_parameters, p_decon_queue, &aux_decon_queues, &score_metric] () -> int {

        DataRow::container predicted_data_future;
        DataRow::container predicted_data_comb;
        try {
            PROFILE_EXEC_TIME(
                    predicted_data_future = api.dataset_service.run_dataset_column_model(
                            p_dataset,
                            p_svr_parameters,
                            training_window,
                            validation_window,
                            validation_parameters.model_number_,
                            p_decon_queue,
                            aux_decon_queues),
                    "Run dataset on slide " << ix_pos << " training period " << training_window << " validation period "
                                            << validation_window);
            if (api.app_properties.get_enable_comb_validate()) PROFILE_EXEC_TIME(
                    predicted_data_comb = api.dataset_service.run_dataset_column_model_comb(
                            p_dataset,
                            p_svr_parameters,
                            training_window,
                            validation_parameters.model_number_,
                            p_decon_queue,
                            aux_decon_queues),
                    "Run dataset comb validation on slide " << ix_pos
            )
        } catch (const svr::common::insufficient_data &ex) {
            LOG4_WARN("Failed processing range " << training_window << " " << validation_window << ". " << ex.what());
            goto __slide;
        } catch (const svr::smo::smo_max_iter_error &ex) {
            LOG4_ERROR("Failed creating model. " << ex.what());
            // set result to max available value.
            result = std::numeric_limits<double>::max() /
                     (1. + validation_parameters.validation_slide_count_); // TODO Figure out a penalty score
            goto __slide;
        } catch (const svr::smo::smo_zero_active_error &ex) {
            LOG4_ERROR("Failed creating model. " << ex.what());
            result = std::numeric_limits<double>::max() / (1. + validation_parameters.validation_slide_count_);
            goto __slide;
        } catch (const std::exception &ex) {
            LOG4_ERROR("Failed creating predicted data. " << ex.what());
        }

        {
            //std::unique_lock<std::mutex> wl(mx);
            auto this_validation_score = score(
                    p_decon_queue->get_data(), predicted_data_future,
                    validation_parameters.model_number_, 0, score_metric, false);
            if (api.app_properties.get_enable_comb_validate())
                this_validation_score += score(
                        p_decon_queue->get_data(), predicted_data_comb,
                                                validation_parameters.model_number_, 0, score_metric, false);
            if (std::isnan(this_validation_score)) {
                LOG4_WARN("Failed to validate slide, ignore it and continue with slides.");
                goto __slide;
            } else {
                result += this_validation_score;
                score_calls_ct += api.app_properties.get_enable_comb_validate() ? 2 : 1;
            }
        }

        __slide:
        switch (validation_parameters.sliding_direction_) {
            case sliding_direction_e::forward: {
                //std::unique_lock<std::mutex> wl(mx);
                training_window.shift(validation_parameters.validation_slide_period_sec_);
                validation_window.shift(validation_parameters.validation_slide_period_sec_);
            }
                break;

            case sliding_direction_e::backward: {
                //std::unique_lock<std::mutex> wl(mx);
                training_window.shift(-validation_parameters.validation_slide_period_sec_);
                validation_window.shift(-validation_parameters.validation_slide_period_sec_);
            }
                break;
        }

        //return ix_pos;
//        }));
    }
    /*
    for(auto &f: futures) {
        int future_result = f.get();
        LOG4_DEBUG("loss: idx = " << future_result << " finished");
    }
    */

    result = result / double(score_calls_ct);
    // log results
    LOG4_INFO(
            "Loss call count " << loss_call_number << " time taken "
                               << loss_resource_measure.get_time_duration_human()
                               << " slides count " << validation_parameters.validation_slide_count_ <<
                               " score " << result << " column "
                               << validation_parameters.column_name_
                               <<
                               " level " << validation_parameters.model_number_ << " on parameters "
                               << p_svr_parameters->to_string());

    LOG4_END();

    return result;
}


double score(
        const svr::datamodel::vektor<double> &reference,
        const svr::datamodel::vektor<double> &predicted,
        const score_metric_e score_metric,
        bool print)
{
    LOG4_BEGIN();

    double result = 0.;
    if (reference.size() != predicted.size()) {
        LOG4_WARN("Reference vector length " << reference.size() << " and predicted vector length "
                                             << predicted.size() << " do not match!");
        //result = std::numeric_limits<double>::quiet_NaN();
    }

    off_t ix = 0;
    auto skipped_ct = ix;
    double predicted_sum = 0., reference_sum = 0.;
    while (ix < reference.size() && ix < predicted.size()) {
        if (print)
            LOG4_DEBUG(ix << " reference value " << reference[ix] << " predicted value " << predicted[ix]);
        if (std::isnan(reference[ix]) || std::isnan(predicted[ix])) {
            LOG4_DEBUG("Skipping " << ix);
            ++skipped_ct;
            ++ix;
            if (score_metric == score_metric_e::CRMSE || score_metric == score_metric_e::CMAE)
                break;
            else
                continue;
        }

        predicted_sum += predicted[ix];
        reference_sum += reference[ix];
        /* TODO Replace with ViennaCL vector operations */
        switch (score_metric) {
            case score_metric_e::RMSE:
                result += std::pow(predicted[ix] - reference[ix], 2);
                break;
            case score_metric_e::MAE:
                result += std::abs(predicted[ix] - reference[ix]);
                break;
            case score_metric_e::CRMSE:
                result += std::pow(predicted_sum - reference_sum, 2);
                break;
            case score_metric_e::CMAE:
                result += std::abs(predicted_sum - reference_sum);
                break;
        }

        ++ix;
    }

    if (skipped_ct >
        (score_metric == score_metric_e::CMAE || score_metric == score_metric_e::CRMSE ? 0 : ix / 2)) {
        LOG4_WARN(
                "Missing values " << skipped_ct << " are more than half predicted count " << ix
                                  << ", score is none!");
        return std::numeric_limits<double>::quiet_NaN();
    }
    result /= double(ix - skipped_ct);

    switch (score_metric) {
        case score_metric_e::RMSE:
        case score_metric_e::CRMSE:
            result = sqrt(result);
            break;

        case score_metric_e::MAE:
        case score_metric_e::CMAE:
            break;
    }

    LOG4_DEBUG("Score of " << (ix - skipped_ct) << " validations is " << result);
    return result;
}


static double single_progress_step{0.0};
static double current_progress{0.0};
static ResourceMeasure resource_measure;

double
score(const svr::datamodel::DataRow::container &ethalon, const svr::datamodel::DataRow::container &predicted,
      const size_t ethalon_column_idx, const size_t predicted_column_idx, const score_metric_e score_metric,
      bool print)
{
    LOG4_BEGIN();
    if (predicted.empty()) {
        LOG4_ERROR("Predicted values are empty!");
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (ethalon.empty()) {
        LOG4_ERROR("Reference values are empty!");
        return std::numeric_limits<double>::quiet_NaN();
    }
    double result = 0.;
    double compared_data_ct = 0.;
    double predicted_sum = 0., reference_sum = 0.;
    for (const auto &predicted_row: predicted) {
        const auto valid_time = predicted_row->get_value_time();
        const auto etha_it = find(ethalon, valid_time);
        if (etha_it == ethalon.end()) {
            LOG4_WARN("Skipping reference value for time " << valid_time << " not found.");
            continue;
        }
        const auto ethalon_value = etha_it->get()->get_values()[ethalon_column_idx];
        const auto predicted_value = predicted_row.get()->get_values()[predicted_column_idx];
        if (print)
            LOG4_INFO(
                    "Time " << bpt::to_simple_string(valid_time) << " ethalon value " << ethalon_value
                            << " predicted value " << predicted_value);

        predicted_sum += predicted_value;
        reference_sum += ethalon_value;
        switch (score_metric) {
            case score_metric_e::RMSE:
                result += std::pow(predicted_value - ethalon_value, 2);
                break;
            case score_metric_e::MAE:
                result += std::abs(predicted_value - ethalon_value);
                break;
            case score_metric_e::CRMSE:
                result += std::pow(predicted_sum - reference_sum, 2);
                break;
            case score_metric_e::CMAE:
                result += std::abs(predicted_sum - reference_sum);
                break;
        }
        compared_data_ct += 1.;
    }
    if (compared_data_ct < 0.5 * predicted.size()) {
        LOG4_WARN(
                "Predicted values " << int(compared_data_ct) << " are less than half predicted count " <<
                                    predicted.size() << ", score is none!");
        return std::numeric_limits<double>::quiet_NaN();
    }

    result /= compared_data_ct;
    switch (score_metric) {
        case score_metric_e::RMSE:
        case score_metric_e::CRMSE:
            result = sqrt(result);
            break;
        case score_metric_e::MAE:
        case score_metric_e::CMAE:
            break;
    }

    LOG4_DEBUG("Score of " << int(compared_data_ct) << " validations is " << result);
    return result;
}


/* This method is expecting input queues with at least frame length data size */
void
prepare_decon_data(
        Dataset_ptr &p_dataset,
        const InputQueue_ptr &p_input_queue,
        std::vector<DeconQueue_ptr> &decon_queues,
        bool force_recalculate)
{
    if (p_input_queue->get_data().empty()) LOG4_THROW("No data in input queue!");

    LOG4_DEBUG("Deconstructing data from " << p_input_queue->get_data().begin()->get()->get_value_time() << " until " << p_input_queue->get_data().rbegin()->get()->get_value_time());
    decon_queues = APP.decon_queue_service.deconstruct(p_input_queue, p_dataset);

    LOG4_END();
}


size_t
get_max_lag_count(const Dataset_ptr &p_dataset, const std::vector<Bounds> &all_bounds)
{
    LOG4_BEGIN();

    size_t max_lag_count = 0;
    for (const auto &bounds: all_bounds) {
        if (bounds.max_bounds.get_lag_count() > max_lag_count)
            max_lag_count = bounds.max_bounds.get_lag_count();
        if (bounds.min_bounds.get_lag_count() > max_lag_count)
            max_lag_count = bounds.min_bounds.get_lag_count();
    }

    for (const auto &dataset_parameters_pair: p_dataset->get_ensemble_svr_parameters())
        for (const auto &dataset_parameters: dataset_parameters_pair.second)
            if (dataset_parameters->get_lag_count() > max_lag_count)
                max_lag_count = dataset_parameters->get_lag_count();
    LOG4_DEBUG("Max lag count is " << max_lag_count);

    return max_lag_count;
}


struct tune_column_task : svr::common::rtp::task
{
    Dataset_ptr p_current_dataset;
    const validation_parameters_t local_validation_parameters;
    const svr::optimizer::NM_parameters nm_parameters;
    const svr::optimizer::PSO_parameters pso_parameters;
    const DeconQueue_ptr p_decon_queue;
    const std::vector<DeconQueue_ptr> aux_decon_queues;

    std::pair<std::string, std::pair<double, SVRParameters_ptr>> result;

    tune_column_task(Dataset_ptr p_current_dataset, const validation_parameters_t &local_validation_parameters,
                     const svr::optimizer::NM_parameters &nm_parameters, const svr::optimizer::PSO_parameters &pso_parameters,
                     const DeconQueue_ptr &p_decon_queue, const std::vector<DeconQueue_ptr> &aux_decon_queues)
            : p_current_dataset(p_current_dataset), local_validation_parameters(local_validation_parameters),
              nm_parameters(nm_parameters), pso_parameters(pso_parameters), p_decon_queue(p_decon_queue),
              aux_decon_queues(aux_decon_queues)
    {}

    void run_job() override
    {
        result = tune_column(p_current_dataset, local_validation_parameters, nm_parameters, pso_parameters,
                             p_decon_queue, aux_decon_queues);
    }

    std::pair<std::string, std::pair<double, SVRParameters_ptr>>
    tune_column(const Dataset_ptr &p_current_dataset,
                const validation_parameters_t &local_validation_parameters,
                const svr::optimizer::NM_parameters &nm_parameters, const svr::optimizer::PSO_parameters &pso_parameters,
                const DeconQueue_ptr &p_decon_queue, const std::vector<DeconQueue_ptr> &aux_decon_queues)
    {
        svr::optimizer::loss_callback_t f = std::bind(
                svr::paramtune::loss,
                std::placeholders::_1,
                p_current_dataset,
                local_validation_parameters,
                DEFAULT_SCORE_METRIC,
                p_decon_queue,
                aux_decon_queues);

        auto idx_level = local_validation_parameters.model_number_;
        LOG4_DEBUG("Running PSO..");
        std::vector<std::pair<double, std::vector<double>>> pso_result;/* = pso(f, pso_parameters, local_validation_parameters.bounds_[idx_level], // TODO Broken because bounds are not set
                              idx_level, local_validation_parameters.column_name_); */
        LOG4_DEBUG("Getting best result..");
        /* EMO - should be PSO, if NM not used!*/
        double current_score = pso_result[0].first;
        /* We stop using Nelder-Mead
        for (size_t ix_point = 0; ix_point < local_validation_parameters.best_points_count_; ix_point++) {
            LOG4_DEBUG("Running NM..");
            auto nm_result = nm(f, pso_result[ix_point], nm_parameters);
            if (best_nm_result.first > nm_result.first) best_nm_result = nm_result;
        }
        */
        const auto key = std::make_pair(p_current_dataset->get_input_queue()->get_table_name(),
                                        local_validation_parameters.column_name_);
//      if (current_score >= best_mse[key][idx_level]) return;
        {
            static std::recursive_mutex dataset_mutex;
            std::scoped_lock<std::recursive_mutex> scope_guard(dataset_mutex);
            auto params = p_current_dataset->get_ensemble_svr_parameters();
            auto good_pointer_params = std::make_shared<SVRParameters>(
                    *(p_current_dataset->get_ensemble_svr_parameters()[key].at(idx_level)));
            LOG4_DEBUG("Current parameters " << good_pointer_params->get_svr_C() << " "
                                             << good_pointer_params->get_svr_epsilon() << " "
                                             << good_pointer_params->get_svr_kernel_param() <<
                                             " " << good_pointer_params->get_svr_kernel_param2() << " "
                                             << good_pointer_params->get_svr_decremental_distance() << " "
                                             << good_pointer_params->get_svr_adjacent_levels_ratio());
            LOG4_DEBUG("Best parameters from PSO " << pso_result[0].second[0] << " " << pso_result[0].second[1]
                                                   << " " << pso_result[0].second[2] << " "
                                                   << pso_result[0].second.size() << " "
                                                   << pso_result[0].second[3] << " " << pso_result[0].second[4]
                                                   << " " << pso_result[0].second[5] << " "
                                                   << pso_result[0].second[6]);
            good_pointer_params->set_svr_C(pso_result[0].second[0]);
            good_pointer_params->set_svr_epsilon(pso_result[0].second[1]);
            good_pointer_params->set_svr_kernel_param(pso_result[0].second[2]);
            good_pointer_params->set_svr_kernel_param2(pso_result[0].second[3]);
            good_pointer_params->set_svr_decremental_distance(pso_result[0].second[4]);
            good_pointer_params->set_svr_adjacent_levels_ratio(pso_result[0].second[5]);
            good_pointer_params->set_lag_count(pso_result[0].second[6]);
            params[key][idx_level] = good_pointer_params;
            p_current_dataset->set_ensemble_svr_parameters_deep(params);
        }


        LOG4_INFO(
                "Found new best parameters for column " << local_validation_parameters.column_name_ << " level "
                                                        << idx_level << " with score " <<
                                                        current_score << " parameters " <<
                                                        p_current_dataset->get_ensemble_svr_parameters()[key].at(
                                                                idx_level)->to_options_string(idx_level));

        auto current_params = std::make_shared<SVRParameters>(
                *(p_current_dataset->get_ensemble_svr_parameters()[key].at(idx_level)));

        current_progress += single_progress_step;
        double timeleft = (100. - current_progress) * resource_measure.get_time_duration() / current_progress;
        LOG4_INFO("Current progress over tuning columns " << current_progress <<
                                                          "%, expected tuning time "
                                                          << resource_measure.time_duration_to_human(timeleft));

        return std::make_pair(local_validation_parameters.column_name_,
                              std::make_pair(current_score, current_params));
    }
};

void prepare_datasets_tweaking(
        const Dataset_ptr &p_current_dataset, std::vector<Dataset_ptr> &datasets_for_tweaking)
{
    // Creating vector of datasets: first - original, all another - for
    datasets_for_tweaking.push_back(p_current_dataset);

}

void calculate_scaling_factors(Dataset_ptr p_current_dataset_, bool force_recalculate)
{
    // Scaling factors computation here.
    auto p_input_queue = p_current_dataset_->get_input_queue()->clone_empty();

    p_input_queue->set_data(AppContext::get_instance().input_queue_service.get_latest_queue_data(
            p_input_queue, std::numeric_limits<int>::max()));

    // Calculate and save scaling factors.
    std::vector<DeconQueue_ptr> decon_queues;
    prepare_decon_data(p_current_dataset_, p_input_queue, decon_queues, force_recalculate);
}


DeconQueue_ptr
get_decon_queue_column(
        const std::vector<DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    for (const DeconQueue_ptr &p_decon_queue: decon_queues)
        if (p_decon_queue->get_input_queue_column_name() == input_queue_column_name &&
            p_decon_queue->get_input_queue_table_name() == input_queue_table_name)
            return p_decon_queue;

    return nullptr;
}


struct tune_level_task : svr::common::rtp::task
{
    const Dataset_ptr p_current_dataset;
    const svr::paramtune::validation_parameters_t validation_params;
    const svr::optimizer::NM_parameters &nm_parameters;
    const svr::optimizer::PSO_parameters &pso_parameters;
    const InputQueue_ptr &p_input_queue;
    const std::vector<DeconQueue_ptr> &decon_queues;
    const std::vector<DeconQueue_ptr> &aux_decon_queues;

    std::pair<std::map<std::string, double>, std::map<std::string, SVRParameters_ptr>> result;

    tune_level_task(
            const Dataset_ptr p_current_dataset,
            const svr::paramtune::validation_parameters_t &validation_params,
            const svr::optimizer::NM_parameters &nm_parameters,
            const svr::optimizer::PSO_parameters &pso_parameters,
            const InputQueue_ptr &p_input_queue,
            const std::vector<DeconQueue_ptr> &decon_queues,
            const std::vector<DeconQueue_ptr> &aux_decon_queues
    )
            : p_current_dataset(p_current_dataset), validation_params(validation_params),
              nm_parameters(nm_parameters),
              pso_parameters(pso_parameters), p_input_queue(p_input_queue), decon_queues(decon_queues),
              aux_decon_queues(aux_decon_queues)
    {}

    void run_job() override
    {
        result = tune_level(p_current_dataset, validation_params, nm_parameters, pso_parameters, p_input_queue,
                            decon_queues, aux_decon_queues);
    }


    std::pair<std::map<std::string, double>, std::map<std::string, SVRParameters_ptr>>
    tune_level(
            const Dataset_ptr &p_current_dataset,
            const svr::paramtune::validation_parameters_t &validation_params,
            const svr::optimizer::NM_parameters &nm_parameters,
            const svr::optimizer::PSO_parameters &pso_parameters,
            const InputQueue_ptr &p_input_queue,
            const std::vector<DeconQueue_ptr> &decon_queues,
            const std::vector<DeconQueue_ptr> &aux_decon_queues)
    {
        LOG4_INFO("tuning dataset " << p_current_dataset->get_dataset_name() << " with level "
                                    << validation_params.model_number_);
        size_t level = validation_params.model_number_;

        std::map<std::string, double> best_column_mse;
        std::map<std::string, SVRParameters_ptr> best_column_params;

        std::vector<svr::future<std::shared_ptr<tune_column_task>>> tune_column_tasks;

        for (const std::string &column_name: p_current_dataset->get_input_queue()->get_value_columns()) {
            LOG4_INFO("Tuning level " << level << " column " << column_name);
            const auto & column_todo = PROPS.get_svr_paramtune_column();
            if (not ((column_todo == "ALL") or (column_todo == column_name))){
                LOG4_DEBUG("Skipping column " << column_name << " as requested in config file");
                continue;
            }


            DeconQueue_ptr p_main_decon_queue = get_decon_queue_column(decon_queues, p_input_queue->get_table_name(),
                                                                       column_name);
            if (!p_main_decon_queue) {
                LOG4_ERROR("Could not find decon queue for column " << column_name << " input queue "
                                                                    << p_input_queue->get_table_name()
                                                                    << ". Aborting.");
                continue;
            }

            std::vector<DeconQueue_ptr> this_column_aux_decon_queues(aux_decon_queues);
            if (AppContext::get_instance().app_properties.get_main_columns_aux())
                for (const auto &p_aux_column_decon_queue: decon_queues)
                    if (p_aux_column_decon_queue->get_input_queue_column_name() !=
                        p_main_decon_queue->get_input_queue_column_name())
                        this_column_aux_decon_queues.push_back(p_aux_column_decon_queue);

            validation_parameters_t local_validation_parameters = validation_params;
            local_validation_parameters.column_name_ = column_name;

            for (size_t model_number = 0;
                 model_number < local_validation_parameters.bounds_.size(); ++model_number) {
                auto &a = local_validation_parameters.bounds_[model_number];
                LOG4_DEBUG("Model " << model_number << " tuning "
                                    << " C " << a.is_tuned.svr_C
                                    << " eps " << a.is_tuned.svr_epsilon
                                    << " g " << a.is_tuned.svr_kernel_param
                                    << " g2 " << a.is_tuned.svr_kernel_param2
                                    << " decr " << a.is_tuned.svr_decremental_distance
                                    << " adj " << a.is_tuned.svr_adjacent_levels_ratio
                                    << " lag " << a.is_tuned.lag_count
                                    << " decrement values " << a.min_bounds.get_svr_decremental_distance()
                                    << " - " << a.max_bounds.get_svr_decremental_distance()
                );
            }

            auto tsk = std::make_shared<tune_column_task>(
                    p_current_dataset, local_validation_parameters, nm_parameters,
                    pso_parameters, p_main_decon_queue, this_column_aux_decon_queues);

            tune_column_tasks.push_back(svr::async([tsk]()
                                                   {
                                                       tsk->run(nullptr);
                                                       return tsk;
                                                   }));
            LOG4_DEBUG("Done.");

            auto tsk2 = tune_column_tasks.back().get();
            auto column_tune_result = tsk2->result;
            auto column_name2 = column_tune_result.first;
            best_column_mse[column_name2] = column_tune_result.second.first;
            best_column_params[column_name2] = column_tune_result.second.second;
        }
/*
        for (auto &fut: tune_column_tasks) {
            auto tsk = fut.get();
            auto column_tune_result = tsk->result;
            auto column_name = column_tune_result.first;
            best_column_mse[column_name] = column_tune_result.second.first;
            best_column_params[column_name] = column_tune_result.second.second;
        }
        */
        LOG4_DEBUG("best column_params size is " << best_column_params.size());


        for (auto it = best_column_mse.begin(); it != best_column_mse.end(); ++it) {
            LOG4_DEBUG("Best mse " << it->first << ": " << it->second);
        }

        for (auto it = best_column_params.begin(); it != best_column_params.end(); ++it) {
            LOG4_DEBUG("Best parameter " << it->first << " : " << it->second->to_string());
        }

        return std::make_pair(best_column_mse, best_column_params);
    }
};

void
tune_transformation(Dataset_ptr &p_current_dataset,
                    const std::vector<std::vector<size_t>> &svr_kernel_types_range,
                    const validation_parameters_t &validation_parameters, const svr::optimizer::NM_parameters &nm_parameters,
                    const svr::optimizer::PSO_parameters &pso_parameters, best_mse_t &best_mse, Dataset_ptr &p_best_dataset,
                    const InputQueue_ptr &p_input_queue)
{

    single_progress_step =
            100. / std::max(1., static_cast<double>(p_current_dataset->get_transformation_levels())) /
            p_best_dataset->get_input_queue()->get_value_columns().size();
    std::vector<svr::future<std::shared_ptr<tune_level_task>>> tune_level_tasks;

    std::vector<DeconQueue_ptr> decon_queues, aux_decon_queues;
    prepare_decon_data(p_current_dataset, p_input_queue, decon_queues, false);

    for (size_t idx_level = 0; idx_level <= p_current_dataset->get_transformation_levels(); ++idx_level) {
	{
	    const auto levels_todo = PROPS.get_svr_paramtune_levels();
	    const auto levels_todo_str = PROPS.get_svr_paramtune_level();
	    const auto it_level = std::find(levels_todo.begin(), levels_todo.end(), idx_level);
            if (levels_todo_str != "ALL" and it_level == levels_todo.end()) {
                LOG4_DEBUG("Skipping level " << idx_level << " as requested in config file");
                continue;
            }
	}

        for (const auto svr_kernel_type: svr_kernel_types_range[idx_level]) {
            // set svr_kernel_type
            for (auto &it: p_current_dataset->get_ensemble_svr_parameters())
                for (auto &p_svr_parameters: it.second)
                    p_svr_parameters->set_kernel_type(static_cast<kernel_type>(svr_kernel_type));


            std::vector<Dataset_ptr> datasets_for_tweaking;
            prepare_datasets_tweaking(p_current_dataset, datasets_for_tweaking);

            for (Dataset_ptr &p_current_tweaking_dataset: datasets_for_tweaking) {

                validation_parameters_t local_validation_parameters(validation_parameters);
                local_validation_parameters.model_number_ = idx_level;
                auto tsk = std::make_shared<tune_level_task>(
                        p_current_tweaking_dataset,
                        local_validation_parameters,
                        nm_parameters, pso_parameters, p_input_queue,
                        decon_queues,
                        aux_decon_queues);
                LOG4_DEBUG("tune_level_tasks::level " << idx_level);
                tune_level_tasks.push_back(svr::async([tsk]()
                                                      {
                                                          tsk->run(nullptr);
                                                          return tsk;
                                                      }));
            }
        }
    }

    auto params = p_best_dataset->get_ensemble_svr_parameters();
    for (auto &fut: tune_level_tasks) {
        auto tsk = fut.get();
        LOG4_DEBUG("A tune level task just completed..");
        auto level_tune_result = tsk->result;
        for (auto column_name : p_best_dataset->get_input_queue()->get_value_columns()) {
            const auto & column_todo = PROPS.get_svr_paramtune_column();
            if (not ((column_todo == "ALL") or (column_todo == column_name))){
                LOG4_DEBUG("Skipping column " << column_name << " as requested in config file");
                continue;
            }
            auto key = std::make_pair(p_best_dataset->get_input_queue()->get_table_name(),
                                      column_name);
            size_t level = level_tune_result.second[column_name]->get_decon_level();
            if (best_mse[key].size() <= level) best_mse[key].resize(level + 1);
            best_mse[key][level] = level_tune_result.first[column_name];
            params[key].at(level) =
                    std::make_shared<SVRParameters>(*(level_tune_result.second[column_name]));
        }
    }
    p_best_dataset->set_ensemble_svr_parameters_deep(params);

}


void init_best_mse(
        const Dataset_ptr &p_dataset, size_t max_transformation_levels, best_mse_t best_mse)
{
    for (const std::string &column_name: p_dataset->get_input_queue()->get_value_columns())
        best_mse[std::make_pair(p_dataset->get_input_queue()->get_table_name(), column_name)] =
                std::vector<double>(max_transformation_levels + 1, std::numeric_limits<double>::max());
}


InputQueue_ptr
get_input_data(
        InputQueue_ptr p_input_queue,
        const size_t frame_length,
        const size_t max_lag_count,
        const bpt::time_period &needed_data_time_period)
{
    InputQueue_ptr p_res_input_queue = AppContext::get_instance().input_queue_service.clone_with_data(
            p_input_queue, needed_data_time_period);

    p_res_input_queue->update_data(
            AppContext::get_instance().input_queue_service.get_latest_queue_data(
                    p_res_input_queue,
                    max_lag_count + frame_length + 1,
                    needed_data_time_period.begin()));

    return p_res_input_queue;
}


bpt::time_period
get_full_time_span(
        const svr::paramtune::validation_parameters_t &validation_parameters,
        const bpt::time_duration &slide_duration)
{
    LOG4_DEBUG(
            "Validation range is " << validation_parameters.validation_range_ << " training range "
                                   << validation_parameters.training_range_ << " slide duration "
                                   << slide_duration);

    const bpt::ptime beginnest_begin =
            validation_parameters.validation_range_.begin() < validation_parameters.training_range_.begin() ?
            validation_parameters.validation_range_.begin() : validation_parameters.training_range_.begin();
    const bpt::ptime endest_end =
            validation_parameters.validation_range_.end() > validation_parameters.training_range_.end() ?
            validation_parameters.validation_range_.end() : validation_parameters.training_range_.end();

    return validation_parameters.sliding_direction_ == sliding_direction_e::forward ?
           bpt::time_period(beginnest_begin, endest_end + slide_duration) : bpt::time_period(
                    beginnest_begin - slide_duration, endest_end);
}


InputQueue_ptr
prepare_input_data(
        const Dataset_ptr &p_dataset,
        const size_t frame_length,
        const validation_parameters_t &validation_parameters,
        const size_t max_lag_count,
        InputQueue_ptr &p_input_queue) /* out */
{
    LOG4_BEGIN();
    bpt::time_period needed_data_time_period = get_full_time_span(
            validation_parameters,
            bpt::time_duration(0, 0, validation_parameters.validation_slide_count_ *
                                     validation_parameters.validation_slide_period_sec_.total_seconds(), 0)
    );

    LOG4_DEBUG(
            "Need input data from " << needed_data_time_period.begin() << " until "
                                    << needed_data_time_period.last());
    p_input_queue = get_input_data(
            p_dataset->get_input_queue(), frame_length, max_lag_count, needed_data_time_period);
    auto input_queue_data = p_input_queue->get_data();
    if (input_queue_data.size() < frame_length) throw std::runtime_error("Not enough data in input queue.");

    LOG4_DEBUG("Prepared " << input_queue_data.size() << " rows from " <<
                           input_queue_data.front()->get_value_time() << " to "
                           << input_queue_data.back()->get_value_time() << "");

    if ((decltype(frame_length)) std::distance(input_queue_data.begin(), find_nearest(input_queue_data,
                                                                                      needed_data_time_period.begin())) <
        frame_length)
        throw std::logic_error("Artefacts could be present in features!");

    LOG4_DEBUG("Prepared " << p_input_queue->get_data().size() << " rows from " <<
                           p_input_queue->get_data().front()->get_value_time() << " to "
                           << p_input_queue->get_data().back()->get_value_time() << "");
    return p_input_queue;
}


size_t calculate_total_loss_calls(
        const Dataset_ptr &p_dataset,
        const std::vector<size_t> &transformation_levels_range,
        const std::vector<std::string> &transformation_names_range,
        const std::vector<std::vector<size_t>> &svr_kernel_types_range,
        const validation_parameters_t &validation_parameters,
        const svr::optimizer::NM_parameters &nm_parameters, const svr::optimizer::PSO_parameters &pso_parameters);

size_t get_longest_wavelet_order(const std::vector<std::string> &transformation_names)
{
    size_t result = 0;
    for (const auto &transformation_name: transformation_names) {
        const auto transformation_order = svr::spectral_transform::modwt_filter_order_from(transformation_name);
        if (transformation_order > result) result = transformation_order;
    }
    return result;
}

Dataset_ptr
tune_dataset(
        const Dataset_ptr &p_dataset,
        const std::vector<size_t> &transformation_levels_range,
        const std::vector<std::string> &transformation_names_range,
        const std::vector<std::vector<size_t>> &svr_kernel_types_range,
        const svr::paramtune::validation_parameters_t &validation_parameters,
        const svr::optimizer::NM_parameters &nm_parameters,
        const svr::optimizer::PSO_parameters &pso_parameters)
{
    resource_measure.set_start_time();
    Dataset_ptr p_best_dataset = std::make_shared<Dataset>(*p_dataset);
    const size_t max_transformation_levels = transformation_levels_range.empty() ? 1 : *std::max_element(
            transformation_levels_range.begin(),
            transformation_levels_range.end());
    const size_t max_lag_count = get_max_lag_count(p_dataset, validation_parameters.bounds_);
    svr::paramtune::best_mse_t best_mse;

    init_best_mse(p_dataset, max_transformation_levels, best_mse);

    total_loss_calls = calculate_total_loss_calls(
            p_dataset, transformation_levels_range, transformation_names_range, svr_kernel_types_range,
            validation_parameters,
            nm_parameters, pso_parameters);
    const auto longest_wavelet_order = get_longest_wavelet_order(transformation_names_range);
    if (transformation_names_range.size() > 1)
        throw std::logic_error("Only one transformation type can be tuned now!");
    /* TODO Heterogenous transform names */

    // maybe add sliding_direction size to get_min_frame_length
    const auto min_frame_length = svr::spectral_transform::get_min_frame_length(
            max_transformation_levels, max_lag_count, longest_wavelet_order, transformation_names_range[0]);

    InputQueue_ptr p_input_queue;
    prepare_input_data(p_dataset, min_frame_length, validation_parameters, max_lag_count, p_input_queue);

    // TODO Parallelize
    for (const size_t &transformation_levels: transformation_levels_range) {
        for (const std::string &transformation_name: transformation_names_range) {
            Dataset_ptr p_current_dataset = std::make_shared<Dataset>(*p_dataset);
            p_current_dataset->set_transformation_levels(transformation_levels);
            p_current_dataset->set_transformation_name(transformation_name);
            p_current_dataset->set_id(p_dataset->get_id());

            tune_transformation(
                    p_current_dataset, svr_kernel_types_range, validation_parameters, nm_parameters,
                    pso_parameters, best_mse,
                    p_best_dataset, p_input_queue);
        }
    }

    return p_best_dataset;
}

size_t
calculate_total_loss_calls(
        const Dataset_ptr &p_dataset,
        const std::vector<size_t> &transformation_levels_range,
        const std::vector<std::string> &transformation_names_range,
        const std::vector<std::vector<size_t>> &svr_kernel_types_range,
        const validation_parameters_t &validation_parameters,
        const svr::optimizer::NM_parameters &nm_parameters, const svr::optimizer::PSO_parameters &pso_parameters)
{
    size_t total_transformation_levels = 0;
    for (const auto transformation_levels: transformation_levels_range)
        total_transformation_levels += transformation_levels + 1;
    const size_t result = total_transformation_levels * p_dataset->get_input_queue()->get_value_columns().size() *
                          transformation_names_range.size() *
                          (pso_parameters.iteration_number_ + 1) * pso_parameters.particles_number_ *
                          p_dataset->get_input_queue()->get_value_columns().size();
    LOG4_INFO("Total loss calls expected count with current parameters is " << result);
    return result;
}

/* TODO first fix then use or drop the whole concept, now its unusable */
Dataset_ptr tune_interchangable_level(
        const Dataset_ptr &p_dataset,
        const boost::posix_time::time_period &training_range,
        const boost::posix_time::time_period &prediction_range,
        bool divide_lower_from_upper)
{
    Dataset_ptr p_best_dataset = std::make_shared<Dataset>(*p_dataset);
    auto best_params = p_best_dataset->get_ensemble_svr_parameters();

    for (auto column_name: p_dataset->get_input_queue()->get_value_columns()) {
        double best_mse = std::numeric_limits<double>::max();
        auto key = std::make_pair(p_dataset->get_input_queue()->get_table_name(),
                                  column_name);
        for (size_t idx_level = 0; idx_level <= p_dataset->get_transformation_levels(); ++idx_level) {
            Dataset_ptr p_current_dataset = std::make_shared<Dataset>(*p_best_dataset);
            auto params = p_current_dataset->get_ensemble_svr_parameters();
            std::vector<SVRParameters_ptr> new_params = p_current_dataset->get_ensemble_svr_parameters()[key];
            if (divide_lower_from_upper) {
                std::fill(new_params.begin(), std::next(new_params.begin(), idx_level),
                          params[key][idx_level]);
                std::fill(std::next(new_params.begin(), idx_level), new_params.end(),
                          params[key][std::min(idx_level + 1, p_dataset->get_transformation_levels())]);
            } else
                std::fill(new_params.begin(), new_params.end(), params[key][idx_level]);
            params[key] = new_params;
            p_current_dataset->set_ensemble_svr_parameters(params);
            svr::business::t_predict_result predict_result;
            AppContext::get_instance().dataset_service.run(
                    p_current_dataset, training_range, prediction_range, predict_result);
            auto p_orig_input_queue = p_current_dataset->get_input_queue()->clone_empty();
            p_orig_input_queue->set_data(AppContext::get_instance().input_queue_service.get_queue_data(
                    p_current_dataset->get_input_queue()->get_table_name(),
                    prediction_range.begin(), prediction_range.last()));
            const auto column_mse = std::numeric_limits<double>::quiet_NaN();
            /*
            const double column_mse = svr::paramtune::score(
                    p_orig_input_queue->get_data(),
                    *(predict_result[column_name]),
                    prediction_range,
                    AppContext::get_instance().input_queue_service.get_value_column_index(
                            p_dataset->get_input_queue(), column_name),
                    0, // Predicted index is 0 because only one column is reconstructed from a decon queue
                    score_metric_e::MAE,
                    true);
                    */
            if (column_mse < best_mse && !std::isnan(column_mse)) {
                best_params[key] = new_params;
                best_mse = column_mse;
                // log results
                LOG4_INFO("Tune interchangable level" << idx_level <<
                                                      " column name " << column_name <<
                                                      " score " << best_mse <<
                                                      " on parameters " << params[key][idx_level]->to_string());

            }
        }
    }
    p_best_dataset->set_ensemble_svr_parameters(best_params);
    return p_best_dataset;

}


} //paramtune
} //svr
#endif
