#if 0
#pragma once

#include "main-config.hpp"
#include "paramtune.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace dt = boost::date_time;

namespace svr { namespace datamodel {
    class InputQueue;
    class PredictionTask;
    class ScalingFactorsTask;
    class DecrementTask;
    class AutotuneTask;
    class DataSet;
    using InputQueue_ptr = std::shared_ptr<InputQueue>;
    using PredictionTask_ptr = std::shared_ptr<PredictionTask>;
    using ScalingFactorsTask_ptr = std::shared_ptr<ScalingFactorsTask>;
    using DecrementTask_ptr = std::shared_ptr<DecrementTask>;
    using AutotuneTask_ptr = std::shared_ptr<AutotuneTask>;
    using DataSet_ptr = std::shared_ptr<DataSet>;
} }

class CLI
{
public:
    CLI();
    bool parse(int argc, char** argv);
    int run_dataset();

    template<typename T> static T
    read_time(const std::string &time_str, const std::string &format_str = POSTGRES_DATETIME_FORMAT_STRING);
    template<typename T> T static
    get_vm_value(const po::variables_map &vm, const std::string &key_name, const std::string &error_msg = "");
    template<typename T> void static
    replace_vm_value(std::map<std::string, po::variable_value>& vm_, const std::string& option, const T& val);
    bool static check_values_count(const po::variables_map &vm, const std::string &values);

private:
    bool show_help();
    bool set_dataset_params(const po::variables_map &vm, datamodel::Dataset_ptr &p_dataset);
    bool set_prediction_params(const po::variables_map &vm_, const PredictionTask_ptr &p_prediction_task);
    bool set_scaling_factors_params(const po::variables_map &vm_, const ScalingFactorsTask_ptr &p_scaling_factors_task);
    bool set_autotune_params(const po::variables_map &vm, const AutotuneTask_ptr &p_autotune_task);
    bool set_decrement_task_params(const po::variables_map &vm, const DecrementTask_ptr &p_decrementTask);

    bool check_max_wavelet_svr_parameters(const std::map<std::string, std::string> &params, const size_t max_transform_level);

    std::vector<svr::paramtune::Bounds> read_autotune_bounds(const std::map<std::string, std::string> &params);
    svr::paramtune::Bounds              read_autotune_bounds(const std::map<std::string, std::string> & params,
                                                             const size_t transformation_level);

    po::options_description gen_desc         = po::options_description("General options");
//    po::options_description userDesc        = po::options_description("User options");
//    po::options_description queueDesc       = po::options_description("Queue options");
    po::options_description dataset_desc     = po::options_description("Dataset options");
    po::options_description prediction_desc  = po::options_description("Prediction options");
    po::options_description scaling_factors_desc  = po::options_description("Scaling factors options");
    po::options_description autotune_desc    = po::options_description("Tweaking options");
    po::options_description decrement_desc   = po::options_description("Decrement options");

    po::variables_map vm_;
    datamodel::Dataset_ptr p_dataset_ = nullptr;
    PredictionTask_ptr p_prediction_task_ = nullptr;
    ScalingFactorsTask_ptr p_scaling_factors_task_ = nullptr;
    AutotuneTask_ptr p_autotune_task_ = nullptr;
    DecrementTask_ptr p_decrement_task_ = nullptr;
    svr::paramtune::validation_parameters_t validation_parameters_;
    svr::optimizer::NM_parameters nm_parameters_;
    svr::optimizer::PSO_parameters pso_parameters_;
    void set_ensemble_svr_parameters(
            std::map<std::string, std::string> &parameters,
            const std::string &queue_table_name,
            bool is_parameters_set,
            const std::vector<std::string> &aux_table_names,
            const size_t max_transform_level,
            datamodel::Dataset_ptr &p_dataset,
            const std::vector<std::vector<size_t>> &svr_kernel_types_range);

    void process_autotune_task();
    void process_prediction_task();
    void process_scaling_factors_task();
    // Performs sliding validation (train + predict on given time windows, then slide all value_times with another given time window)
    void process_decrement_task();
};
#endif