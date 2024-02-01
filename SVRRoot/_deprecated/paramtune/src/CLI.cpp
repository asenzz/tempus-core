#if 0 // Paramtune is deprecated, models are tuned on the fly
#include "CLI.h"

#include <model/AutotuneTask.hpp>
#include <model/PredictionTask.hpp>
#include <model/ScalingFactorsTask.hpp>
#include <model/User.hpp>
#include <model/DecrementTask.hpp>

#include <appcontext.hpp>


using namespace svr::common;
using namespace svr::datamodel;
using namespace svr::context;
using namespace svr::paramtune;
using namespace bpt;

CLI::CLI()
{
    gen_desc.add_options()
            ("help", "produce help message")
            ("options-file", po::value<std::string>()->default_value(""), "Path to options from file")
            ("app-config", po::value<std::string>()->default_value("../config/app.config"),
             "Path to file with SQL configuration")
            ("user-name", po::value<string>(), "user name")   // always necessary
            ("use-onlinesvr", po::value<bool>()->default_value(false),
             "Use onlinesvr for online training on historical data")
            ("delete-dataset-id", po::value<bigint>(), "delete dataset by id");

    dataset_desc.add_options()
            ("dataset-name", po::value<string>(), "name of dataset")     // always necessary
            ("dataset-id", po::value<bigint>(), "entity id")
            ("dataset-description", po::value<string>()->default_value(std::string("")),
             "description of dataset")
            ("dataset-priority", po::value<int>()->default_value(int(2)),
             "dataset prriority: 4 - lowest, 0 - highest")
            ("transform-levels", po::value<size_t>(), "transform-levels count")
            ("transform-wavelet-name", po::value<string>(), "transform name: "
                                                            "stft; fk4, fk6, fk8, fk12; "
                                                            "db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15; "
                                                            "bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, "
                                                            "bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8; "
                                                            "coif1, coif2, coif3, coif4, coif5; "
                                                            "sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, "
                                                            "sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20")
            ("svr-max-lookback-time-gap", po::value<string>(), "max-lookback-time-gap in format DD,HH:MM:SS")
            ("is-active", po::value<bool>()->default_value(false), "Dataset activation indicator")
            ("svr-parameters", po::value<string>(),
             "A json string of all parameters related to autotune and prediction tasks")
            ("table-name", po::value<string>(), "input queue table name")
            ("aux-table-names", po::value<string>(), "auxiliary queue names");

//    queueDesc.add_options()
//        ("queue-name",      po::value<string>(),                        "queue name")                      // always necessary
//        ("resolution",      po::value<string>(),                        "resolution in format %H:%M:%S")   // always necessary
//        ("deviation",       po::value<string>(),                        "legal time deviation in format %H:%M:%S")
//        ("timezone",        po::value<long>()->default_value(long(0.0)),
//                                                                        "timezone offset in hours")
//        ("value-columns",   po::value<string>(),                        "value names of columns")
//        ("sparse",          po::value<bool>()->default_value(false),    "data is sparce")

//        ("table-description", po::value<string>(),                      "input queue table description")
//        ("data-filename",   po::value<string>(),                        "file from which data is given")
//        ;

    prediction_desc.add_options()
            ("prediction-task-id", po::value<bigint>(), "prediction task id")
            ("start-training-time", po::value<string>(), "time from which we use data for decompose")
            ("end-training-time", po::value<string>(), "time till which we use data (including) for decompose")
            ("start-forecast-time", po::value<string>(), "time from which we predict values")
            ("end-forecast-time", po::value<string>(), "time till which we predict values");

    scaling_factors_desc.add_options()
            ("scaling-factors-task-id", po::value<bigint>(), "scaling-factors task id")
            ("force-recalculate-scaling-factors", po::value<bool>(), "force recalculate scaling factors");

    autotune_desc.add_options()
            ("autotune-task-id", po::value<bigint>(), "autotune task id")
            ("result-dataset-id", po::value<bigint>(), "result dataset id where tweaked parameters will be saved")
            ("start-validation-time", po::value<string>(),
             "time from which we predict values for calculating MSE at every auto tuning cycle")
            ("end-validation-time", po::value<string>(),
             "time till which we predict values for calculating MSE at every auto tuning cycle")
            ("pso-topology", po::value<std::string>()->default_value("random"),
             "PSO topology strategy: global or random")
            ("pso-particle-count", po::value<size_t>()->default_value(20), "Total number of particles in PSO")
            ("pso-iteration-count", po::value<size_t>()->default_value(10), "Total number of iterations in PSO")
            ("pso-best-points-count", po::value<size_t>()->default_value(3),
             "Number of best parameter sets obtained using PSO for which the Nelder-Mead method is further applied")
            ("nm-max-iterations", po::value<size_t>()->default_value(5), "Maximum number of Nelder-Mead iterations")
            ("nm-tolerance", po::value<double>()->default_value(1e-8), "Nelder-Mead error tolerance")
            ("validation-slide-count", po::value<size_t>()->default_value(1), "Number of window slides")
            ("validation-slide-direction", po::value<string>()->default_value("forward"), "Direction of sliding")
            ("validation-slide-period", po::value<std::string>()->default_value("0,00:00:00"),
             "Step size through window sliding DD,HH:MM:SS")
            ("pso-phi-velocity", po::value<double>()->default_value(1.0), "phi_velocity parameter in PSO")
            ("pso-phi-local", po::value<double>()->default_value(2.0), "phi_local parameter in PSO")
            ("pso-phi-global", po::value<double>()->default_value(2.0), "phi_global parameter in PSO")
            ("pso-state-file", po::value<string>()->default_value(""), "Filename to load PSO state from it");

    decrement_desc.add_options()
            ("decrement-task-id", po::value<bigint>(), "decrement task id")
            ("decrement-step", po::value<std::string>(), "decrement step size DD,HH:MM:SS");
}

void CLI::set_ensemble_svr_parameters(
        std::map<std::string, std::string> &parameters,
        const std::string &queue_table_name,
        bool is_parameters_set,
        const std::vector<std::string> &aux_table_names,
        const size_t max_transform_level,
        datamodel::Dataset_ptr &p_dataset,
        const std::vector<std::vector<size_t>> &svr_kernel_types_range)
{
    std::map<std::pair<std::string, std::string>, std::vector<datamodel::SVRParameters_ptr>> ensembles_svr_parameters;
    std::vector<std::string> table_names = aux_table_names;
    table_names.push_back(queue_table_name);
    auto &ap = PROPS;
    for (const std::string &table_name: table_names) {
        for (const std::string &column_name: p_dataset->get_input_queue()->get_value_columns()) {
            vector<datamodel::SVRParameters_ptr> vec_svr_parameters;
            for (size_t idx_model = 0; idx_model <= max_transform_level; ++idx_model)
                if (is_parameters_set) {
                    std::string idx_model_str = std::to_string(idx_model);
                    vec_svr_parameters.push_back(
                            std::make_shared<SVRParameters>(
                                    0,
                                    0,
                                    p_dataset->get_input_queue()->get_table_name(),
                                    column_name,
                                    idx_model,
                                    std::stod(parameters["svr_c_" + idx_model_str]),
                                    std::stod(parameters["svr_epsilon_" + idx_model_str]),
                                    std::stod(parameters["svr_kernel_param_" + idx_model_str]),
                                    std::stod(parameters["svr_kernel_param2_" + idx_model_str]),
                                    std::stoull(parameters["svr_decremental_distance_" + idx_model_str]),
                                    std::stod(parameters["svr_adjacent_levels_ratio_" + idx_model_str]),
                                    static_cast<kernel_type_e>(svr_kernel_types_range[idx_model][0]),
                                    std::stoull(parameters["lag_count_" + idx_model_str]),
                                    DEFAULT_APP_HYPERPARAMS(ap)
                            ));
                } else {
                    vec_svr_parameters.push_back(
                            std::make_shared<SVRParameters>(
                                    0,
                                    0,
                                    p_dataset->get_input_queue()->get_table_name(),
                                    column_name,
                                    idx_model,
                                    DEFAULT_SVRPARAM_SVR_COST,
                                    DEFAULT_SVRPARAM_SVR_EPSILON,
                                    DEFAULT_SVRPARAM_KERNEL_PARAM_1,
                                    DEFAULT_SVRPARAM_KERNEL_PARAM_2,
                                    DEFAULT_SVRPARAM_DECREMENT_DISTANCE,
                                    DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO,
                                    DEFAULT_SVRPARAM_KERNEL_TYPE,
                                    DEFAULT_SVRPARAM_LAG_COUNT,
                                    DEFAULT_APP_HYPERPARAMS(ap)
                            ));
                }

            ensembles_svr_parameters[std::make_pair(table_name, column_name)] = vec_svr_parameters;
        }
    }
    p_dataset->get_ensembles().clear();
    p_dataset->set_ensemble_svr_parameters(ensembles_svr_parameters);
}

bool CLI::set_dataset_params(
        const po::variables_map &vm, datamodel::Dataset_ptr &p_dataset)
{
    bool is_parameters_set;
    size_t max_transform_level;
    std::vector<std::vector<size_t>> svr_kernel_types_range;
    std::map<std::string, std::string> parameters;
    std::vector<size_t> transformation_levels_range;
    std::vector<std::string> transformation_names_range;

    if (vm.count("dataset-id")) p_dataset->set_id(vm["dataset-id"].as<bigint>());
    if (vm.count("user-name")) p_dataset->set_user_name(vm["user-name"].as<string>());
    if (vm.count("dataset-name")) p_dataset->set_dataset_name(vm["dataset-name"].as<string>());
    string queue_table_name = vm.count("table-name") ? vm["table-name"].as<string>() : string();
    std::vector<std::string> aux_table_names;
    if (!queue_table_name.empty()) {
        datamodel::InputQueue_ptr p_input_queue = AppContext::get_instance().input_queue_service.get_queue_metadata(
                queue_table_name);
        p_dataset->set_input_queue(p_input_queue);
    }

    if (vm.count("dataset-description")) p_dataset->set_description(vm["dataset-description"].as<string>());
    if (vm.count("dataset-priority")) p_dataset->set_priority(static_cast<Priority>(vm["dataset-priority"].as<int>()));
    if (vm.count("is-active")) p_dataset->set_is_active(vm["is-active"].as<bool>());
    if (vm.count("svr-max-lookback-time-gap"))
        p_dataset->set_max_lookback_time_gap(
                date_time_string_to_seconds(vm["svr-max-lookback-time-gap"].as<string>()));
    if (vm.count("svr-parameters") < 1) goto _bail;

    parameters = svr::common::json_to_map(vm["svr-parameters"].as<string>());

    is_parameters_set = true;

    transformation_levels_range = svr::common::parse_string_range(parameters["transformation_levels"]);
    transformation_names_range = svr::common::parse_string_range(parameters["transformation_name"],
                                                                 svr::datamodel::transformation_names);

    if (transformation_levels_range.size() > 1 || transformation_names_range.size() > 1
        || vm["svr-parameters"].as<string>().find(svr::common::dd_separator) != std::string::npos) {
        is_parameters_set = false;
    }

    max_transform_level = *std::max_element(transformation_levels_range.begin(), transformation_levels_range.end());
    p_dataset->set_transformation_levels(max_transform_level);
    if (transformation_names_range.empty()) {
        LOG4_ERROR("transformation_names_range.size() == 0");
        throw std::runtime_error("transformation_names_range.size() == 0");
    }
/*
    for (auto it = transformation_names_range.cbegin(); it != transformation_names_range.cend(); ++it)
        if (*it != *transformation_names_range.cbegin())
            throw std::runtime_error("Transformation names are not all identical!");
*/
    p_dataset->set_transformation_name(transformation_names_range[0]);
    if (!check_max_wavelet_svr_parameters(parameters, max_transform_level)) {
        LOG4_ERROR("Wrong svr-parameters" << max_transform_level);
        return false;
    }

    for (size_t i = 0; i <= max_transform_level; ++i) {
        svr_kernel_types_range.push_back(
                svr::common::parse_string_range(parameters["svr_kernel_type_" + std::to_string(i)]));
        if (is_parameters_set && svr_kernel_types_range[i].size() > 1)
            is_parameters_set = false;
    }

    // autotune needed if we have svr_parameters diapason
    if (!is_parameters_set && vm.find("autotune-task-id") == vm.end() && vm.find("scaling-factors-task-id") == vm.end()) { // TODO Revise buggy
        LOG4_ERROR("Either autotune or scaling factors task needs to be specified");
        return false;
    }

    // set ensembles_svr_parameters
    set_ensemble_svr_parameters(parameters, queue_table_name, is_parameters_set, aux_table_names, max_transform_level,
                                p_dataset, svr_kernel_types_range);

    _bail:
    AppContext::get_instance().dataset_service.save(
            p_dataset); // TODO Remove save dataset because needed real dataset_id

    return true;
}


bool CLI::set_prediction_params(
        const po::variables_map &vm,
        const PredictionTask_ptr &p_prediction_task)
{
    if (vm.count("prediction-task-id")) p_prediction_task->set_id(vm["prediction-task-id"].as<bigint>());
    if (vm.count("dataset-id")) p_prediction_task->set_dataset_id(vm["dataset-id"].as<bigint>());
    if (vm.count("start-training-time"))
        p_prediction_task->set_start_train_time(
                CLI::read_time<bpt::ptime>(vm["start-training-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-training-time"))
        p_prediction_task->set_end_train_time(
                CLI::read_time<ptime>(vm["end-training-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("start-forecast-time"))
        p_prediction_task->set_start_prediction_time(
                CLI::read_time<ptime>(vm["start-forecast-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-forecast-time"))
        p_prediction_task->set_end_prediction_time(
                CLI::read_time<ptime>(vm["end-forecast-time"].as<string>(), "%Y.%m.%d,%H:%M"));

    if (p_prediction_task->get_start_train_time() > p_prediction_task->get_end_train_time()
        || p_prediction_task->get_start_prediction_time() > p_prediction_task->get_end_prediction_time()) {
        LOG4_ERROR("Wrong time diapasons! Should be start time <= end time.");
        return false;
    }
    return true;
}

bool CLI::set_scaling_factors_params(
        const po::variables_map &vm,
        const ScalingFactorsTask_ptr &p_scaling_factors_task)
{
    if (vm.count("scaling-factors-task-id")) p_scaling_factors_task->set_id(vm["scaling-factors-task-id"].as<bigint>());
    if (vm.count("dataset-id")) p_scaling_factors_task->set_dataset_id(vm["dataset-id"].as<bigint>());
    if (vm.count("force-recalculate-scaling-factors")) p_scaling_factors_task->set_force_recalculate_scaling_factors(vm["force-recalculate-scaling-factors"].as<bool>());
    return true;
}


bool CLI::set_autotune_params(
        const boost::program_options::variables_map &vm,
        const AutotuneTask_ptr &p_autotune_task)
{
    //TODO: check if it's needed
//    if(is_parameters_set)
//    {
//        throw std::runtime_error("All parameters are fixed. Autotune task cannot be proceeded.");
//    }
    if (vm.count("dataset-id")) p_autotune_task->set_dataset_id(vm["dataset-id"].as<bigint>());
    if (vm.count("result-dataset-id")) p_autotune_task->set_result_dataset_id(vm["result-dataset-id"].as<bigint>());
    try {
        if (vm.count("svr-parameters")) p_autotune_task->set_parameters_from_string(vm["svr-parameters"].as<string>());
    } catch (const std::exception &ex) {
        LOG4_FATAL("Error parsing SVR parameters string. " << ex.what());
        return false;
    }
    if (vm.count("start-training-time"))
        p_autotune_task->set_start_train_time(
                read_time<ptime>(vm["start-training-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-training-time"))
        p_autotune_task->set_end_train_time(
                read_time<ptime>(vm["end-training-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("start-validation-time"))
        p_autotune_task->set_start_validation_time(
                read_time<ptime>(vm["start-validation-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-validation-time"))
        p_autotune_task->set_end_validation_time(
                read_time<ptime>(vm["end-validation-time"].as<string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("pso-topology")) {
        string topology = vm["pso-topology"].as<string>();
        if (topology == "global")
            p_autotune_task->set_pso_topology(static_cast<size_t>(svr::optimizer::PsoTopology::global));
        else if (topology == "ring")
            p_autotune_task->set_pso_topology(static_cast<size_t>(svr::optimizer::PsoTopology::ring));
        else if (topology == "random")
            p_autotune_task->set_pso_topology(static_cast<size_t>(svr::optimizer::PsoTopology::random));
        else {
            LOG4_ERROR("Wrong pso-topology");
            return false;
        }
    }

    if (vm.count("pso-particle-count"))
        p_autotune_task->set_pso_particles_number(vm["pso-particle-count"].as<size_t>());
    if (vm.count("pso-iteration-count"))
        p_autotune_task->set_pso_iteration_number(vm["pso-iteration-count"].as<size_t>());
    if (vm.count("pso-best-points-count"))
        p_autotune_task->set_pso_best_points_counter(vm["pso-best-points-count"].as<size_t>());
    if (vm.count("pso-state-file"))
        p_autotune_task->set_pso_state_file(vm["pso-state-file"].as<std::string>());
    if (vm.count("nm-max-iterations"))
        p_autotune_task->set_nm_max_iteration_number(vm["nm-max-iterations"].as<size_t>());
    if (vm.count("nm-tolerance")) p_autotune_task->set_nm_tolerance(vm["nm-tolerance"].as<double>());

    if (vm.count("validation-slide-count"))
        p_autotune_task->set_vp_slide_count(vm["validation-slide-count"].as<size_t>());
    if (vm.count("validation-slide-direction")) {
        string direction = vm["validation-slide-direction"].as<string>();
        if (direction == "forward") p_autotune_task->set_vp_sliding_direction(0);
        else if (direction == "backward") p_autotune_task->set_vp_sliding_direction(1);
        else {
            LOG4_ERROR("Wrong validation-slide-count");
            return false;
        }
    }
    if (vm.count("validation-slide-period"))
        p_autotune_task->set_vp_slide_period_sec(
                date_time_string_to_seconds(vm["validation-slide-period"].as<string>()));
    return true;
}

bool CLI::set_decrement_task_params(
        const boost::program_options::variables_map &vm,
        const DecrementTask_ptr &p_decrementTask)
{
    if (vm.count("decrement-task-id")) p_decrementTask->set_id(vm["decrement-task-id"].as<bigint>());
    if (vm.count("dataset-id")) p_decrementTask->set_dataset_id(vm["dataset-id"].as<bigint>());
    if (vm.count("start-training-time"))
        p_decrementTask->set_start_train_time(
                read_time<bpt::ptime>(vm["start-training-time"].as<std::string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-training-time"))
        p_decrementTask->set_end_train_time(
                read_time<bpt::ptime>(vm["end-training-time"].as<std::string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("start-validation-time"))
        p_decrementTask->set_start_validation_time(
                read_time<bpt::ptime>(vm["start-validation-time"].as<std::string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("end-validation-time"))
        p_decrementTask->set_end_validation_time(
                read_time<bpt::ptime>(vm["end-validation-time"].as<std::string>(), "%Y.%m.%d,%H:%M"));
    if (vm.count("svr-parameters")) p_decrementTask->set_parameters(vm["svr-parameters"].as<std::string>());
    if (vm.count("decrement-step")) p_decrementTask->set_decrement_step(vm_["decrement-step"].as<string>());

    if (p_decrementTask->get_start_train_time() > p_decrementTask->get_end_train_time()
        || p_decrementTask->get_start_validation_time() > p_decrementTask->get_end_validation_time()) {
        LOG4_ERROR("Wrong time diapasons! Should be start-time <= end-time");
        return false;
    }
    return true;
}

bool CLI::check_max_wavelet_svr_parameters(const map<string, string> &params, const size_t max_transform_level)
{
    for (size_t i = 0; i <= max_transform_level; ++i) {
        if (!(params.count("svr_kernel_type_" + std::to_string(i))
              && params.count("svr_c_" + std::to_string(i))
              && params.count("svr_epsilon_" + std::to_string(i))
              && params.count("svr_kernel_param_" + std::to_string(i))
              && params.count("svr_kernel_param2_" + std::to_string(i))
              && params.count("svr_decremental_distance_" + std::to_string(i))
              && params.count("svr_adjacent_levels_ratio_" + std::to_string(i))
              && params.count("lag_count_" + std::to_string(i)))) {
            return false;
        }
    }
    return true;
}

std::vector<Bounds> CLI::read_autotune_bounds(const std::map<string, string> &params)
{
    std::vector<Bounds> bounds;
    std::vector<size_t> transformation_levels_range{
            svr::common::parse_string_range(params.at("transformation_levels"))};
    const size_t max_levels = *std::max_element(transformation_levels_range.begin(), transformation_levels_range.end());
    for (size_t current_level = 0; current_level <= max_levels; current_level++)
        bounds.push_back(read_autotune_bounds(p_autotune_task_->get_parameters(), current_level));
    return bounds;
}

#define GET_MIN_MAX() \
    sep_pos = value.find(separator); \
    min_value = value.substr(0, sep_pos); \
    max_value = value.substr(sep_pos + separator.size(), value.size()); \
    if (sep_pos != std::string::npos || min_value == max_value) {


Bounds CLI::read_autotune_bounds(const std::map<string, string> &params, const size_t transformation_level)
{
    const std::string separator{".."};

    std::string value, min_value, max_value;
    size_t sep_pos;
    size_t ss;

    Bounds level_bounds;
    value = params.at("svr_c_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_C(std::stof(min_value));
        level_bounds.max_bounds.set_svr_C(std::stod(max_value));
        level_bounds.is_tuned.svr_C = true;
    } else {
        level_bounds.min_bounds.set_svr_C(std::stod(value, &ss));
        level_bounds.max_bounds.set_svr_C(std::stod(value, &ss));
        level_bounds.is_tuned.svr_C = false;
    }

    value = params.at("svr_epsilon_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_epsilon(std::stof(min_value));
        level_bounds.max_bounds.set_svr_epsilon(std::stod(max_value));
        level_bounds.is_tuned.svr_epsilon = true;
    } else {
        level_bounds.min_bounds.set_svr_epsilon(std::stod(value, &ss));
        level_bounds.max_bounds.set_svr_epsilon(std::stod(value, &ss));
        level_bounds.is_tuned.svr_epsilon = false;
    }

    value = params.at("svr_kernel_param_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_kernel_param(std::stof(min_value));
        level_bounds.max_bounds.set_svr_kernel_param(std::stod(max_value));
        level_bounds.is_tuned.svr_kernel_param = true;
    } else {
        level_bounds.min_bounds.set_svr_kernel_param(std::stod(value, &ss));
        level_bounds.max_bounds.set_svr_kernel_param(std::stod(value, &ss));
        level_bounds.is_tuned.svr_kernel_param = false;
    }

    value = params.at("svr_kernel_param2_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_kernel_param2(std::stod(min_value));
        level_bounds.max_bounds.set_svr_kernel_param2(std::stod(max_value));
        level_bounds.is_tuned.svr_kernel_param2 = true;
    } else {
        level_bounds.min_bounds.set_svr_kernel_param2(std::stod(value, &ss));
        level_bounds.max_bounds.set_svr_kernel_param2(std::stod(value, &ss));
        level_bounds.is_tuned.svr_kernel_param2 = false;
    }

    value = params.at("svr_decremental_distance_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_decremental_distance(
                static_cast<u_int64_t>(std::stoull(min_value)));
        level_bounds.max_bounds.set_svr_decremental_distance(
                static_cast<u_int64_t>(std::stoull(max_value)));
        level_bounds.is_tuned.svr_decremental_distance = true;
    } else {
        level_bounds.min_bounds.set_svr_decremental_distance(static_cast<u_int64_t>(std::stoull(value)));
        level_bounds.max_bounds.set_svr_decremental_distance(static_cast<u_int64_t>(std::stoull(value)));
        level_bounds.is_tuned.svr_decremental_distance = false;
    }

    value = params.at("svr_adjacent_levels_ratio_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_svr_adjacent_levels_ratio(std::stod(min_value));
        level_bounds.max_bounds.set_svr_adjacent_levels_ratio(std::stod(max_value));
        level_bounds.is_tuned.svr_adjacent_levels_ratio = true;
    } else {
        level_bounds.min_bounds.set_svr_adjacent_levels_ratio(std::stod(value, &ss));
        level_bounds.max_bounds.set_svr_adjacent_levels_ratio(std::stod(value, &ss));
        level_bounds.is_tuned.svr_adjacent_levels_ratio = false;
    }

    value = params.at("lag_count_" + std::to_string(transformation_level));
    GET_MIN_MAX() ;
        level_bounds.min_bounds.set_lag_count(std::stoull(min_value));
        level_bounds.max_bounds.set_lag_count(std::stoull(max_value));
        level_bounds.is_tuned.lag_count = true;
    } else {
        level_bounds.min_bounds.set_lag_count(std::stoull(value, &ss));
        level_bounds.max_bounds.set_lag_count(std::stoull(value, &ss));
        level_bounds.is_tuned.lag_count = false;
    }

    return level_bounds;
}

template<typename T>
T CLI::read_time(const string &time_str, const string &format_str)
{
    T time;
    locale format = locale(locale::classic(), new time_input_facet(format_str));
    std::istringstream is(time_str);
    is.imbue(format);
    is >> time;
    return time;
}

template<typename T>
T CLI::get_vm_value(const po::variables_map &vm, const string &key_name, const string &error_msg)
{
    if (!vm.count(key_name.c_str()))
        throw std::runtime_error(error_msg.empty() ? "Key " + key_name + " doesn't exist"
                                                   : error_msg);

    return vm[key_name.c_str()].as<T>();
}

template<typename T>
void CLI::replace_vm_value(std::map<string, po::variable_value> &vm, const string &option, const T &val)
{
    vm[option].value() = boost::any(val);
}

bool CLI::check_values_count(const po::variables_map &vm, const string &values)
{
    stringstream ss(values);
    string key;
    while (ss >> key) {
        if (!vm.count(key)) {
            LOG4_ERROR("Key " << key << " wasn't set");
            return false;
        }
    }
    return true;
}

bool CLI::show_help()
{
    std::cout << gen_desc << "\n";
    /* << dataset_desc << "\n"
              << prediction_desc << "\n" << autotune_desc << "\n" << decrement_desc << "\n"; */
    return false;
}


static const PredictionTask c_paramtune_default_prediction_task(
        0, 0,
        CLI::read_time<bpt::ptime>("2015.03.23,02:00", "%Y.%m.%d,%H:%M"),
        CLI::read_time<bpt::ptime>("2015.03.24,22:00", "%Y.%m.%d,%H:%M"),
        CLI::read_time<bpt::ptime>("2015.03.24,22:01", "%Y.%m.%d,%H:%M"),
        CLI::read_time<bpt::ptime>("2015.03.24,22:15", "%Y.%m.%d,%H:%M"));

static const ScalingFactorsTask c_paramtune_default_scaling_factors_task(
        0, 0);

static const AutotuneTask c_paramtune_default_autotune_task(
        0, 0, 0, bpt::second_clock::local_time(), bpt::second_clock::local_time(),
        "{\"transformation_levels\":\"1\", \"transformation_name\":\"bior1.3\", \"svr_c_0\":\"1.0..200.0\", \"svr_epsilon_0\":\"0.000001..0.3\",\
       \"svr_kernel_param_0\":\"0.1..2.5\", \"svr_kernel_param2_0\":\"0.1..2.5\", \"svr_decremental_distance_0\":\"1\",\
       \"svr_adjacent_levels_ratio_0\":\"0.0..1.0\", \"svr_kernel_type_0\":\"3\",\"svr_error_tolerance_0\": \"tune\",\
       \"lookback_rows_0\": \"1..10\", \"lag_count_0\": \"95\", \"svr_c_1\":\"1.0..200.0\", \"svr_epsilon_1\":\"0.000001..0.3\",\
       \"svr_kernel_param_1\":\"0.1..2.5\", \"svr_kernel_param2_1\":\"0.1..2.5\", \"svr_decremental_distance_1\":\"1\",\
       \"svr_adjacent_levels_ratio_1\":\"0.0..1.0\", \"svr_kernel_type_1\":\"3\" ,\"svr_error_tolerance_1\": \"tune\",\
       \"lookback_rows_1\": \"1..10\", \"lag_count_1\": \"25\"}",
        CLI::read_time<bpt::ptime>("2015.03.23,02:00"),
        CLI::read_time<bpt::ptime>("2015.03.24,22:00"),
        CLI::read_time<bpt::ptime>("2015.03.24,22:01"), CLI::read_time<bpt::ptime>("2015.03.24,22:15"),
        0, 1, bpt::seconds(20), 1, 1, 1, 0, 1, double(1e-08));


bool CLI::parse(int argc, char **argv)
{
    gen_desc.add(dataset_desc);
    gen_desc.add(prediction_desc);
    gen_desc.add(scaling_factors_desc);
    gen_desc.add(autotune_desc);
    gen_desc.add(decrement_desc);

    // parse command line
    try {
        po::store(po::command_line_parser(argc, argv).options(gen_desc).run(), vm_);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Illegal argument!" << ex.what());
        return show_help();
    }

    if (vm_.count("help")) return show_help();

    //parse options file
    std::string options_file;
    if (vm_.count("options-file")
        && !(options_file = vm_["options-file"].as<std::string>()).empty()) {
        ifstream f;
        f.open(options_file, std::fstream::in);
        if (!f.is_open()) {
            LOG4_ERROR("Can't open options file " << options_file);
            return false;
        }
        string ss;
        char c;
        bool skipspace = false;
        while (f >> noskipws >> c) {
            if (c == '{') skipspace = true;
            else if (c == '}') skipspace = false;
            else if (skipspace && (c == '\r' || c == '\n' || c == '\t')) c = ' ';
            ss.push_back(c);
        }
        f.close();
        cout << ss;

        try {
            stringstream strstream(ss);
            po::store(po::parse_config_file(strstream, gen_desc), vm_);
        } catch (const boost::program_options::error_with_option_name &ex) {
            LOG4_ERROR("Illegal argument! " << ex.what());
            return show_help();
        }
    }

//    for (const auto &vm_pair: vm_)
//        LOG4_DEBUG(formatter() << "Key " << vm_pair.first << ", value " << vm_pair.second);

    // Initialize context object
    if (vm_["app-config"].as<string>().empty()) {
        LOG4_ERROR("app-config are not set or empty");
        return false;
    }

    try {
        AppContext::init_instance(vm_["app-config"].as<string>().c_str());
    } catch (const std::invalid_argument &ex) {
        LOG4_FATAL("Failed loading configuration. " << ex.what());
        return false;
    }

    if (vm_.count("delete-dataset-id")) {
        bigint id{vm_["delete-dataset-id"].as<bigint>()};
        datamodel::Dataset_ptr p_dataset = AppContext::get_instance().dataset_service.get(id);
        AppContext::get_instance().dataset_service.remove(p_dataset);
        return false;
    }

    bigint dataset_id = 0;
    if (vm_.count("prediction-task-id")) {
        bigint id{vm_["prediction-task-id"].as<bigint>()};
        // load task
        p_prediction_task_ = AppContext::get_instance().prediction_task_service.get_by_id(id);
        if (p_prediction_task_ == nullptr) {
            LOG4_WARN("PredictionTask with id:" << id << " doesn't exist");
            p_prediction_task_ = std::make_shared<PredictionTask>(c_paramtune_default_prediction_task);
            replace_vm_value<bigint>(vm_, "prediction-task-id", 0);
        }
        // override task parameters from cli arguments
        if (!set_prediction_params(vm_, p_prediction_task_)) {
            show_help();
            return false;
        }
        dataset_id = p_prediction_task_->get_dataset_id();
    }

    if (vm_.count("scaling-factors-task-id")) {
        bigint id{vm_["scaling-factors-task-id"].as<bigint>()};
        // load task
        p_scaling_factors_task_ = AppContext::get_instance().scaling_factors_task_service.get_by_id(id);
        if (p_scaling_factors_task_== nullptr) {
            LOG4_WARN("ScalingFactorsTask with id:" << id << " doesn't exist");
            p_scaling_factors_task_ = std::make_shared<ScalingFactorsTask>(c_paramtune_default_scaling_factors_task);
            replace_vm_value<bigint>(vm_, "scaling-factors-task-id", 0);
        }
        // override task parameters from cli arguments
        if (!set_scaling_factors_params(vm_, p_scaling_factors_task_)) {
            show_help();
            return false;
        }
        dataset_id = p_scaling_factors_task_->get_dataset_id();
    }

    if (vm_.count("autotune-task-id")) {
        bigint id{vm_["autotune-task-id"].as<bigint>()};
        // load task
        p_autotune_task_ = AppContext::get_instance().autotune_task_service.get_by_id(id);
        if (p_autotune_task_ == nullptr) {
            LOG4_WARN("AutotuneTask with id:" << id << " doesn't exist");
            p_autotune_task_ = std::make_shared<AutotuneTask>(c_paramtune_default_autotune_task);
            replace_vm_value<bigint>(vm_, "autotune-task-id", 0);
        }
        // override task parameters from cli arguments
        if (!set_autotune_params(vm_, p_autotune_task_)) {
            show_help();
            return false;
        }
        dataset_id = p_autotune_task_->get_dataset_id();
    }

    if (vm_.count("decrement-task-id")) {
        bigint id{vm_["decrement-task-id"].as<bigint>()};
        // load task
        p_decrement_task_ = AppContext::get_instance().decrement_task_service.get_by_id(id);
        if (p_decrement_task_ == nullptr) {
            LOG4_WARN("DecrementTask with id:" << id << " doesn't exist");
            p_decrement_task_ = std::make_shared<DecrementTask>();
            replace_vm_value<bigint>(vm_, "decrement-task-id", 0);
        }
        // override task parameters from cli arguments
        if (!set_decrement_task_params(vm_, p_decrement_task_)) {
            show_help();
            return false;
        }
        dataset_id = p_decrement_task_->get_dataset_id();
    }

    if (!p_autotune_task_ && !p_decrement_task_ && !p_prediction_task_ && !p_scaling_factors_task_) {
        LOG4_WARN("Any task-id wasn't set");
        show_help();
        return false;
    }

    try {
        if (dataset_id) p_dataset_ = AppContext::get_instance().dataset_service.get(dataset_id);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Failed loading dataset with id " << dataset_id << ". " << ex.what());
    }
    if (p_dataset_ == nullptr) {
        LOG4_INFO("Dataset with id: " << dataset_id << " doesn't exist. Setting defaults.");
        p_dataset_ = std::make_shared<Dataset>(
                Dataset(0, "Auto created", "svrwave", "q_svrwave_eurusd_60", {}, Priority::Normal, "Auto created", 1, "fk4",
                        boost::posix_time::time_duration(seconds(1)), std::vector<datamodel::Ensemble_ptr>(), false));
    }

    return set_dataset_params(vm_, p_dataset_);
}

bpt::time_duration advance_slides(validation_parameters_t params)
{
    bpt::time_duration advanced_slide;
    switch (params.sliding_direction_) {
        case sliding_direction_e::forward:
            advanced_slide = params.validation_slide_period_sec_;
            break;

        case sliding_direction_e::backward:
            advanced_slide = -params.validation_slide_period_sec_;
            break;
    }
    return advanced_slide;
}


void CLI::process_autotune_task()
{
    validation_parameters_.training_range_ = bpt::time_period(p_autotune_task_->get_start_train_time(),
                                                              p_autotune_task_->get_end_train_time());
    validation_parameters_.validation_range_ = bpt::time_period(p_autotune_task_->get_start_validation_time(),
                                                                p_autotune_task_->get_end_validation_time());
    validation_parameters_.sliding_direction_ = static_cast<sliding_direction_e>(p_autotune_task_->get_vp_sliding_direction());
    validation_parameters_.validation_slide_count_ = p_autotune_task_->get_vp_slide_count();
    validation_parameters_.validation_slide_period_sec_ = p_autotune_task_->get_vp_slide_period_sec();
    validation_parameters_.best_points_count_ = p_autotune_task_->get_pso_best_points_counter();
    validation_parameters_.bounds_ = read_autotune_bounds(p_autotune_task_->get_parameters());

    // setting particle swarm optimization parameters
    pso_parameters_.pso_topology_ = static_cast<svr::optimizer::PsoTopology>(p_autotune_task_->get_pso_topology());
    pso_parameters_.particles_number_ = p_autotune_task_->get_pso_particles_number();
    pso_parameters_.iteration_number_ = p_autotune_task_->get_pso_iteration_number();
    pso_parameters_.state_file_ = p_autotune_task_->get_pso_state_file();

    // setting nelder-mead parameters
    nm_parameters_.max_iteration_number_ = p_autotune_task_->get_nm_max_iteration_number();
    nm_parameters_.tolerance_ = p_autotune_task_->get_nm_tolerance();

    //set status in progress
    p_autotune_task_->set_dataset_id(p_dataset_->get_id());
    p_autotune_task_->set_status(1);
    AppContext::get_instance().autotune_task_service.save(p_autotune_task_);

    // parse parameters: transformation_levels, transformation_name, svr_kernel_type
    std::map<std::string, std::string> parameters{p_autotune_task_->get_parameters()};

    std::vector<size_t> transformation_levels_range{
            svr::common::parse_string_range(parameters["transformation_levels"])};
    std::vector<std::string> transformation_names_range{
            svr::common::parse_string_range(parameters["transformation_name"], svr::datamodel::transformation_names)};

    std::vector<std::vector<size_t>> svr_kernel_types_range;
    const size_t transformation_max_levels{
            *std::max_element(transformation_levels_range.begin(), transformation_levels_range.end())};
    for (size_t i = 0; i <= transformation_max_levels; ++i)
        svr_kernel_types_range.push_back(
                svr::common::parse_string_range(parameters["svr_kernel_type_" + std::to_string(i)]));

    auto p_best_dataset = svr::paramtune::tune_dataset(
            p_dataset_,
            transformation_levels_range,
            transformation_names_range,
            svr_kernel_types_range,
            validation_parameters_, nm_parameters_,
            pso_parameters_);

    LOG4_DEBUG(
            "Validation parameters used in training range " << validation_parameters_.training_range_.begin() << " to "
                                                            << validation_parameters_.training_range_.end());
    double final_tuned_mse = 0.;
    size_t validation_ct = 0;
    auto training_window = validation_parameters_.training_range_;
    auto validation_window = validation_parameters_.validation_range_;
    for (size_t ix_pos = 0; ix_pos < validation_parameters_.validation_slide_count_; ++ix_pos) {

        std::vector<datamodel::DeconQueue_ptr> predicted_decon_queues;
        auto column_mse = std::numeric_limits<double>::quiet_NaN();
        LOG4_DEBUG("Training and predicting for final score.");

        try {
            predicted_decon_queues = AppContext::get_instance().dataset_service.run_dataset_aux(
                    p_best_dataset, training_window, validation_window);
        } catch (const std::exception &ex) {
            LOG4_DEBUG("Exception during prediction of decon queues " << ex.what() << " Skipping validation slide");
            training_window.shift(advance_slides(validation_parameters_));
            validation_window.shift(advance_slides(validation_parameters_));
            validation_ct += 1.;
            final_tuned_mse = std::numeric_limits<double>::max();
            continue;
        }
        LOG4_DEBUG("Predicted decons done, extracting reference values.");
        LOG4_DEBUG("Validation windows " << validation_window);
        auto p_validation_input_queue = p_dataset_->get_input_queue()->clone_empty();
        p_validation_input_queue->set_data(AppContext::get_instance().input_queue_service.load(
                p_dataset_->get_input_queue()->get_table_name(),
                predicted_decon_queues[0]->get_data().begin()->get()->get_value_time(),
                predicted_decon_queues[0]->get_data().rbegin()->get()->get_value_time()));

        LOG4_DEBUG("Reconstructing and scoring.");

        // Reconstruct the values; extract the predictions to compare with etalon
        for (const auto &p_predicted_decon_queue: predicted_decon_queues) {
            try {
                const auto predicted_data = AppContext::get_instance().decon_queue_service.reconstruct(
                        datarow_range(p_predicted_decon_queue->get_data()),
                        p_best_dataset->get_transformation_name(),
                        p_best_dataset->get_transformation_levels());


                //predicted_decon_queues are returned filled with data for the full training_range and thus
                //they need to be trimmed to validation_period only for score
                data_row_container trimmed_predicted;
                for (auto rev_iter = predicted_data.rbegin();
                     rev_iter->get()->get_value_time() >= validation_window.begin() && rev_iter != predicted_data.rend(); ++rev_iter)
                    if (validation_window.contains(rev_iter->get()->get_value_time()))
                        trimmed_predicted.push_back(*rev_iter);

                column_mse = svr::paramtune::score(
                        p_validation_input_queue->get_data(),
                        trimmed_predicted,
                        AppContext::get_instance().input_queue_service.get_value_column_index(
                                                p_dataset_->get_input_queue(), p_predicted_decon_queue->get_input_queue_column_name()),
                        0,  // Predicted index is 0 because only one column is reconstructed from a decon queue
                        score_metric_e::RMSE,
                        true);
            } catch (std::exception &ex) {
                LOG4_WARN("Exception during slide scoring: " << ex.what());
                column_mse = std::numeric_limits<double>::quiet_NaN();
            }

            LOG4_INFO("Slide score for column " << p_predicted_decon_queue->get_input_queue_column_name() << " is "
                                                << column_mse);
            if (!std::isnan(column_mse)) {
                final_tuned_mse += column_mse;
                validation_ct++;
            }
        }
        training_window.shift(advance_slides(validation_parameters_));
        validation_window.shift(advance_slides(validation_parameters_));
    }
    if (validation_ct > 0) final_tuned_mse /= double(validation_ct);
    else final_tuned_mse = std::numeric_limits<double>::max();

    //final_tuned_mse = std::numeric_limits<double>::quiet_NaN();
    LOG4_INFO(
            "Final score for dataset " << final_tuned_mse << " for validation range "
                                       << validation_parameters_.validation_range_ <<
                                       " " << validation_parameters_.validation_slide_count_ << " slides, "
                                       << validation_ct << " actual valid slides across all columns.");

    p_best_dataset->set_id(p_autotune_task_->get_result_dataset_id());
    p_best_dataset->set_ensembles(
            AppContext::get_instance().ensemble_service.init_ensembles_from_dataset(p_best_dataset));

    AppContext::get_instance().dataset_service.save(p_best_dataset);

    //set status done
    p_autotune_task_->set_mse(std::isnan(final_tuned_mse) ? -1. : final_tuned_mse);
    p_autotune_task_->set_done_time(bpt::second_clock::local_time());
    p_autotune_task_->set_status(2);
    p_autotune_task_->set_result_dataset_id(p_best_dataset->get_id());
    AppContext::get_instance().autotune_task_service.save(p_autotune_task_);
}


void CLI::process_prediction_task()
{
    p_prediction_task_->set_dataset_id(p_dataset_->get_id());
    p_prediction_task_->set_status(1);
    AppContext::get_instance().prediction_task_service.save(p_prediction_task_);
    auto prediction_period = bpt::time_period(
            p_prediction_task_->get_start_prediction_time(), p_prediction_task_->get_end_prediction_time());
    auto predicted_decon_queues =
            AppContext::get_instance().dataset_service.run_dataset_aux(
                    p_dataset_,
                    bpt::time_period(p_prediction_task_->get_start_train_time(),
                                     p_prediction_task_->get_end_train_time()),
                    prediction_period);
    auto p_orig_input_queue = p_dataset_->get_input_queue()->clone_empty();
    p_orig_input_queue->set_data(AppContext::get_instance().input_queue_service.load(
            p_dataset_->get_input_queue()->get_table_name(),
            p_prediction_task_->get_start_prediction_time(),
            p_prediction_task_->get_end_prediction_time()));

    double final_dataset_mse = 0.;
    size_t columns_ct = 0;
    for (const auto &p_predicted_decon_queue: predicted_decon_queues) {
        DataRow::container predicted_data = AppContext::get_instance().decon_queue_service.reconstruct(
                datarow_range(
                        p_predicted_decon_queue->get_data().begin(),
                        p_predicted_decon_queue->get_data().end(),
                        p_predicted_decon_queue->get_data()),
                p_dataset_->get_transformation_name(),
                p_dataset_->get_transformation_levels());

        const auto column_mse = svr::paramtune::score(
                p_orig_input_queue->get_data(),
                predicted_data,
                AppContext::get_instance().input_queue_service.get_value_column_index(
                                p_dataset_->get_input_queue(), p_predicted_decon_queue->get_input_queue_column_name()),
                0,  // Predicted index is 0 because only one input queue column is reconstructed from a decon queue
                score_metric_e::RMSE,
                true);

        LOG4_INFO(
                "Score for column " << p_predicted_decon_queue->get_input_queue_column_name() << " is " << column_mse);
        final_dataset_mse += column_mse;
        columns_ct++;
    }
    final_dataset_mse /= double(columns_ct);
    LOG4_INFO("Dataset " << this->p_dataset_->get_id() << " " << this->p_dataset_->get_dataset_name()
                         << " prediction score " << final_dataset_mse);
    p_prediction_task_->set_status(2);
    p_prediction_task_->set_mse(std::isnan(final_dataset_mse) ? -1. : final_dataset_mse);
    AppContext::get_instance().prediction_task_service.save(p_prediction_task_);
}

void CLI::process_scaling_factors_task()
{
    p_scaling_factors_task_->set_dataset_id(p_dataset_->get_id());
    p_scaling_factors_task_->set_status(1);
    AppContext::get_instance().scaling_factors_task_service.save(p_scaling_factors_task_);

    calculate_scaling_factors(p_dataset_, p_scaling_factors_task_->get_force_recalculate_scaling_factors());

    // Save the scaling factors task data
    LOG4_INFO("Dataset " << this->p_dataset_->get_id() << " " << this->p_dataset_->get_dataset_name()
                         << " has scaling factors calculated.");
    p_scaling_factors_task_->set_status(2);
    p_scaling_factors_task_->set_mse(1);
    AppContext::get_instance().scaling_factors_task_service.save(p_scaling_factors_task_);
    LOG4_INFO("End.");
}

namespace
{
std::vector<double> to_vector(const datamodel::SVRParameters_ptr &svr_parameters)
{
    std::vector<double> result;

    result.push_back(svr_parameters->get_svr_C());
    result.push_back(svr_parameters->get_svr_epsilon());
    result.push_back(svr_parameters->get_svr_kernel_param());
    result.push_back(svr_parameters->get_svr_kernel_param2());
    result.push_back(svr_parameters->get_svr_adjacent_levels_ratio());
    result.push_back(static_cast<double>(svr_parameters->get_lag_count()));

    return result;
}

}

/* TODO Refactor! */
void CLI::process_decrement_task()
{
    validation_parameters_t vp{validation_parameters_};
    vp.training_range_ = bpt::time_period(p_decrement_task_->get_start_train_time(),
                                          p_decrement_task_->get_end_train_time());
    vp.validation_range_ = bpt::time_period(p_decrement_task_->get_start_validation_time(),
                                            p_decrement_task_->get_end_validation_time());
    vp.bounds_ = std::vector<Bounds>(p_dataset_->get_transformation_levels());


    std::map<std::string, std::vector<std::vector<std::pair<size_t, double>>>> decremental_mses;

    // for each column
    for (auto column: p_dataset_->get_input_queue()->get_value_columns()) {
        auto key = std::make_pair(p_dataset_->get_input_queue()->get_table_name(), column);
        vp.column_name_ = column;
        decremental_mses[column].resize(p_dataset_->get_transformation_levels());

        // for each level
        for (size_t level = 0; level < p_dataset_->get_transformation_levels(); ++level) {
            vp.model_number_ = level;

            // do decrement
            for (bpt::ptime local_start_train_time = p_decrement_task_->get_start_train_time();
                 local_start_train_time < p_decrement_task_->get_end_train_time() - bpt::minutes(100);
                 local_start_train_time = local_start_train_time +
                                          date_time_string_to_seconds(p_decrement_task_->get_decrement_step())) {
                vp.training_range_ = bpt::time_period(local_start_train_time, p_decrement_task_->get_end_train_time());
                /* TODO Implement getting of data from database */
                datamodel::DeconQueue_ptr p_decon_queue;
                std::vector<datamodel::DeconQueue_ptr> aux_decon_queues;
                const datamodel::SVRParameters_ptr p_param_set = p_dataset_->get_ensemble_svr_parameters()[key][level];

                const double current_mse = loss(
                        to_vector(p_param_set),
                        p_dataset_,
                        vp,
                        score_metric_e::RMSE,
                        p_decon_queue,
                        aux_decon_queues);

                // save current MSE
                size_t decremental_distance = static_cast<size_t>(vp.training_range_.length().total_seconds()) /
                                              static_cast<size_t>(p_dataset_->get_input_queue()->get_resolution().total_seconds());

                decremental_mses[column][level].emplace_back(std::make_pair(decremental_distance, current_mse));
            }
        }
    }

    std::map<std::string, std::vector<size_t>> suggested_values;
    for (auto &column : p_dataset_->get_input_queue()->get_value_columns()) {
        for (auto &decremental_mse : decremental_mses[column]) {
            if (!decremental_mse.empty())
                suggested_values[column].push_back(
                        std::min_element(decremental_mse.begin(), decremental_mse.end(),
                                         [](const std::pair<size_t, double> &a,
                                            const std::pair<size_t, double> &b)
                                         { return a.second < b.second; })->first);
        }
    }

    // write mse_values to string
    std::string mse_values_json_string{"{"};
    for (auto &column : p_dataset_->get_input_queue()->get_value_columns()) {
        size_t model_number{0};
        mse_values_json_string.append("\"" + column + "\":{");

        for (auto &decremental_mse : decremental_mses[column]) {
            mse_values_json_string.append("\"model_" + std::to_string(model_number) + "\":{");

            for (auto &decrement : decremental_mse)
                mse_values_json_string.append("\"" + std::to_string(decrement.first) + "\":" +
                                              svr::common::to_string_with_precision(decrement.second) + ",");

            mse_values_json_string.pop_back(); // remove last ','
            mse_values_json_string.append("},");

            ++model_number;
        }

        mse_values_json_string.pop_back(); // remove last ','
        mse_values_json_string.append("},");
    }

    mse_values_json_string.pop_back(); // remove last ','
    mse_values_json_string.append("}");

    // write suggested values to string
    std::string suggested_values_json_string{"{"};

    for (auto &column : p_dataset_->get_input_queue()->get_value_columns()) {
        size_t model_number{0};
        suggested_values_json_string.append("\"" + column + "\":{");

        for (auto &sv : suggested_values[column]) {
            suggested_values_json_string.append("\"model_" + std::to_string(model_number) + "\":" +
                                                "\"" + std::to_string(sv) + "\",");
            ++model_number;
        }
        suggested_values_json_string.pop_back(); // remove last ','
        suggested_values_json_string.append("},");
    }

    suggested_values_json_string.pop_back(); // remove last ','
    suggested_values_json_string.append("}");


    // for test
    LOG4_INFO("Score values: " << mse_values_json_string);
    LOG4_INFO("Suggested score values: " << suggested_values_json_string);

    //write to database
    try {
        p_decrement_task_->set_dataset_id(p_dataset_->get_id());
        p_decrement_task_->set_status(1);
        AppContext::get_instance().decrement_task_service.save(p_decrement_task_);
        p_decrement_task_->set_parameters(p_dataset_->parameters_to_string());
        p_decrement_task_->set_values(mse_values_json_string);
        p_decrement_task_->set_suggested_value(suggested_values_json_string);
        p_decrement_task_->set_end_task_time(bpt::second_clock::local_time());

        p_decrement_task_->set_status(2);
        AppContext::get_instance().decrement_task_service.save(p_decrement_task_);
    }
    catch (std::exception &e) {
        LOG4_ERROR(e.what());
    }
}


int CLI::run_dataset()
{
    LOG4_DEBUG("Dataset to be executed: " << p_dataset_->to_string());
    // TODO: if the wavelet type of the dataset is different than the new requested wavelet type,
    // update the wavelet type and scaling factors.
    ResourceMeasure resource_measure;
    resource_measure.set_start_time();
    PROPS.set_autotune_running(true);

    if (p_autotune_task_)
        try {
            process_autotune_task();
        } catch (std::exception &e) {
            p_autotune_task_->set_status(3);
            AppContext::get_instance().autotune_task_service.save(p_autotune_task_);
            LOG4_ERROR(e.what());
        }
    if (p_decrement_task_) process_decrement_task();

    if (p_prediction_task_) process_prediction_task();

    if (p_scaling_factors_task_) process_scaling_factors_task();

    resource_measure.print_measure_info();

    return 0;
}

#endif