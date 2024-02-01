#ifndef TWEAKINGPARAMS_H
#define TWEAKINGPARAMS_H

#include <vector>
#include <future>
#include "optimizer.hpp"
#include "model/Dataset.hpp"
#include "model/InputQueue.hpp"
#include "util/ResourceMeasure.hpp"


namespace svr {
namespace paramtune {

using namespace svr::datamodel;


#define DEFAULT_SCORE_METRIC (score_metric_e::RMSE)
#define LOSS_WORSE_SCORE (std::numeric_limits<double>::max())


enum class sliding_direction_e : size_t {forward = 0, backward = 1};
enum class score_metric_e : size_t {RMSE = 0, MAE = 1, CRMSE = 2, CMAE = 3};

typedef std::map<std::pair<std::string, std::string>, std::vector<double>> best_mse_t;

struct validation_parameters_t
{
    bpt::time_period training_range_ {bpt::time_period(bpt::ptime(), bpt::ptime())};
    bpt::time_period validation_range_ {bpt::time_period(bpt::ptime(), bpt::ptime())};

    sliding_direction_e sliding_direction_ {sliding_direction_e::forward};
    size_t validation_slide_count_ {1};
    bpt::seconds validation_slide_period_sec_ {0};

    size_t best_points_count_{1};
    size_t model_number_{0};

    std::string column_name_{""};

    std::vector<Bounds> bounds_;
};

using ValidationParameters_ptr = std::shared_ptr<validation_parameters_t>;

double loss(
        const std::vector<double> &parameters,
        datamodel::Dataset_ptr &p_dataset,
        validation_parameters_t &validation_parameters,
        const score_metric_e score_metric,
        const datamodel::DeconQueue_ptr &p_decon_queue,
        const std::vector<datamodel::DeconQueue_ptr> &aux_decon_queues);
#if 0
double score(
        const svr::datamodel::vektor<double> &reference,
        const svr::datamodel::vektor<double> &predicted,
        const score_metric_e score_metric,
        bool print = false);
#endif

double score(const svr::datamodel::DataRow::container &ethalon, const svr::datamodel::DataRow::container &predicted,
             const size_t ethalon_column_idx, const size_t predicted_column_idx, const score_metric_e score_metric,
             bool print = false);

datamodel::DeconQueue_ptr get_decon_queue_column(
        const std::vector<datamodel::DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name);

void prepare_datasets_tweaking(
        const datamodel::Dataset_ptr &current_dataset, std::vector<datamodel::Dataset_ptr> &datasets_for_tweaking);

datamodel::InputQueue_ptr get_input_data(
        datamodel::InputQueue_ptr p_input_queue,
        const size_t frame_length,
        const size_t max_lag_count,
        const bpt::time_period &needed_data_time_period);

datamodel::InputQueue_ptr prepare_input_data(
        const datamodel::Dataset_ptr &p_dataset,
        const size_t frame_length,
        const validation_parameters_t &validation_parameters_t, 
        const size_t max_lag_count,
        datamodel::InputQueue_ptr &p_input_queue); /* out */

bpt::time_period get_full_time_span(
        const svr::paramtune::validation_parameters_t &validation_parameters_t,
        const bpt::time_duration &slide_duration);

std::pair<std::string, std::pair<double, datamodel::SVRParameters_ptr>> tune_column(const datamodel::Dataset_ptr &p_current_dataset, const validation_parameters_t &local_validation_parameters,
                 const svr::optimizer::NM_parameters &nm_parameters, const svr::optimizer::PSO_parameters &pso_parameters,
                 const datamodel::DeconQueue_ptr &p_decon_queue, const std::vector<datamodel::DeconQueue_ptr> &aux_decon_queues);

void prepare_decon_data(datamodel::Dataset_ptr &p_dataset, const datamodel::InputQueue_ptr &p_input_queue,
                   std::vector<datamodel::DeconQueue_ptr> &decon_queues, const bool force_recalculate);

void tune_transformation(datamodel::Dataset_ptr &p_current_dataset, const std::vector<std::vector<size_t>> &svr_kernel_types_range,
                         const validation_parameters_t &validation_parameters, const svr::optimizer::NM_parameters &nm_parameters,
                         const svr::optimizer::PSO_parameters &pso_parameters, best_mse_t &best_mse, datamodel::Dataset_ptr &p_best_dataset,
                         const datamodel::InputQueue_ptr &p_input_queue);

void calculate_scaling_factors(datamodel::Dataset_ptr p_current_dataset_, bool force_recalculate);

datamodel::Dataset_ptr tune_dataset(
        const datamodel::Dataset_ptr &p_dataset,
        const std::vector<size_t> &transformation_levels_range,
        const std::vector<std::string> &transformation_names_range,
        const std::vector<std::vector<size_t>> &svr_kernel_types_range,
        const svr::paramtune::validation_parameters_t &validation_parameters,
        const svr::optimizer::NM_parameters &nm_parameters,
        const svr::optimizer::PSO_parameters &pso_parameters);

datamodel::Dataset_ptr tune_interchangable_level(
        const datamodel::Dataset_ptr &p_dataset,
        const boost::posix_time::time_period &training_range,
        const boost::posix_time::time_period &prediction_range,
        bool divide_lower_from_upper = false);

std::pair<std::map<std::string, double>, std::map<std::string, datamodel::SVRParameters_ptr>> tune_level(
        const datamodel::Dataset_ptr &p_current_dataset,
        const svr::paramtune::validation_parameters_t &validation_params,
        const svr::optimizer::NM_parameters &nm_parameters,
        const svr::optimizer::PSO_parameters &pso_parameters,
        const datamodel::InputQueue_ptr &p_input_queue,
        const std::vector<datamodel::DeconQueue_ptr> &decon_queues,
        const std::vector<datamodel::DeconQueue_ptr> &aux_decon_queues);


} //paramtune
} //svr


#endif // TWEAKINGPARAMS_H
