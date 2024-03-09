#include <string>
#include "common/constants.hpp"
#include "common/parallelism.hpp"
#include "util/time_utils.hpp"
#include "DAO/IQScalingFactorDAO.hpp"
#include "model/InputQueue.hpp"
#include "model/Dataset.hpp"
#include "IQScalingFactorService.hpp"


namespace svr {
namespace business {

const std::function<double(double)> IQScalingFactorService::C_default_scaler = [](const double v) -> double { return v; };

bool IQScalingFactorService::exists(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.exists(p_iq_scaling_factor->get_id());
}

int IQScalingFactorService::save(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.save(p_iq_scaling_factor);
}

int IQScalingFactorService::remove(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.remove(p_iq_scaling_factor);
}

std::deque<datamodel::IQScalingFactor_ptr> IQScalingFactorService::find_all_by_dataset_id(const bigint dataset_id)
{
    return iq_scaling_factor_dao_.find_all_by_dataset_id(dataset_id);
}

std::deque<datamodel::IQScalingFactor_ptr> IQScalingFactorService::calculate(const datamodel::InputQueue &input_queue, const size_t dataset_id, const size_t tail)
{
    if (input_queue.get_data().empty() || input_queue.get_value_columns().empty()) {
        LOG4_ERROR("Input queue " << input_queue << " is empty.");
        return std::deque<datamodel::IQScalingFactor_ptr>();
    }

    const auto iter_start = tail > input_queue.size() ? input_queue.begin() : (input_queue.get_data().rbegin() + tail).base();
    const auto columns_ct = input_queue.get_value_columns().size();
    const size_t row_ct = std::distance(iter_start, input_queue.end());

    LOG4_DEBUG("Calculating input queue " << input_queue.get_table_name() << ", dataset " << dataset_id << " scaling factors on last " << row_ct << " values.");

    arma::mat iq_values(row_ct, columns_ct);
#pragma omp parallel for collapse(2) num_threads(adj_threads(row_ct * columns_ct)) schedule(static, 1 + (row_ct * columns_ct) / std::thread::hardware_concurrency())
    for (size_t r = 0; r < row_ct; ++r)
        for (size_t c = 0; c < columns_ct; ++c)
            iq_values(r, c) = (**(iter_start + r))[c];

    const arma::rowvec dc_offset = arma::mean(iq_values);
    LOG4_TRACE("DC offsets " << dc_offset);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(iq_values.n_cols))
    for (size_t c = 0; c < iq_values.n_cols; ++c)
        iq_values.col(c) -= dc_offset[c];
    iq_values = arma::median(arma::abs(iq_values)) / common::C_input_obseg_labels;
    LOG4_TRACE("Scaling factors " << iq_values);
    std::deque<datamodel::IQScalingFactor_ptr> result(columns_ct);
#pragma omp parallel for num_threads(adj_threads(columns_ct)) schedule(static, 1)
    for (size_t c = 0; c < columns_ct; ++c)
        result[c] = ptr<svr::datamodel::IQScalingFactor>(0, dataset_id, input_queue.get_table_name(), input_queue.get_value_column(c), iq_values[c], dc_offset[c]);

    return result;
}


bool IQScalingFactorService::check(const std::deque<datamodel::IQScalingFactor_ptr> &iqsf, const std::deque<std::string> &value_columns)
{
    std::deque<bool> present(value_columns.size(), false);
    OMP_LOCK(present_l);
#pragma omp parallel for num_threads(adj_threads(value_columns.size() * iqsf.size())) collapse(2)
    for (const auto &sf: iqsf)
        for (size_t i = 0; i < value_columns.size(); ++i)
            if (sf->get_input_queue_column_name() == value_columns[i]
                && std::isnormal(sf->get_scaling_factor())
                && common::isnormalz(sf->get_dc_offset())) {
                omp_set_lock(&present_l);
                present[i] = true;
                omp_unset_lock(&present_l);
            }
    return std::all_of(std::execution::par_unseq, present.begin(), present.end(), [](const auto &el) { return el; });
}


void IQScalingFactorService::prepare(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const bool save_factors)
{
    if (check(dataset.get_iq_scaling_factors(input_queue), input_queue.get_value_columns())) return;

    dataset.set_iq_scaling_factors(iq_scaling_factor_dao_.find_all_by_dataset_id(dataset.get_id()), false);
    if (check(dataset.get_iq_scaling_factors(input_queue), input_queue.get_value_columns())) return;
    const auto resolution_ratio = dataset.get_input_queue()->get_resolution() / input_queue.get_resolution();
#ifdef INTEGRATION_TEST
    auto test_input_queue = input_queue;
    test_input_queue.get_data().erase(test_input_queue.end() - (INTEGRATION_TEST_VALIDATION_WINDOW - 1) * resolution_ratio, test_input_queue.end());
    PROFILE_EXEC_TIME(dataset.set_iq_scaling_factors(calculate(
            test_input_queue, dataset.get_id(), dataset.get_max_possible_residuals_length() + dataset.get_max_decrement() * resolution_ratio), true),
                      "Calculate input queue scaling factors for " << input_queue.get_table_name());
#else
    PROFILE_EXEC_TIME(dataset.set_iq_scaling_factors(calculate(
            input_queue, dataset.get_id(), dataset.get_max_possible_residuals_length() + dataset.get_max_decrement() * resolution_ratio), true),
        "Calculate input queue scaling factors for " << input_queue.get_table_name());
#endif

    if (!save_factors) return;
    const auto &dataset_iqsf = dataset.get_iq_scaling_factors(input_queue);
#pragma omp parallel for num_threads(adj_threads(dataset_iqsf.size())) schedule(static, 1)
    for (const auto &p_sf: dataset_iqsf) {
        if (exists(p_sf)) remove(p_sf);
        save(p_sf);
    }
}


void IQScalingFactorService::prepare(datamodel::Dataset &dataset, const bool save)
{
    prepare(dataset, *dataset.get_input_queue(), save);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_aux_input_queues().size()))
    for (const auto &p_aux_input: dataset.get_aux_input_queues()) prepare(dataset, *p_aux_input, save);
}

t_iqscaler
IQScalingFactorService::get_scaler(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const std::string &column_name)
{
    const auto p_iq_scaling_factor = dataset.get_iq_scaling_factor(input_queue.get_table_name(), column_name);
    if (!p_iq_scaling_factor) LOG4_THROW("Couldn't find scaling factors for " << input_queue.get_table_name() << ", " << column_name);
    const auto scaling_factor = p_iq_scaling_factor->get_scaling_factor();
    const auto dc_offset = p_iq_scaling_factor->get_dc_offset();
    LOG4_TRACE("Scaler for " << input_queue.get_table_name() << ", column " << column_name << ", factor " << scaling_factor << ", offset " << dc_offset);
    return [scaling_factor, dc_offset](const double v) -> double { return (v - dc_offset) / scaling_factor; };
}

t_iqscaler
IQScalingFactorService::get_unscaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name)
{
    const auto p_iq_scaling_factor = dataset.get_iq_scaling_factor(table_name, column_name);
    if (!p_iq_scaling_factor) LOG4_THROW("Can't find scaling factor for " << table_name << ", " << column_name);
    const auto scaling_factor = p_iq_scaling_factor->get_scaling_factor();
    const auto dc_offset = p_iq_scaling_factor->get_dc_offset();
    LOG4_TRACE("Unscaler for " << table_name << ", column " << column_name << ", factor " << scaling_factor << ", offset " << dc_offset);
    return [scaling_factor, dc_offset](const double v) -> double { return v * scaling_factor + dc_offset; };
}

}
}
