#include "IQScalingFactorService.hpp"
#include <DAO/IQScalingFactorDAO.hpp>
#include <string>
#include "model/InputQueue.hpp"
#include "model/Dataset.hpp"


using namespace svr::common;


namespace svr {
namespace business {

bool IQScalingFactorService::exists(const IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.exists(p_iq_scaling_factor->get_id());
}

int IQScalingFactorService::save(const IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.save(p_iq_scaling_factor);
}

int IQScalingFactorService::remove(const IQScalingFactor_ptr &p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.remove(p_iq_scaling_factor);
}

std::deque<IQScalingFactor_ptr> IQScalingFactorService::find_all_by_dataset_id(const bigint dataset_id)
{
    return iq_scaling_factor_dao_.find_all_by_dataset_id(dataset_id);
}

std::deque<IQScalingFactor_ptr> IQScalingFactorService::calculate(const datamodel::InputQueue &input_queue, const size_t dataset_id, const size_t decrement)
{
    if (input_queue.get_data().empty() || input_queue.get_value_columns().empty()) {
        LOG4_ERROR("Input queue " << input_queue << " is empty.");
        return std::deque<IQScalingFactor_ptr>();
    }

    const auto iter_start = decrement > input_queue.size() ? input_queue.begin() : (input_queue.get_data().rbegin() + decrement).base();
    const auto columns_ct = input_queue.get_value_columns().size();
    const size_t row_ct = std::distance(iter_start, input_queue.end());
    arma::mat iq_values(row_ct, columns_ct);
#pragma omp parallel for collapse(2) num_threads(adj_threads(row_ct * columns_ct)) schedule(static, 1 + (row_ct * columns_ct) / std::thread::hardware_concurrency())
    for (size_t r = 0; r < row_ct; ++r)
        for (size_t c = 0; c < columns_ct; ++c)
            iq_values(r, c) = (iter_start + r)->get()->get_value(c);
    iq_values = arma::median(iq_values) / common::C_input_obseg_labels;
    std::deque<IQScalingFactor_ptr> result(columns_ct);
#pragma omp parallel for num_threads(adj_threads(columns_ct)) schedule(static, 1)
    for (size_t c = 0; c < columns_ct; ++c)
        result[c] = std::make_shared<svr::datamodel::IQScalingFactor>(0, dataset_id, input_queue.get_table_name(), input_queue.get_value_column(c), iq_values[c]);

    return result;
}


void IQScalingFactorService::prepare(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue)
{
    auto iqsf = dataset.get_iq_scaling_factors(input_queue);
    if (iqsf.size() == input_queue.get_value_columns().size()) return;

    dataset.set_iq_scaling_factors(iq_scaling_factor_dao_.find_all_by_dataset_id(dataset.get_id()));
    iqsf = dataset.get_iq_scaling_factors(input_queue);
    if (iqsf.size() == input_queue.get_value_columns().size()) return;

    iqsf = calculate(input_queue, dataset.get_id(), dataset.get_max_decrement());
    dataset.set_iq_scaling_factors(iqsf);

#pragma omp parallel for num_threads(adj_threads(iqsf.size())) schedule(static, 1)
    for (const auto &p_sf: iqsf) {
        if (exists(p_sf)) remove(p_sf);
        save(p_sf);
    }
}


void IQScalingFactorService::prepare(datamodel::Dataset &dataset)
{
    prepare(dataset, *dataset.get_input_queue());
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_aux_input_queues().size()))
    for (const auto &p_aux_input: dataset.get_aux_input_queues()) prepare(dataset, *p_aux_input);
}


t_iqscaler
IQScalingFactorService::get_scaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name)
{
    const auto p_iq_scaling_factor = dataset.get_iq_scaling_factor(table_name, column_name);
    if (!p_iq_scaling_factor) LOG4_THROW("Can't find scaling factor for " << table_name << ", " << column_name);
    const double scaling_factor = p_iq_scaling_factor->get_scaling_factor();
    return [scaling_factor](const double v) -> double { return v / scaling_factor; };
}

t_iqscaler
IQScalingFactorService::get_unscaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name)
{
    const auto p_iq_scaling_factor = dataset.get_iq_scaling_factor(table_name, column_name);
    if (!p_iq_scaling_factor) LOG4_THROW("Can't find scaling factor for " << table_name << ", " << column_name);
    const double scaling_factor = p_iq_scaling_factor->get_scaling_factor();
    return [scaling_factor](const double v) -> double { return v / scaling_factor; };
}


void IQScalingFactorService::scale(const datamodel::Dataset_ptr &p_dataset, const bool unscale)
{
    if (p_dataset->get_input_queue()->get_data().empty()) {
        LOG4_ERROR("InputQueue is empty");
        return;
    }

    if (p_dataset->get_iq_scaling_factors().empty()) {
        LOG4_WARN("IQScalingFactors is empty");

        p_dataset->set_iq_scaling_factors(find_all_by_dataset_id(p_dataset->get_id()));

        if (p_dataset->get_iq_scaling_factors().empty() == true) {
            LOG4_INFO("Calculating scaling factors...");

            p_dataset->set_iq_scaling_factors(calculate(*p_dataset->get_input_queue(), p_dataset->get_id(), std::numeric_limits<size_t>::max()));

            for (auto &scaling_factor: p_dataset->get_iq_scaling_factors())
                save(scaling_factor);
        }
    }

    const double scaling_factor = unscale
                                  ? 1 / p_dataset->get_iq_scaling_factors()[0]->get_scaling_factor()
                                  : p_dataset->get_iq_scaling_factors()[0]->get_scaling_factor();

    for (auto &datarow_pair: p_dataset->get_input_queue()->get_data())
        datarow_pair->set_tick_volume(datarow_pair->get_tick_volume() * scaling_factor);
}

}
}
