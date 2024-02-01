#include "IQScalingFactorService.hpp"
#include <DAO/IQScalingFactorDAO.hpp>
#include "model/InputQueue.hpp"

using namespace svr::common;

namespace svr {
namespace business {

bool IQScalingFactorService::exists(const IQScalingFactor_ptr& p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.exists(p_iq_scaling_factor->get_id());
}

int IQScalingFactorService::save(const IQScalingFactor_ptr& p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.save(p_iq_scaling_factor);
}

int IQScalingFactorService::remove(const IQScalingFactor_ptr& p_iq_scaling_factor)
{
    return iq_scaling_factor_dao_.remove(p_iq_scaling_factor);
}

std::deque<IQScalingFactor_ptr> IQScalingFactorService::find_all_by_dataset_id(const bigint dataset_id)
{
    return iq_scaling_factor_dao_.find_all_by_dataset_id(dataset_id);
}

std::deque<IQScalingFactor_ptr> IQScalingFactorService::calculate(const datamodel::InputQueue_ptr &p_input_queue, const size_t dataset_id, const double alpha)
{
    if(p_input_queue->get_data().empty() == true)
    {
        LOG4_ERROR("InputQueue is empty");
        return std::deque<IQScalingFactor_ptr>();
    }

    std::deque<double> iq_tick_volume;
    for(auto& row : p_input_queue->get_data())
        iq_tick_volume.push_back(row->get_tick_volume());

    std::sort(iq_tick_volume.begin(), iq_tick_volume.end());
    const size_t pos = std::round(iq_tick_volume.size() * (1.0 - alpha));

    std::deque<IQScalingFactor_ptr> result;

    result.push_back(std::make_shared<svr::datamodel::IQScalingFactor>(
                         0, dataset_id, p_input_queue->get_table_name(), iq_tick_volume[pos]));

    return result;
}

void IQScalingFactorService::scale(const datamodel::Dataset_ptr& p_dataset, const bool unscale)
{
    if(p_dataset->get_input_queue()->get_data().empty())
    {
        LOG4_ERROR("InputQueue is empty");
        return;
    }

    if(p_dataset->get_iq_scaling_factors().empty())
    {
        LOG4_WARN("IQScalingFactors is empty");

        p_dataset->set_iq_scaling_factors(find_all_by_dataset_id(p_dataset->get_id()));

        if(p_dataset->get_iq_scaling_factors().empty() == true)
        {
            LOG4_INFO("Calculating scaling factors...");

            p_dataset->set_iq_scaling_factors(calculate(p_dataset->get_input_queue(), p_dataset->get_id()));

            for (auto& scaling_factor : p_dataset->get_iq_scaling_factors())
                save(scaling_factor);
        }
    }

    const double scaling_factor  = unscale
        ? 1 / p_dataset->get_iq_scaling_factors()[0]->get_scaling_factor()
        : p_dataset->get_iq_scaling_factors()[0]->get_scaling_factor();

    for(auto& datarow_pair : p_dataset-> get_input_queue()->get_data())
        datarow_pair->set_tick_volume(datarow_pair->get_tick_volume() * scaling_factor);
}

}
}
