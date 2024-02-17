#include "TsDQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR(TsDQScalingFactorDAO, DQScalingFactorDAO)
{}

bigint TsDQScalingFactorDAO::get_next_id()
{
    return ts_call<bigint>(&DQScalingFactorDAO::get_next_id);
}


bool TsDQScalingFactorDAO::exists(const datamodel::DQScalingFactor_ptr& dq_scaling_factor)
{
    return ts_call<bool>(&DQScalingFactorDAO::exists, dq_scaling_factor);
}


int TsDQScalingFactorDAO::save(const datamodel::DQScalingFactor_ptr& scaling_task)
{
    return ts_call<int>(&DQScalingFactorDAO::save, scaling_task);
}


int TsDQScalingFactorDAO::remove(const datamodel::DQScalingFactor_ptr& scaling_task)
{
    return ts_call<int>(&DQScalingFactorDAO::remove, scaling_task);
}


svr::datamodel::dq_scaling_factor_container_t TsDQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    return ts_call<svr::datamodel::dq_scaling_factor_container_t>(&DQScalingFactorDAO::find_all_by_dataset_id, dataset_id);
}


}
}
