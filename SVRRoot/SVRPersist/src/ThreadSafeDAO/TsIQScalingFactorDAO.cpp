#include "TsIQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR(TsIQScalingFactorDAO, IQScalingFactorDAO)
{}

bigint TsIQScalingFactorDAO::get_next_id()
{
    return ts_call<bigint>(&IQScalingFactorDAO::get_next_id);
}


bool TsIQScalingFactorDAO::exists(const bigint id)
{
    return ts_call<bool>(&IQScalingFactorDAO::exists, id);
}


int TsIQScalingFactorDAO::save(const datamodel::IQScalingFactor_ptr &scalingTask)
{
    return ts_call<int>(&IQScalingFactorDAO::save, scalingTask);
}


int TsIQScalingFactorDAO::remove(const datamodel::IQScalingFactor_ptr &scalingTask)
{
    return ts_call<int>(&IQScalingFactorDAO::remove, scalingTask);
}


std::deque<datamodel::IQScalingFactor_ptr> TsIQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::deque<datamodel::IQScalingFactor_ptr>>(&IQScalingFactorDAO::find_all_by_dataset_id, dataset_id);
}


}
}
