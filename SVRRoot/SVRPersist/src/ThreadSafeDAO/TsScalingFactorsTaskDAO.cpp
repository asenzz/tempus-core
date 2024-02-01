#include "TsScalingFactorsTaskDAO.hpp"

namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsScalingFactorsTaskDAO, ScalingFactorsTaskDAO)
{}

bigint TsScalingFactorsTaskDAO::get_next_id()
{
    return ts_call<bigint>(&ScalingFactorsTaskDAO::get_next_id);
}


bool TsScalingFactorsTaskDAO::exists(const bigint id)
{
    return ts_call<bool>(&ScalingFactorsTaskDAO::exists, id);
}


int TsScalingFactorsTaskDAO::save(const ScalingFactorsTask_ptr& scalingFactorsTask)
{
    return ts_call<int>(&ScalingFactorsTaskDAO::save, scalingFactorsTask);
}


int TsScalingFactorsTaskDAO::remove(const ScalingFactorsTask_ptr& scalingFactorsTask)
{
    return ts_call<int>(&ScalingFactorsTaskDAO::remove, scalingFactorsTask);
}


    ScalingFactorsTask_ptr TsScalingFactorsTaskDAO::get_by_id(const bigint id)
{
    return ts_call<ScalingFactorsTask_ptr>(&ScalingFactorsTaskDAO::get_by_id, id);
}


} /* namespace dao */
} /* namespace svr */
