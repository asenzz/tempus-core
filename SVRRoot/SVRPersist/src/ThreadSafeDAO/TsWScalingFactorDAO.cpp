#include <deque>
#include "TsWScalingFactorDAO.hpp"
#include "DAO/WScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR(TsWScalingFactorDAO, WScalingFactorDAO)
{}

bigint TsWScalingFactorDAO::get_next_id()
{
    return ts_call<bigint>(&WScalingFactorDAO::get_next_id);
}


bool TsWScalingFactorDAO::exists(const bigint id)
{
    return ts_call<bool>(&WScalingFactorDAO::exists, id);
}


int TsWScalingFactorDAO::save(const datamodel::WScalingFactor_ptr &sf)
{
    return ts_call<int>(&WScalingFactorDAO::save, sf);
}


int TsWScalingFactorDAO::remove(const datamodel::WScalingFactor_ptr &sf)
{
    return ts_call<int>(&WScalingFactorDAO::remove, sf);
}


std::deque<datamodel::WScalingFactor_ptr> TsWScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::deque<datamodel::WScalingFactor_ptr>>(&WScalingFactorDAO::find_all_by_dataset_id, dataset_id);
}


}
}
