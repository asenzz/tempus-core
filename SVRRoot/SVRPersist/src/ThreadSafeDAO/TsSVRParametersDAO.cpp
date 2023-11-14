#include "TsSVRParametersDAO.hpp"

namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsSVRParametersDAO, SVRParametersDAO)
{}

bigint TsSVRParametersDAO::get_next_id()
{
    return ts_call<bigint>(&SVRParametersDAO::get_next_id);
}


bool TsSVRParametersDAO::exists(const bigint id)
{
    return ts_call<bool>(&SVRParametersDAO::exists, id);
}


int TsSVRParametersDAO::save(const SVRParameters_ptr& svr_parameters)
{
    return ts_call<int>(&SVRParametersDAO::save, svr_parameters);
}


int TsSVRParametersDAO::remove(const SVRParameters_ptr& svr_parameters)
{
    return ts_call<int>(&SVRParametersDAO::remove, svr_parameters);
}


int TsSVRParametersDAO::remove_by_dataset_id(const bigint dataset_id)
{
    return ts_call<int>(&SVRParametersDAO::remove_by_dataset_id, dataset_id);
}


std::vector<SVRParameters_ptr> TsSVRParametersDAO::get_all_svrparams_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::vector<SVRParameters_ptr>>(&SVRParametersDAO::get_all_svrparams_by_dataset_id, dataset_id);
}

size_t TsSVRParametersDAO::get_dataset_levels(const bigint dataset_id)
{
return ts_call<size_t>(&SVRParametersDAO::get_dataset_levels, dataset_id);
}

}}
