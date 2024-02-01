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


int TsSVRParametersDAO::save(const datamodel::SVRParameters_ptr& svr_parameters)
{
    return ts_call<int>(&SVRParametersDAO::save, svr_parameters);
}


int TsSVRParametersDAO::remove(const datamodel::SVRParameters_ptr& svr_parameters)
{
    return ts_call<int>(&SVRParametersDAO::remove, svr_parameters);
}


int TsSVRParametersDAO::remove_by_dataset_id(const bigint dataset_id)
{
    return ts_call<int>(&SVRParametersDAO::remove_by_dataset_id, dataset_id);
}


std::deque<datamodel::SVRParameters_ptr> TsSVRParametersDAO::get_all_svrparams_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::deque<datamodel::SVRParameters_ptr>>(&SVRParametersDAO::get_all_svrparams_by_dataset_id, dataset_id);
}

std::deque<datamodel::SVRParameters_ptr> TsSVRParametersDAO::get_svrparams(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level)
{
    return ts_call<std::deque<datamodel::SVRParameters_ptr>>(&SVRParametersDAO::get_svrparams, dataset_id, input_queue_column_name, decon_level);
}

size_t TsSVRParametersDAO::get_dataset_levels(const bigint dataset_id)
{
return ts_call<size_t>(&SVRParametersDAO::get_dataset_levels, dataset_id);
}

}}
