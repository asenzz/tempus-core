#include "TsModelDAO.hpp"

namespace svr { namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsModelDAO, ModelDAO)
{}

bigint TsModelDAO::get_next_id()
{
    return ts_call<bigint>(&ModelDAO::get_next_id);
}


bool TsModelDAO::exists(const bigint model_id)
{
    return ts_call<bool>(&ModelDAO::exists, model_id);
}


int TsModelDAO::save(const datamodel::Model_ptr& model)
{
    return ts_call<int>(&ModelDAO::save, model);
}


int TsModelDAO::remove(const datamodel::Model_ptr& model)
{
    return ts_call<int>(&ModelDAO::remove, model);
}


int TsModelDAO::remove_by_ensemble_id(const bigint ensemble_id)
{
    return ts_call<int>(&ModelDAO::remove_by_ensemble_id, ensemble_id);
}


datamodel::Model_ptr TsModelDAO::get_by_id(const bigint model_id)
{
    return ts_call<datamodel::Model_ptr>(&ModelDAO::get_by_id, model_id);
}


datamodel::Model_ptr TsModelDAO::get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level)
{
    return ts_call<datamodel::Model_ptr>(&ModelDAO::get_by_ensemble_id_and_decon_level, ensemble_id, decon_level);
}


std::deque<datamodel::Model_ptr> TsModelDAO::get_all_ensemble_models(const bigint ensemble_id)
{
    return ts_call<std::deque<datamodel::Model_ptr>>(&ModelDAO::get_all_ensemble_models, ensemble_id);
}

std::deque<datamodel::OnlineMIMOSVR_ptr> TsModelDAO::get_svr_by_model_id(const bigint model_id)
{
    return ts_call<std::deque<datamodel::OnlineMIMOSVR_ptr>>(&ModelDAO::get_svr_by_model_id, model_id);
}

}
}
