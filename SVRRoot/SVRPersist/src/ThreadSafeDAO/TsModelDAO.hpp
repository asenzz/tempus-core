#pragma once

#include "TsDaoBase.hpp"
#include <DAO/ModelDAO.hpp>


namespace svr { namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsModelDAO, ModelDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint model_id);
    virtual int save(const datamodel::Model_ptr& model);
    virtual int remove(const datamodel::Model_ptr& model);
    virtual int remove_by_ensemble_id(const bigint ensemble_id);
    virtual datamodel::Model_ptr get_by_id(const bigint model_id);
    virtual datamodel::Model_ptr get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level);
    virtual std::deque<datamodel::Model_ptr> get_all_ensemble_models(const bigint ensemble_id);
    virtual std::deque<OnlineMIMOSVR_ptr> get_svr_by_model_id(const bigint model_id);
};

}
}

using ModelDAO_ptr = std::shared_ptr<svr::dao::ModelDAO>;
