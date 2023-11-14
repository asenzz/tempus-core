#pragma once

#include "TsDaoBase.hpp"
#include <DAO/ModelDAO.hpp>


namespace svr { namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsModelDAO, ModelDAO)

    virtual bigint get_next_id();
    virtual bool exists(bigint model_id);
    virtual int save(const Model_ptr& model);
    virtual int remove(const Model_ptr& model);
    virtual int remove_by_ensemble_id(bigint ensemble_id);
    virtual Model_ptr get_by_id(bigint model_id);
    virtual Model_ptr get_by_ensemble_id_and_decon_level(bigint ensemble_id, size_t decon_level);
    virtual std::vector<Model_ptr> get_all_ensemble_models(bigint ensemble_id);
};

}
}

using ModelDAO_ptr = std::shared_ptr<svr::dao::ModelDAO>;
