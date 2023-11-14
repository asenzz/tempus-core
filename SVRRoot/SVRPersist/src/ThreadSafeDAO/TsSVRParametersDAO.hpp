#pragma once

#include "TsDaoBase.hpp"
#include <DAO/SVRParametersDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsSVRParametersDAO, SVRParametersDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const SVRParameters_ptr& svr_parameters);
    virtual int remove(const SVRParameters_ptr& svr_parameters);
    virtual int remove_by_dataset_id(const bigint dataset_id);

    virtual std::vector<SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id);
    virtual size_t get_dataset_levels(const bigint dataset_id);
};

}
}

using SVRParametersDAO_ptr = std::shared_ptr<svr::dao::SVRParametersDAO>;
