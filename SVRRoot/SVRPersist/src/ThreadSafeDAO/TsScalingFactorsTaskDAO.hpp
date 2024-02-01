#pragma once

#include "TsDaoBase.hpp"
#include <DAO/ScalingFactorsTaskDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsScalingFactorsTaskDAO, ScalingFactorsTaskDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const ScalingFactorsTask_ptr& scalingFactorsTask);
    virtual int remove(const ScalingFactorsTask_ptr& scalingFactorsTask);
    virtual ScalingFactorsTask_ptr get_by_id(const bigint id);

};

} /* namespace dao */
} /* namespace svr */
