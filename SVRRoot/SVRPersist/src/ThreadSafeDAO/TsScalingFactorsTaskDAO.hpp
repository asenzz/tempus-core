#pragma once

#include "TsDaoBase.hpp"
#include <DAO/ScalingFactorsTaskDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsScalingFactorsTaskDAO, ScalingFactorsTaskDAO)

    virtual bigint get_next_id();
    virtual bool exists(bigint id);
    virtual int save(const ScalingFactorsTask_ptr& scalingFactorsTask);
    virtual int remove(const ScalingFactorsTask_ptr& scalingFactorsTask);
    virtual ScalingFactorsTask_ptr get_by_id(bigint id);

};

} /* namespace dao */
} /* namespace svr */
