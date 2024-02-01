#pragma once

#include "TsDaoBase.hpp"
#include <DAO/IQScalingFactorDAO.hpp>


namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsIQScalingFactorDAO, IQScalingFactorDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const IQScalingFactor_ptr& scalingTask);
    virtual int remove(const IQScalingFactor_ptr& scalingTask);
    virtual std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
