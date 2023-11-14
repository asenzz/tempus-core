#pragma once

#include "TsDaoBase.hpp"
#include <DAO/DecrementTaskDAO.hpp>

namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsDecrementTaskDAO, DecrementTaskDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const DecrementTask_ptr& decrementTask);
    virtual int remove(const DecrementTask_ptr& decrementTask);
    virtual DecrementTask_ptr get_by_id(const bigint id);
};

} // namespace dao
} // namespace svr
