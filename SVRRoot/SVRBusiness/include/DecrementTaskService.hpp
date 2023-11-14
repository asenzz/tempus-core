#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class DecrementTaskDAO; } }

namespace svr { namespace datamodel { class DecrementTask; } }
using DecrementTask_ptr = std::shared_ptr<svr::datamodel::DecrementTask>;

namespace svr {
namespace business {

class DecrementTaskService
{
private:
    svr::dao::DecrementTaskDAO& decrementTaskDao;

public:
    DecrementTaskService(svr::dao::DecrementTaskDAO& decrementTaskDao) :
        decrementTaskDao(decrementTaskDao)
    {}

    bool exists(const DecrementTask_ptr& decrementTask);

    int save(const DecrementTask_ptr& decrementTask);
    int remove(const DecrementTask_ptr& decrementTask);

    DecrementTask_ptr get_by_id(const bigint id);
};

} // namespace business
} // namespace svr

using DecrementTaskService_ptr = std::shared_ptr<svr::business::DecrementTaskService>;
