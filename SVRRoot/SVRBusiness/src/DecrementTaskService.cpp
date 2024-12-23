#include "DecrementTaskService.hpp"
#include <DAO/DecrementTaskDAO.hpp>
#include <model/DecrementTask.hpp>
#include <util/validation_utils.hpp>

#include <appcontext.hpp>

using namespace svr::common;
using namespace svr::context;

namespace svr {
namespace business {

bool DecrementTaskService::exists(const DecrementTask_ptr& decrementTask)
{
    REJECT_NULLPTR(decrementTask);
    return decrementTaskDao.exists(decrementTask->get_id());
}

int DecrementTaskService::save(const DecrementTask_ptr& decrementTask)
{
    REJECT_NULLPTR(decrementTask);

    return decrementTaskDao.save(decrementTask);
}

int DecrementTaskService::remove(const DecrementTask_ptr& decrementTask)
{
    REJECT_NULLPTR(decrementTask);

    return decrementTaskDao.remove(decrementTask);
}

DecrementTask_ptr DecrementTaskService::get_by_id(const bigint id)
{
    return decrementTaskDao.get_by_id(id);
}

} // namespace business
} // namespace svr
