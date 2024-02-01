#include "ScalingFactorsTaskService.hpp"
#include <util/ValidationUtils.hpp>
#include "appcontext.hpp"

#include "model/ScalingFactorsTask.hpp"
#include "DAO/ScalingFactorsTaskDAO.hpp"

using namespace svr::common;
using namespace svr::context;

namespace svr{
namespace business{

bool ScalingFactorsTaskService::exists(const ScalingFactorsTask_ptr &scalingFactorsTask)
{
    reject_nullptr(scalingFactorsTask);
    return scalingFactorsTaskDao.exists(scalingFactorsTask->get_id());
}

int ScalingFactorsTaskService::save(ScalingFactorsTask_ptr &scalingFactorsTask)
{
    reject_nullptr(scalingFactorsTask);

    return scalingFactorsTaskDao.save(scalingFactorsTask);
}

ScalingFactorsTask_ptr ScalingFactorsTaskService::get_by_id(const bigint id)
{
    return scalingFactorsTaskDao.get_by_id(id);
}

} /* namespace business */
} /* namespace svr */
