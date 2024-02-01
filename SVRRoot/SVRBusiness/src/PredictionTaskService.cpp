#include "PredictionTaskService.hpp"
#include <util/ValidationUtils.hpp>
#include "appcontext.hpp"

#include "model/PredictionTask.hpp"
#include "DAO/PredictionTaskDAO.hpp"

using namespace svr::common;
using namespace svr::context;

namespace svr{
namespace business{

bool PredictionTaskService::exists(const PredictionTask_ptr &predictionTask)
{
    reject_nullptr(predictionTask);
    return predictionTaskDao.exists(predictionTask->get_id());
}

int PredictionTaskService::save(PredictionTask_ptr &predictionTask)
{
    reject_nullptr(predictionTask);

    return predictionTaskDao.save(predictionTask);
}

PredictionTask_ptr PredictionTaskService::get_by_id(const bigint id)
{
    return predictionTaskDao.get_by_id(id);
}

} /* namespace business */
} /* namespace svr */
