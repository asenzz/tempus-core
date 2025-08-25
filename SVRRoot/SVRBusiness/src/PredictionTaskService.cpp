#include "PredictionTaskService.hpp"
#include "DAO/PredictionTaskDAO.hpp"
#include "appcontext.hpp"
#include "model/PredictionTask.hpp"
#include "util/validation_utils.hpp"

namespace svr {
namespace business {

bool PredictionTaskService::exists(const datamodel::PredictionTask_ptr& predictionTask)
{
    REJECT_NULLPTR(predictionTask);
    return predictionTaskDao.exists(predictionTask->get_id());
}

int PredictionTaskService::save(datamodel::PredictionTask_ptr& predictionTask)
{
    REJECT_NULLPTR(predictionTask);
    return predictionTaskDao.save(predictionTask);
}

datamodel::PredictionTask_ptr PredictionTaskService::get_by_id(const bigint id) { return predictionTaskDao.get_by_id(id); }

} /* namespace business */
} /* namespace svr */
