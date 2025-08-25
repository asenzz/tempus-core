#pragma once

#include <common/types.hpp>
#include <memory>

namespace svr {
namespace dao {
class PredictionTaskDAO;
}

namespace datamodel {
class PredictionTask;
using PredictionTask_ptr = std::shared_ptr<datamodel::PredictionTask>;
} // namespace datamodel

namespace business {

class PredictionTaskService
{

    dao::PredictionTaskDAO& predictionTaskDao;

public:
    PredictionTaskService(dao::PredictionTaskDAO& predictionTaskDao) : predictionTaskDao(predictionTaskDao) {}

    bool exists(const datamodel::PredictionTask_ptr&);

    int save(datamodel::PredictionTask_ptr& predictionTask);

    datamodel::PredictionTask_ptr get_by_id(const bigint id);
};

using PredictionTaskService_ptr = std::shared_ptr<business::PredictionTaskService>;

} /* namespace business */
} /* namespace svr */
