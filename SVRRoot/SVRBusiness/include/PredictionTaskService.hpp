#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class PredictionTaskDAO; }

namespace datamodel { 
class PredictionTask; 
using PredictionTask_ptr = std::shared_ptr<svr::datamodel::PredictionTask>;
}

namespace business {

class PredictionTaskService {

    svr::dao::PredictionTaskDAO & predictionTaskDao;

public:

    PredictionTaskService(svr::dao::PredictionTaskDAO &predictionTaskDao) :
        predictionTaskDao(predictionTaskDao) {}

    bool exists(const PredictionTask_ptr &);

    int save(PredictionTask_ptr& predictionTask);

    PredictionTask_ptr get_by_id(const bigint id);
};

using PredictionTaskService_ptr = std::shared_ptr<svr::business::PredictionTaskService>;

} /* namespace business */
} /* namespace svr */
