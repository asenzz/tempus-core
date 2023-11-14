#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class ScalingFactorsTaskDAO; } }

namespace svr { namespace datamodel { class ScalingFactorsTask; } }
using ScalingFactorsTask_ptr = std::shared_ptr<svr::datamodel::ScalingFactorsTask>;

namespace svr {
namespace business {


class ScalingFactorsTaskService {

    svr::dao::ScalingFactorsTaskDAO & scalingFactorsTaskDao;

public:

    ScalingFactorsTaskService(svr::dao::ScalingFactorsTaskDAO &scalingFactorsTaskDao) :
        scalingFactorsTaskDao(scalingFactorsTaskDao) {}

    bool exists(const ScalingFactorsTask_ptr &);

    int save(ScalingFactorsTask_ptr& scalingFactorsTask);

    ScalingFactorsTask_ptr get_by_id(bigint id);
};

} /* namespace business */
} /* namespace svr */

using ScalingFactorsTaskService_ptr = std::shared_ptr<svr::business::ScalingFactorsTaskService>;
