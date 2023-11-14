#pragma once

#include "TsDaoBase.hpp"
#include <DAO/PredictionTaskDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsPredictionTaskDAO, PredictionTaskDAO)

    virtual bigint get_next_id();
    virtual bool exists(bigint id);
    virtual int save(const PredictionTask_ptr& predictionTask);
    virtual int remove(const PredictionTask_ptr& predictionTask);
    virtual PredictionTask_ptr get_by_id(bigint id);

};

} /* namespace dao */
} /* namespace svr */
