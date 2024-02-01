#include "TsPredictionTaskDAO.hpp"

namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsPredictionTaskDAO, PredictionTaskDAO)
{}

bigint TsPredictionTaskDAO::get_next_id()
{
    return ts_call<bigint>(&PredictionTaskDAO::get_next_id);
}


bool TsPredictionTaskDAO::exists(const bigint id)
{
    return ts_call<bool>(&PredictionTaskDAO::exists, id);
}


int TsPredictionTaskDAO::save(const PredictionTask_ptr& predictionTask)
{
    return ts_call<int>(&PredictionTaskDAO::save, predictionTask);
}


int TsPredictionTaskDAO::remove(const PredictionTask_ptr& predictionTask)
{
    return ts_call<int>(&PredictionTaskDAO::remove, predictionTask);
}


PredictionTask_ptr TsPredictionTaskDAO::get_by_id(const bigint id)
{
    return ts_call<PredictionTask_ptr>(&PredictionTaskDAO::get_by_id, id);
}


} /* namespace dao */
} /* namespace svr */
