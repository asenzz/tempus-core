#include "TsDecrementTaskDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsDecrementTaskDAO, DecrementTaskDAO)
{}

bigint TsDecrementTaskDAO::get_next_id()
{
    return ts_call<bigint>(&DecrementTaskDAO::get_next_id);
}


bool TsDecrementTaskDAO::exists(const bigint id)
{
    return ts_call<bool>(&DecrementTaskDAO::exists, id);
}


int TsDecrementTaskDAO::save(const DecrementTask_ptr& decrementTask)
{
    return ts_call<int>(&DecrementTaskDAO::save, decrementTask);
}


int TsDecrementTaskDAO::remove(const DecrementTask_ptr& decrementTask)
{
    return ts_call<int>(&DecrementTaskDAO::remove, decrementTask);
}


DecrementTask_ptr TsDecrementTaskDAO::get_by_id(const bigint id)
{
    return ts_call<DecrementTask_ptr>(&DecrementTaskDAO::get_by_id, id);
}


} // namespace dao
} // namespace svr
