#include "TsAutotuneTaskDAO.hpp"

namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsAutotuneTaskDAO, AutotuneTaskDAO)
{}

bigint TsAutotuneTaskDAO::get_next_id()
{
    return ts_call<bigint>(&AutotuneTaskDAO::get_next_id);
}


bool TsAutotuneTaskDAO::exists(bigint id)
{
    return ts_call<bool>(&AutotuneTaskDAO::exists, id);
}


int TsAutotuneTaskDAO::save(const AutotuneTask_ptr& autotuneTask)
{
    return ts_call<int>(&AutotuneTaskDAO::save, autotuneTask);
}


int TsAutotuneTaskDAO::remove(const AutotuneTask_ptr& autotuneTask)
{
    return ts_call<int>(&AutotuneTaskDAO::remove, autotuneTask);
}


AutotuneTask_ptr TsAutotuneTaskDAO::get_by_id(bigint id)
{
    return ts_call<AutotuneTask_ptr>(&AutotuneTaskDAO::get_by_id, id);
}


std::vector<AutotuneTask_ptr> TsAutotuneTaskDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::vector<AutotuneTask_ptr>>(&AutotuneTaskDAO::find_all_by_dataset_id, dataset_id);
}


} /* namespace dao */
} /* namespace svr */
