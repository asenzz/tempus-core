#include "AutotuneTaskService.hpp"
#include <util/validation_utils.hpp>
#include "appcontext.hpp"

#include <model/AutotuneTask.hpp>
#include <DAO/AutotuneTaskDAO.hpp>

using svr::common::reject_nullptr;

namespace svr{
namespace business{

bool AutotuneTaskService::exists(const AutotuneTask_ptr &autotuneTask)
{
    reject_nullptr(autotuneTask);
    return autotuneTaskDao.exists(autotuneTask->get_id());
}

int AutotuneTaskService::save(AutotuneTask_ptr &autotuneTask)
{
    reject_nullptr(autotuneTask);

    if (autotuneTask->get_id() == 0)
        autotuneTask->set_id(autotuneTaskDao.get_next_id());

    return autotuneTaskDao.save(autotuneTask);
}

int AutotuneTaskService::remove(const AutotuneTask_ptr &autotuneTask)
{
    reject_nullptr(autotuneTask);

    return autotuneTaskDao.remove(autotuneTask);
}

AutotuneTask_ptr AutotuneTaskService::get_by_id(const bigint id)
{
    return autotuneTaskDao.get_by_id(id);
}

std::vector<AutotuneTask_ptr> AutotuneTaskService::find_all_by_dataset_id(const bigint dataset_id)
{
    return autotuneTaskDao.find_all_by_dataset_id(dataset_id);
}

} /* namespace business */
} /* namespace svr */
