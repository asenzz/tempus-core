#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class AutotuneTaskDAO; } }

namespace svr { namespace datamodel { class AutotuneTask; } }
using AutotuneTask_ptr = std::shared_ptr<svr::datamodel::AutotuneTask>;

namespace svr {
namespace business {


class AutotuneTaskService {

    svr::dao::AutotuneTaskDAO & autotuneTaskDao;

public:

    AutotuneTaskService(svr::dao::AutotuneTaskDAO &AutotuneTaskDao) :
        autotuneTaskDao(AutotuneTaskDao) {}

    bool exists(const AutotuneTask_ptr &);

    int save(AutotuneTask_ptr& AutotuneTask);
    int remove(const AutotuneTask_ptr& AutotuneTask);

    AutotuneTask_ptr get_by_id(const bigint id);
    std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id);

//    void set_parameters_to_dataset(AutotuneTask_ptr autotune_task, datamodel::Dataset_ptr dataset);

};

} /* namespace business */
} /* namespace svr */

using AutotuneTaskService_ptr = std::shared_ptr<svr::business::AutotuneTaskService>;
