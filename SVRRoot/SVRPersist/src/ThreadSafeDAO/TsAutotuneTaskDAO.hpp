#pragma once

#include "TsDaoBase.hpp"
#include <DAO/AutotuneTaskDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsAutotuneTaskDAO, AutotuneTaskDAO)

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const AutotuneTask_ptr& autotuneTask);
    virtual int remove(const AutotuneTask_ptr& autotuneTask);
    virtual AutotuneTask_ptr get_by_id(const bigint id);
    virtual std::vector<AutotuneTask_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

} /* namespace dao */
} /* namespace svr */
