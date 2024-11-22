#pragma once

#include "TsDaoBase.hpp"
#include <DAO/WScalingFactorDAO.hpp>


namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsWScalingFactorDAO, WScalingFactorDAO)

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const datamodel::WScalingFactor_ptr &sf);

    virtual int remove(const datamodel::WScalingFactor_ptr &sf);

    virtual std::deque<datamodel::WScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
