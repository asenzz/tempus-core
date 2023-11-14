#pragma once

#include "TsDaoBase.hpp"
#include <DAO/DQScalingFactorDAO.hpp>


namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsDQScalingFactorDAO, svr::dao::DQScalingFactorDAO)

    virtual bigint get_next_id();
    virtual bool exists(const DQScalingFactor_ptr& dq_scaling_factor);
    virtual int save(const DQScalingFactor_ptr& scalingTask);
    virtual int remove(const DQScalingFactor_ptr& scalingTask);
    virtual svr::datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id);
};

}
}
