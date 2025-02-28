#pragma once

#include "DAO/DQScalingFactorDAO.hpp"
#include "model/DQScalingFactor.hpp"

namespace svr {
namespace dao {

class PgDQScalingFactorDAO : public DQScalingFactorDAO
{
public:
    explicit PgDQScalingFactorDAO(common::PropertiesReader& tempus_config, dao::DataSource& data_source);

    virtual bigint get_next_id();
    virtual bool exists(const datamodel::DQScalingFactor_ptr& p_dq_scaling_factor);
    virtual int save(const datamodel::DQScalingFactor_ptr& p_dq_scaling_factor);
    virtual int remove(const datamodel::DQScalingFactor_ptr& p_dqscaling_factor);
    virtual datamodel::dq_scaling_factor_container_t find_all_by_model_id(const bigint dataset_id);
};

}
}
