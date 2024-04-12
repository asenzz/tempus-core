#pragma once

#include <DAO/DQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class AsyncDQScalingFactorDAO : public DQScalingFactorDAO
{
public:
    explicit AsyncDQScalingFactorDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source);
    ~AsyncDQScalingFactorDAO();

    virtual bigint get_next_id();
    virtual bool exists(const datamodel::DQScalingFactor_ptr& dq_scaling_factor);
    virtual int save(const datamodel::DQScalingFactor_ptr& dq_scaling_factor);
    virtual int remove(const datamodel::DQScalingFactor_ptr& dq_scaling_factor);
    virtual svr::datamodel::dq_scaling_factor_container_t find_all_by_model_id(const bigint model_id);

private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}
