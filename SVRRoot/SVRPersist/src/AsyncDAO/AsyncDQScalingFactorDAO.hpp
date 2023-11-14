#pragma once

#include <DAO/DQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class AsyncDQScalingFactorDAO : public DQScalingFactorDAO
{
public:
    explicit AsyncDQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource);
    ~AsyncDQScalingFactorDAO();

    virtual bigint get_next_id();
    virtual bool exists(const DQScalingFactor_ptr& dq_scaling_factor);
    virtual int save(const DQScalingFactor_ptr& dq_scaling_factor);
    virtual int remove(const DQScalingFactor_ptr& dq_scaling_factor);
    virtual svr::datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id);

private:
    class AsyncImpl;
    AsyncImpl & pImpl;
};

}
}
