#pragma once

#include "DAO/WScalingFactorDAO.hpp"

namespace svr {
namespace dao {

class AsyncWScalingFactorDAO : public WScalingFactorDAO
{
public:
    explicit AsyncWScalingFactorDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source);

    ~AsyncWScalingFactorDAO();

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const datamodel::WScalingFactor_ptr &p_w_scaling_factor);

    virtual int remove(const datamodel::WScalingFactor_ptr &p_w_scaling_factor);

    virtual std::deque<datamodel::WScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

private:
    struct AsyncImpl;
    AsyncImpl &pImpl;
};

}
}
