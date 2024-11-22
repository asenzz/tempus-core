#pragma once

#include <DAO/WScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class PgWScalingFactorDAO : public WScalingFactorDAO
{
public:
    explicit PgWScalingFactorDAO(common::PropertiesFileReader &tempus_config, DataSource &data_source);

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const datamodel::WScalingFactor_ptr &p_iq_scaling_factor);

    virtual int remove(const datamodel::WScalingFactor_ptr &p_iq_scaling_factor);

    virtual std::deque<datamodel::WScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
