#pragma once

#include "DAO/IQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

class PgIQScalingFactorDAO : public IQScalingFactorDAO
{
public:
    explicit PgIQScalingFactorDAO(common::PropertiesReader &tempus_config, DataSource &data_source);

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual int remove(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual std::deque<datamodel::IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
