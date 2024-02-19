#pragma once

#include <DAO/IQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class PgIQScalingFactorDAO : public IQScalingFactorDAO
{
public:
    explicit PgIQScalingFactorDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source);

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual int remove(const IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
