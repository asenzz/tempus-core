#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel { class WScalingFactor; using WScalingFactor_ptr = std::shared_ptr<WScalingFactor>; }}

namespace svr {
namespace dao {

class WScalingFactorDAO : public AbstractDAO
{
public:
    static WScalingFactorDAO *build(common::PropertiesFileReader &tempus_config, DataSource &data_source, common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit WScalingFactorDAO(common::PropertiesFileReader &tempus_config, DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const bigint id) = 0;

    virtual int save(const datamodel::WScalingFactor_ptr &p_iq_scaling_factor) = 0;

    virtual int remove(const datamodel::WScalingFactor_ptr &p_iq_scaling_factor) = 0;

    virtual std::deque<datamodel::WScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id) = 0;
};

}
}

