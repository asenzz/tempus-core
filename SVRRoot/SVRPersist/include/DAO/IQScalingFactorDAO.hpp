#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel { class IQScalingFactor; using IQScalingFactor_ptr = std::shared_ptr<IQScalingFactor>; }}

namespace svr {
namespace dao {

class IQScalingFactorDAO : public AbstractDAO
{
public:
    static IQScalingFactorDAO *build(common::PropertiesFileReader &tempus_config,
                                     DataSource &data_source,
                                     common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit IQScalingFactorDAO(common::PropertiesFileReader &tempus_config,
                                DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const bigint id) = 0;

    virtual int save(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor) = 0;

    virtual int remove(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor) = 0;

    virtual std::deque<datamodel::IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id) = 0;
};

}
}

