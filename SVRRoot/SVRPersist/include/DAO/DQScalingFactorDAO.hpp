#pragma once

#include <set>
#include "DAO/AbstractDAO.hpp"
#include "model/DQScalingFactor.hpp"

namespace svr {
namespace dao {

class DQScalingFactorDAO : public AbstractDAO
{
public:
    static DQScalingFactorDAO *build(
            common::PropertiesReader &sql_properties,
            dao::DataSource &data_source,
            const common::ConcreteDaoType dao_type,
            const bool use_threadsafe_dao);

    explicit DQScalingFactorDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual int save(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual int remove(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual datamodel::dq_scaling_factor_container_t find_all_by_model_id(const bigint model_id) = 0;
};

using ScalingFactorDAO_ptr = std::shared_ptr<dao::DQScalingFactorDAO>;

} // namespace dao
} // namespace svr