#pragma once

#include <set>

#include <DAO/AbstractDAO.hpp>
#include <model/DQScalingFactor.hpp>

namespace svr {
namespace dao {


class DQScalingFactorDAO : public AbstractDAO
{
public:
    static DQScalingFactorDAO *build(
            svr::common::PropertiesFileReader &sql_properties,
            svr::dao::DataSource &data_source,
            svr::common::ConcreteDaoType dao_type,
            bool use_threadsafe_dao);

    explicit DQScalingFactorDAO(svr::common::PropertiesFileReader &sql_properties,
                                svr::dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual int save(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual int remove(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) = 0;

    virtual svr::datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id) = 0;
};

} // namespace dao
} // namespace svr

using ScalingFactorDAO_ptr = std::shared_ptr<svr::dao::DQScalingFactorDAO>;
