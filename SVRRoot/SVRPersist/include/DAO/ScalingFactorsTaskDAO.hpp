#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class ScalingFactorsTask; } }
using ScalingFactorsTask_ptr = std::shared_ptr<svr::datamodel::ScalingFactorsTask>;

namespace svr{
namespace dao{

class ScalingFactorsTaskDAO: public AbstractDAO
{
public:
    static ScalingFactorsTaskDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit ScalingFactorsTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const ScalingFactorsTask_ptr& scalingFactorsTask) = 0;
    virtual int remove(const ScalingFactorsTask_ptr& scalingFactorsTask) = 0;
    virtual ScalingFactorsTask_ptr get_by_id(const bigint id) = 0;

};

} /* namespace dao */
} /* namespace svr */

using ScalingFactorsTaskDAO_ptr = std::shared_ptr<svr::dao::ScalingFactorsTaskDAO>;
