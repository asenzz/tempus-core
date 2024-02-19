#include <DAO/ScalingFactorsTaskDAO.hpp>
#include "PgDAO/PgScalingFactorsTaskDAO.hpp"
#include "AsyncDAO/AsyncScalingFactorsTaskDAO.hpp"
#include "ThreadSafeDAO/TsScalingFactorsTaskDAO.hpp"

namespace svr {
namespace dao {

ScalingFactorsTaskDAO * ScalingFactorsTaskDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<ScalingFactorsTaskDAO, PgScalingFactorsTaskDAO, AsyncScalingFactorsTaskDAO, TsScalingFactorsTaskDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

ScalingFactorsTaskDAO::ScalingFactorsTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "ScalingFactorsTaskDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */
