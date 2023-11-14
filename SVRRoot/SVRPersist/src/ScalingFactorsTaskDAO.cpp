#include <DAO/ScalingFactorsTaskDAO.hpp>
#include "PgDAO/PgScalingFactorsTaskDAO.hpp"
#include "AsyncDAO/AsyncScalingFactorsTaskDAO.hpp"
#include "ThreadSafeDAO/TsScalingFactorsTaskDAO.hpp"

namespace svr {
namespace dao {

ScalingFactorsTaskDAO * ScalingFactorsTaskDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<ScalingFactorsTaskDAO, PgScalingFactorsTaskDAO, AsyncScalingFactorsTaskDAO, TsScalingFactorsTaskDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

ScalingFactorsTaskDAO::ScalingFactorsTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "ScalingFactorsTaskDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */
