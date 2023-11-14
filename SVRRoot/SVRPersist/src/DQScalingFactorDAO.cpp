#include <DAO/DQScalingFactorDAO.hpp>
#include "PgDAO/PgDQScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncDQScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsDQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DQScalingFactorDAO* DQScalingFactorDAO::build(svr::common::PropertiesFileReader& sqlProperties,
                                              svr::dao::DataSource& dataSource,
                                              svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DQScalingFactorDAO, PgDQScalingFactorDAO, AsyncDQScalingFactorDAO, TsDQScalingFactorDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

DQScalingFactorDAO::DQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties,
                                       svr::dao::DataSource& dataSource) :
    AbstractDAO(sqlProperties, dataSource, "DQScalingFactorDAO.properties")
{}

}
}
