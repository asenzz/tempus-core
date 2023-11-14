#include <DAO/IQScalingFactorDAO.hpp>
#include "PgDAO/PgIQScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncIQScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsIQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

IQScalingFactorDAO* IQScalingFactorDAO::build(svr::common::PropertiesFileReader& sqlProperties,
                                              svr::dao::DataSource& dataSource,
                                              svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<IQScalingFactorDAO, PgIQScalingFactorDAO, AsyncIQScalingFactorDAO, TsIQScalingFactorDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

IQScalingFactorDAO::IQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties,
                                       svr::dao::DataSource& dataSource) :
    AbstractDAO(sqlProperties, dataSource, "IQScalingFactorDAO.properties")
{}

}
}
