#include <DAO/IQScalingFactorDAO.hpp>
#include "PgDAO/PgIQScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncIQScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsIQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

IQScalingFactorDAO* IQScalingFactorDAO::build(svr::common::PropertiesFileReader& tempus_config,
                                              svr::dao::DataSource& data_source,
                                              svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<IQScalingFactorDAO, PgIQScalingFactorDAO, AsyncIQScalingFactorDAO, TsIQScalingFactorDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

IQScalingFactorDAO::IQScalingFactorDAO(svr::common::PropertiesFileReader& tempus_config,
                                       svr::dao::DataSource& data_source) :
    AbstractDAO(tempus_config, data_source, "IQScalingFactorDAO.properties")
{}

}
}
