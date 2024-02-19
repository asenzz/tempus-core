#include <DAO/DQScalingFactorDAO.hpp>
#include "PgDAO/PgDQScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncDQScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsDQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DQScalingFactorDAO* DQScalingFactorDAO::build(svr::common::PropertiesFileReader& tempus_config,
                                              svr::dao::DataSource& data_source,
                                              svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DQScalingFactorDAO, PgDQScalingFactorDAO, AsyncDQScalingFactorDAO, TsDQScalingFactorDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

DQScalingFactorDAO::DQScalingFactorDAO(svr::common::PropertiesFileReader& tempus_config,
                                       svr::dao::DataSource& data_source) :
    AbstractDAO(tempus_config, data_source, "DQScalingFactorDAO.properties")
{}

}
}
