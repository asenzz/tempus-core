#include <DAO/WScalingFactorDAO.hpp>
#include "PgDAO/PgWScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncWScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsWScalingFactorDAO.hpp"

namespace svr {
namespace dao {

WScalingFactorDAO *WScalingFactorDAO::build(svr::common::PropertiesFileReader &tempus_config,
                                            svr::dao::DataSource &data_source,
                                            svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<WScalingFactorDAO, PgWScalingFactorDAO, AsyncWScalingFactorDAO, TsWScalingFactorDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

WScalingFactorDAO::WScalingFactorDAO(svr::common::PropertiesFileReader &tempus_config,
                                     svr::dao::DataSource &data_source) :
        AbstractDAO(tempus_config, data_source, "WScalingFactorDAO.properties")
{}

}
}
