#include "DAO/WScalingFactorDAO.hpp"
#include "PgDAO/PgWScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncWScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsWScalingFactorDAO.hpp"

namespace svr {
namespace dao {

WScalingFactorDAO *WScalingFactorDAO::build(common::PropertiesReader &tempus_config,
                                            dao::DataSource &data_source,
                                            const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<WScalingFactorDAO, PgWScalingFactorDAO, AsyncWScalingFactorDAO, TsWScalingFactorDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

WScalingFactorDAO::WScalingFactorDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source) :
        AbstractDAO(tempus_config, data_source, "WScalingFactorDAO.properties")
{}

}
}
