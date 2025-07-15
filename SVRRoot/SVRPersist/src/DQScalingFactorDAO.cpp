#include "DAO/DQScalingFactorDAO.hpp"
#include "PgDAO/PgDQScalingFactorDAO.hpp"
#include "AsyncDAO/AsyncDQScalingFactorDAO.hpp"
#include "ThreadSafeDAO/TsDQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

DQScalingFactorDAO *DQScalingFactorDAO::build(
        common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<DQScalingFactorDAO, PgDQScalingFactorDAO, AsyncDQScalingFactorDAO, TsDQScalingFactorDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

DQScalingFactorDAO::DQScalingFactorDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source) :
        AbstractDAO(tempus_config, data_source, "DQScalingFactorDAO.properties")
{}

}
}