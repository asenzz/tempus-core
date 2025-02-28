#include "PgDAO/PgDeconQueueDAO.hpp"
#include "AsyncDAO/AsyncDeconQueueDAO.hpp"
#include "ThreadSafeDAO/TsDeconQueueDAO.hpp"

namespace svr {
namespace dao {

DeconQueueDAO *DeconQueueDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<DeconQueueDAO, PgDeconQueueDAO, AsyncDeconQueueDAO, TsDeconQueueDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

DeconQueueDAO::DeconQueueDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : AbstractDAO(tempus_config, data_source, "DeconQueueDAO.properties")
{}

}
}
