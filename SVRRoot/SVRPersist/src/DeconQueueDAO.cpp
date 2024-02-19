#include "PgDAO/PgDeconQueueDAO.hpp"
#include "AsyncDAO/AsyncDeconQueueDAO.hpp"
#include "ThreadSafeDAO/TsDeconQueueDAO.hpp"

using svr::common::ConcreteDaoType;

namespace svr{
namespace dao{

DeconQueueDAO * DeconQueueDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DeconQueueDAO, PgDeconQueueDAO, AsyncDeconQueueDAO, TsDeconQueueDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

DeconQueueDAO::DeconQueueDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "DeconQueueDAO.properties")
{}

}
}
