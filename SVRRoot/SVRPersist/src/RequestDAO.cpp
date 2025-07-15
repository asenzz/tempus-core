#include "DAO/RequestDAO.hpp"
#include "DAO/DataSource.hpp"
#include "PgDAO/PgRequestDAO.hpp"
#include "AsyncDAO/AsyncRequestDAO.hpp"
#include "ThreadSafeDAO/TsRequestDAO.hpp"

namespace svr {
namespace dao {

RequestDAO::RequestDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source) : AbstractDAO(tempus_config, data_source, "RequestDAO.properties")
{
}

RequestDAO *RequestDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<RequestDAO, PgRequestDAO, AsyncRequestDAO, TsRequestDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

}
}

