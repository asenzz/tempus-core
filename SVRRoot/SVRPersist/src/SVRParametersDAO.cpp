#include "DAO/SVRParametersDAO.hpp"
#include "PgDAO/PgSVRParametersDAO.hpp"
#include "AsyncDAO/AsyncSVRParametersDAO.hpp"
#include "ThreadSafeDAO/TsSVRParametersDAO.hpp"

namespace svr {
namespace dao {

SVRParametersDAO *SVRParametersDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<SVRParametersDAO, PgSVRParametersDAO, AsyncSVRParametersDAO, TsSVRParametersDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

SVRParametersDAO::SVRParametersDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : AbstractDAO(tempus_config, data_source, "SVRParametersDAO.properties")
{}

}
}
