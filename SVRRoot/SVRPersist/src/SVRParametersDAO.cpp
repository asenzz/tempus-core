#include <DAO/SVRParametersDAO.hpp>
#include "PgDAO/PgSVRParametersDAO.hpp"
#include "AsyncDAO/AsyncSVRParametersDAO.hpp"
#include "ThreadSafeDAO/TsSVRParametersDAO.hpp"

namespace svr{
namespace dao{

SVRParametersDAO * SVRParametersDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<SVRParametersDAO, PgSVRParametersDAO, AsyncSVRParametersDAO, TsSVRParametersDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

SVRParametersDAO::SVRParametersDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "SVRParametersDAO.properties")
{}

}
}
