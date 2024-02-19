#include <DAO/ModelDAO.hpp>
#include "PgDAO/PgModelDAO.hpp"
#include "AsyncDAO/AsyncModelDAO.hpp"
#include "ThreadSafeDAO/TsModelDAO.hpp"

namespace svr {
namespace dao {

ModelDAO * ModelDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<ModelDAO, PgModelDAO, AsyncModelDAO, TsModelDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

ModelDAO::ModelDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "ModelDAO.properties")
{}


}
}
