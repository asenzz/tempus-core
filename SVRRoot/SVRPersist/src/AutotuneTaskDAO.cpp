#include <DAO/AutotuneTaskDAO.hpp>
#include "PgDAO/PgAutotuneTaskDAO.hpp"
#include "AsyncDAO/AsyncAutotuneTaskDAO.hpp"
#include "ThreadSafeDAO/TsAutotuneTaskDAO.hpp"

namespace svr {
namespace dao {

AutotuneTaskDAO * AutotuneTaskDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<AutotuneTaskDAO, PgAutotuneTaskDAO, AsyncAutotuneTaskDAO, TsAutotuneTaskDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}


AutotuneTaskDAO::AutotuneTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
:AbstractDAO(tempus_config, data_source, "AutotuneTaskDAO.properties")
{}


} /* namespace dao */
} /* namespace svr */
