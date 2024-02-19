#include <DAO/DecrementTaskDAO.hpp>
#include "PgDAO/PgDecrementTaskDAO.hpp"
#include "AsyncDAO/AsyncDecrementTaskDAO.hpp"
#include "ThreadSafeDAO/TsDecrementTaskDAO.hpp"

namespace svr {
namespace dao {

DecrementTaskDAO * DecrementTaskDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DecrementTaskDAO, PgDecrementTaskDAO, AsyncDecrementTaskDAO, TsDecrementTaskDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}


DecrementTaskDAO::DecrementTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "DecrementTaskDAO.properties")
{}

} // namespace dao
} // namespace svr
