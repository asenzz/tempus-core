#include "DAO/DecrementTaskDAO.hpp"
#include "PgDAO/PgDecrementTaskDAO.hpp"
#include "AsyncDAO/AsyncDecrementTaskDAO.hpp"
#include "ThreadSafeDAO/TsDecrementTaskDAO.hpp"

namespace svr {
namespace dao {

DecrementTaskDAO * DecrementTaskDAO::build(common::PropertiesReader& tempus_config, dao::DataSource& data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<DecrementTaskDAO, PgDecrementTaskDAO, AsyncDecrementTaskDAO, TsDecrementTaskDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}


DecrementTaskDAO::DecrementTaskDAO(common::PropertiesReader& tempus_config, dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "DecrementTaskDAO.properties")
{}

} // namespace dao
} // namespace svr
