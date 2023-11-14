#include <DAO/InputQueueDAO.hpp>
#include <model/InputQueue.hpp>
#include "PgDAO/PgInputQueueDAO.hpp"
#include "AsyncDAO/AsyncInputQueueDAO.hpp"
#include "ThreadSafeDAO/TsInputQueueDAO.hpp"

namespace svr {
namespace dao {

using svr::common::ConcreteDaoType;

InputQueueDAO * InputQueueDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<InputQueueDAO, PgInputQueueDAO, AsyncInputQueueDAO, TsInputQueueDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

InputQueueDAO::InputQueueDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source)
: AbstractDAO(sql_properties, data_source, "InputQueueDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */

