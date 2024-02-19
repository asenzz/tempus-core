#include <DAO/InputQueueDAO.hpp>
#include <model/InputQueue.hpp>
#include "PgDAO/PgInputQueueDAO.hpp"
#include "AsyncDAO/AsyncInputQueueDAO.hpp"
#include "ThreadSafeDAO/TsInputQueueDAO.hpp"

namespace svr {
namespace dao {

using svr::common::ConcreteDaoType;

InputQueueDAO *InputQueueDAO::build(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source, ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<InputQueueDAO, PgInputQueueDAO, AsyncInputQueueDAO, TsInputQueueDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

InputQueueDAO::InputQueueDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source)
        : AbstractDAO(sql_properties, data_source, "InputQueueDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */

