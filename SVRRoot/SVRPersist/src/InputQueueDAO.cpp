#include <DAO/InputQueueDAO.hpp>
#include <model/InputQueue.hpp>
#include "PgDAO/PgInputQueueDAO.hpp"
#include "AsyncDAO/AsyncInputQueueDAO.hpp"
#include "ThreadSafeDAO/TsInputQueueDAO.hpp"

namespace svr {
namespace dao {

InputQueueDAO *InputQueueDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<InputQueueDAO, PgInputQueueDAO, AsyncInputQueueDAO, TsInputQueueDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

InputQueueDAO::InputQueueDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source)
        : AbstractDAO(sql_properties, data_source, "InputQueueDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */

