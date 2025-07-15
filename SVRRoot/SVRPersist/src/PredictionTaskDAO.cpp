#include "DAO/PredictionTaskDAO.hpp"
#include "PgDAO/PgPredictionTaskDAO.hpp"
#include "AsyncDAO/AsyncPredictionTaskDAO.hpp"
#include "ThreadSafeDAO/TsPredictionTaskDAO.hpp"

namespace svr {
namespace dao {

PredictionTaskDAO * PredictionTaskDAO::build(common::PropertiesReader& tempus_config, dao::DataSource& data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<PredictionTaskDAO, PgPredictionTaskDAO, AsyncPredictionTaskDAO, TsPredictionTaskDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

PredictionTaskDAO::PredictionTaskDAO(common::PropertiesReader& tempus_config, dao::DataSource& data_source)
: AbstractDAO(tempus_config, data_source, "PredictionTaskDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */
