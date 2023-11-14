#include <DAO/PredictionTaskDAO.hpp>
#include "PgDAO/PgPredictionTaskDAO.hpp"
#include "AsyncDAO/AsyncPredictionTaskDAO.hpp"
#include "ThreadSafeDAO/TsPredictionTaskDAO.hpp"

namespace svr {
namespace dao {

PredictionTaskDAO * PredictionTaskDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<PredictionTaskDAO, PgPredictionTaskDAO, AsyncPredictionTaskDAO, TsPredictionTaskDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

PredictionTaskDAO::PredictionTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "PredictionTaskDAO.properties")
{}

} /* namespace dao */
} /* namespace svr */
