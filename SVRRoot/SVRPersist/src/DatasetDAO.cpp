#include <DAO/DatasetDAO.hpp>
#include "PgDAO/PgDatasetDAO.hpp"
#include "AsyncDAO/AsyncDatasetDAO.hpp"
#include "ThreadSafeDAO/TsDatasetDAO.hpp"

namespace svr {
namespace dao {

DatasetDAO * DatasetDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DatasetDAO, PgDatasetDAO, AsyncDatasetDAO, TsDatasetDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}


DatasetDAO::DatasetDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:AbstractDAO(sqlProperties, dataSource, "DatasetDAO.properties")
{}

}
}
