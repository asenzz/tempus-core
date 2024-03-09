#include <DAO/DatasetDAO.hpp>
#include "PgDAO/PgDatasetDAO.hpp"
#include "AsyncDAO/AsyncDatasetDAO.hpp"
#include "ThreadSafeDAO/TsDatasetDAO.hpp"

namespace svr {
namespace dao {

DatasetDAO *
DatasetDAO::build(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DatasetDAO, PgDatasetDAO, AsyncDatasetDAO, TsDatasetDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}


DatasetDAO::DatasetDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
        : AbstractDAO(tempus_config, data_source, "DatasetDAO.properties")
{}

}
}
