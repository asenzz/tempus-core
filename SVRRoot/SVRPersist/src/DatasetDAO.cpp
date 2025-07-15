#include "DAO/DatasetDAO.hpp"
#include "PgDAO/PgDatasetDAO.hpp"
#include "AsyncDAO/AsyncDatasetDAO.hpp"
#include "ThreadSafeDAO/TsDatasetDAO.hpp"

namespace svr {
namespace dao {

DatasetDAO *
DatasetDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<DatasetDAO, PgDatasetDAO, AsyncDatasetDAO, TsDatasetDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}


DatasetDAO::DatasetDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : AbstractDAO(tempus_config, data_source, "DatasetDAO.properties")
{}

}
}
