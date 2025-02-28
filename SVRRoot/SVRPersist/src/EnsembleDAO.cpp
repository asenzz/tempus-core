#include "DAO/EnsembleDAO.hpp"
#include "PgDAO/PgEnsembleDAO.hpp"
#include "AsyncDAO/AsyncEnsembleDAO.hpp"
#include "ThreadSafeDAO/TsEnsembleDAO.hpp"

namespace svr {
namespace dao {

EnsembleDAO *EnsembleDAO::build(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao)
{
    return AbstractDAO::build<EnsembleDAO, PgEnsembleDAO, AsyncEnsembleDAO, TsEnsembleDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

EnsembleDAO::EnsembleDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : AbstractDAO(tempus_config, data_source, "EnsembleDAO.properties")
{}

}
}
