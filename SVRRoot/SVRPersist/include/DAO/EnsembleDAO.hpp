#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel {
class Ensemble;
class Dataset;
class DeconQueue;
} }
using Ensemble_ptr=std::shared_ptr<svr::datamodel::Ensemble>;
using Dataset_ptr=std::shared_ptr<svr::datamodel::Dataset>;
using DeconQueue_ptr=std::shared_ptr<svr::datamodel::DeconQueue>;

namespace svr {
namespace dao {

class EnsembleDAO : public AbstractDAO
{
public:
    static EnsembleDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit EnsembleDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;

    virtual Ensemble_ptr get_by_id(bigint id) = 0;

    virtual bool exists(bigint ensemble_id) = 0;
    virtual bool exists(const Ensemble_ptr &ensemble) = 0;

    virtual Ensemble_ptr get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue) = 0;

    virtual std::vector<Ensemble_ptr> find_all_ensembles_by_dataset_id(bigint dataset_id) = 0;

    virtual int save(const Ensemble_ptr &Ensemble) = 0;

    virtual int remove(const Ensemble_ptr &Ensemble) = 0;
};

}
}

using EnsembleDAO_ptr = std::shared_ptr <svr::dao::EnsembleDAO>;
