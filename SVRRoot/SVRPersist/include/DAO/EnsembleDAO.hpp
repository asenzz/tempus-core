#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel {
class Ensemble;
class Dataset;
class DeconQueue;
using Ensemble_ptr = std::shared_ptr<Ensemble>;
using Dataset_ptr = std::shared_ptr<Dataset>;
using DeconQueue_ptr = std::shared_ptr<DeconQueue>;
} }

namespace svr {
namespace dao {

class EnsembleDAO : public AbstractDAO
{
public:
    static EnsembleDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit EnsembleDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;

    virtual datamodel::Ensemble_ptr get_by_id(const bigint id) = 0;

    virtual bool exists(const bigint ensemble_id) = 0;
    virtual bool exists(const datamodel::Ensemble_ptr &ensemble) = 0;

    virtual datamodel::Ensemble_ptr get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset, const datamodel::DeconQueue_ptr& decon_queue) = 0;

    virtual std::deque<datamodel::Ensemble_ptr> find_all_ensembles_by_dataset_id(const bigint dataset_id) = 0;

    virtual int save(const datamodel::Ensemble_ptr &Ensemble) = 0;

    virtual int remove(const datamodel::Ensemble_ptr &Ensemble) = 0;
};

}
}

using EnsembleDAO_ptr = std::shared_ptr <svr::dao::EnsembleDAO>;
