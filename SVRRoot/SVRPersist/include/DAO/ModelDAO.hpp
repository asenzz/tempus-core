#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel { class Model; }}
using Model_ptr = std::shared_ptr<svr::datamodel::Model>;

namespace svr { namespace dao {

class ModelDAO: public AbstractDAO
{
public:
    static ModelDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);
    explicit ModelDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    virtual bigint get_next_id() = 0;
    virtual bool exists(bigint model_id) = 0;
    virtual int save(const Model_ptr& model) = 0;
    virtual int remove(const Model_ptr& model) = 0;
    virtual int remove_by_ensemble_id(bigint ensemble_id) = 0;
    virtual Model_ptr get_by_id(bigint model_id) = 0;
    virtual Model_ptr get_by_ensemble_id_and_decon_level(bigint ensemble_id, size_t decon_level) = 0;
    virtual std::vector<Model_ptr> get_all_ensemble_models(bigint ensemble_id) = 0;
};

}
}

using ModelDAO_ptr = std::shared_ptr<svr::dao::ModelDAO>;
