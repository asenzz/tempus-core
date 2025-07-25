#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr {

namespace datamodel {
class OnlineSVR;
using OnlineSVR_ptr = std::shared_ptr<OnlineSVR>;
class Model;
using Model_ptr = std::shared_ptr<Model>;
}

namespace dao {

class ModelDAO: public AbstractDAO
{
public:
    static ModelDAO *build(common::PropertiesReader& sql_properties, DataSource& data_source, common::ConcreteDaoType dao_type, bool use_threadsafe_dao);
    explicit ModelDAO(common::PropertiesReader& sql_properties, DataSource& data_source);
    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint model_id) = 0;
    virtual int save(const datamodel::Model_ptr& model) = 0;
    virtual int remove(const datamodel::Model_ptr& model) = 0;
    virtual int remove_by_ensemble_id(const bigint ensemble_id) = 0;
    virtual datamodel::Model_ptr get_by_id(const bigint model_id) = 0;
    virtual datamodel::Model_ptr get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level) = 0;
    virtual std::deque<datamodel::Model_ptr> get_all_ensemble_models(const bigint ensemble_id) = 0;
    virtual std::deque<datamodel::OnlineSVR_ptr> get_svr_by_model_id(const bigint model_id) = 0;
};

using ModelDAO_ptr = std::shared_ptr<dao::ModelDAO>;

}
}