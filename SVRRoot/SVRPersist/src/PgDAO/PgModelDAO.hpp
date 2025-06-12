#ifndef PGMODELDAO_HPP
#define PGMODELDAO_HPP

#include "DAO/ModelDAO.hpp"
#include "onlinesvr.hpp"

namespace svr {
namespace dao {

class PgModelDAO : public ModelDAO {
public:
    explicit PgModelDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    bigint get_next_id();

    bool exists(const bigint model_id);

    bool svr_exists(const bigint svr_id);

    int save(const datamodel::Model_ptr &model);

    int remove(const datamodel::Model_ptr &model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    datamodel::Model_ptr get_by_id(const bigint model_id);

    datamodel::Model_ptr get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level);

    std::deque<datamodel::Model_ptr> get_all_ensemble_models(const bigint ensemble_id);

    std::deque<datamodel::OnlineSVR_ptr> get_svr_by_model_id(const bigint model_id);
};

}
}

#endif /* PGMODELDAO_HPP */