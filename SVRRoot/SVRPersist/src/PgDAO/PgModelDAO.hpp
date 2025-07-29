#ifndef PGMODELDAO_HPP
#define PGMODELDAO_HPP

#include "DAO/ModelDAO.hpp"
#include "onlinesvr.hpp"

namespace svr {
namespace dao {

class PgModelDAO : public ModelDAO {
public:
    explicit PgModelDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    bigint get_next_id() override;

    bool exists(bigint model_id) override;

    bool svr_exists(bigint svr_id);

    int save(const datamodel::Model_ptr &model) override;

    int remove(const datamodel::Model_ptr &model) override;

    int remove_by_ensemble_id(bigint ensemble_id) override;

    datamodel::Model_ptr get_by_id(bigint model_id) override;

    datamodel::Model_ptr get_by_ensemble_id_and_decon_level(bigint ensemble_id, size_t decon_level) override;

    std::deque<datamodel::Model_ptr> get_all_ensemble_models(bigint ensemble_id) override;

    std::deque<datamodel::OnlineSVR_ptr> get_svr_by_model_id(bigint model_id) override;
};

}
}

#endif /* PGMODELDAO_HPP */