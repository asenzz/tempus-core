#ifndef PGMODELDAO_HPP
#define PGMODELDAO_HPP

#include <DAO/ModelDAO.hpp>

namespace svr { namespace dao {

class PgModelDAO: public ModelDAO
{
public:
    explicit PgModelDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    bigint get_next_id();

    bool exists(const bigint model_id);

    int save(const datamodel::Model_ptr& model);

    int remove(const datamodel::Model_ptr& model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    datamodel::Model_ptr get_by_id(const bigint model_id);

    datamodel::Model_ptr get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level);

    std::deque<datamodel::Model_ptr> get_all_ensemble_models(const bigint ensemble_id);
};

}
}

#endif /* PGMODELDAO_HPP */

