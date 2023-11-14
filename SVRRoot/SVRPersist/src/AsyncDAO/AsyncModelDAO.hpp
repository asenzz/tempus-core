#ifndef ASYNCMODELDAO_HPP
#define ASYNCMODELDAO_HPP

#include <DAO/ModelDAO.hpp>

namespace svr { namespace dao {

class AsyncModelDAO: public ModelDAO
{
public:
    explicit AsyncModelDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncModelDAO();

    bigint get_next_id();

    bool exists(bigint model_id);

    int save(const Model_ptr& model);

    int remove(const Model_ptr& model);

    int remove_by_ensemble_id(bigint ensemble_id);

    Model_ptr get_by_id(bigint model_id);

    Model_ptr get_by_ensemble_id_and_decon_level(bigint ensemble_id, size_t decon_level);

    std::vector<Model_ptr> get_all_ensemble_models(bigint ensemble_id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}

#endif /* ASYNCMODELDAO_HPP */

