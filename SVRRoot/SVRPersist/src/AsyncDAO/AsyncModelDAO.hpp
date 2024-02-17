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

    bool exists(const bigint model_id);

    int save(const datamodel::Model_ptr& model);

    int remove(const datamodel::Model_ptr& model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    datamodel::Model_ptr get_by_id(const bigint model_id);

    datamodel::Model_ptr get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level);

    std::deque<datamodel::Model_ptr> get_all_ensemble_models(const bigint ensemble_id);

    std::deque<OnlineMIMOSVR_ptr> get_svr_by_model_id(const bigint model_id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}

#endif /* ASYNCMODELDAO_HPP */

