#ifndef ASYNCSVRPARAMETERSDAO_HPP
#define ASYNCSVRPARAMETERSDAO_HPP

#include <DAO/SVRParametersDAO.hpp>

namespace svr{
namespace dao{

class AsyncSVRParametersDAO: public SVRParametersDAO
{
public:
    explicit AsyncSVRParametersDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncSVRParametersDAO();

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const datamodel::SVRParameters_ptr &p_svr_parameters);
    virtual int remove(const datamodel::SVRParameters_ptr& svr_parameters);
    virtual int remove_by_dataset_id(const bigint dataset_id);

    virtual std::deque<datamodel::SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id);
    virtual std::deque<datamodel::SVRParameters_ptr> get_svrparams(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level, const size_t step);
    virtual size_t get_dataset_levels(const bigint dataset_id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

}
}

#endif /* ASYNCSVRPARAMETERSDAO_HPP */
