#ifndef PGSVRPARAMETERSDAO_HPP
#define PGSVRPARAMETERSDAO_HPP

#include <DAO/SVRParametersDAO.hpp>

namespace svr{
namespace dao{

class PgSVRParametersDAO: public SVRParametersDAO
{
public:
    explicit PgSVRParametersDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const datamodel::SVRParameters_ptr& svr_parameters);
    virtual int remove(const datamodel::SVRParameters_ptr& svr_parameters);
    virtual int remove_by_dataset_id(const bigint ensemble_id);

    virtual std::deque<datamodel::SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id);
    virtual std::deque<datamodel::SVRParameters_ptr> get_svrparams(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level);
    virtual size_t get_dataset_levels(const bigint dataset_id);
};

}
}

#endif /* PGSVRPARAMETERSDAO_HPP */

