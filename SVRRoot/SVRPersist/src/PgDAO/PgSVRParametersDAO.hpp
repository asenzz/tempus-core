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
    virtual int save(const SVRParameters_ptr& svr_parameters);
    virtual int remove(const SVRParameters_ptr& svr_parameters);
    virtual int remove_by_dataset_id(const bigint ensemble_id);

    virtual std::vector<SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id);
    virtual size_t get_dataset_levels(const bigint dataset_id);
};

}
}

#endif /* PGSVRPARAMETERSDAO_HPP */

