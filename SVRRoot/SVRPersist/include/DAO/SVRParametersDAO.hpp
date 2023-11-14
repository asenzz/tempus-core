#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel { class SVRParameters; } }
using SVRParameters_ptr = std::shared_ptr<svr::datamodel::SVRParameters>;

namespace svr{
namespace dao{

class SVRParametersDAO: public AbstractDAO
{
public:
    static SVRParametersDAO *build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit SVRParametersDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const SVRParameters_ptr& svr_parameters) = 0;
    virtual int remove(const SVRParameters_ptr& svr_parameters) = 0;
    virtual int remove_by_dataset_id(const bigint dataset_id) = 0;

    virtual std::vector<SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id) = 0;

    virtual size_t get_dataset_levels(const bigint dataset_id) = 0;
};

}
}

using SVRParametersDAO_ptr = std::shared_ptr<svr::dao::SVRParametersDAO>;
