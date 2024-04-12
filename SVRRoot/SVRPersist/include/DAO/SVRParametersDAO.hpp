#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr { namespace datamodel {
class SVRParameters;
using SVRParameters_ptr = std::shared_ptr<SVRParameters>;
} }

namespace svr{
namespace dao{

class SVRParametersDAO: public AbstractDAO
{
public:
    static SVRParametersDAO *build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit SVRParametersDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const datamodel::SVRParameters_ptr &p_svr_parameters) = 0;
    virtual int remove(const datamodel::SVRParameters_ptr& svr_parameters) = 0;
    virtual int remove_by_dataset_id(const bigint dataset_id) = 0;

    virtual std::deque<datamodel::SVRParameters_ptr> get_all_svrparams_by_dataset_id(const bigint dataset_id) = 0;
    virtual std::deque<datamodel::SVRParameters_ptr> get_svrparams(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level) = 0;

    virtual size_t get_dataset_levels(const bigint dataset_id) = 0;
};

}
}

using SVRParametersDAO_ptr = std::shared_ptr<svr::dao::SVRParametersDAO>;
