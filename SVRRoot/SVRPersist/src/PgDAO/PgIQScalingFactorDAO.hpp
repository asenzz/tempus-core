#pragma once

#include <DAO/IQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class PgIQScalingFactorDAO : public IQScalingFactorDAO
{
public:
    explicit PgIQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties,
                                  svr::dao::DataSource& dataSource);

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const IQScalingFactor_ptr& iQscalingFactor);
    virtual int remove(const IQScalingFactor_ptr& iQscalingFactor);
    virtual std::vector<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);
};

}
}
