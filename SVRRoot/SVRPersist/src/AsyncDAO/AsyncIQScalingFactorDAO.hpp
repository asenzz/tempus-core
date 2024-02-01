#pragma once

#include <DAO/IQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class AsyncIQScalingFactorDAO : public IQScalingFactorDAO
{
public:
    explicit AsyncIQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource);
    ~AsyncIQScalingFactorDAO();

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const IQScalingFactor_ptr& iQScalingFactor);
    virtual int remove(const IQScalingFactor_ptr& iQscalingFactor);
    virtual std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

private:
    class AsyncImpl;
    AsyncImpl & pImpl;
};

}
}
