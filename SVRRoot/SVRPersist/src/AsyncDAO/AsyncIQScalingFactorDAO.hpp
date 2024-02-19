#pragma once

#include <DAO/IQScalingFactorDAO.hpp>

namespace svr {
namespace dao {

class AsyncIQScalingFactorDAO : public IQScalingFactorDAO
{
public:
    explicit AsyncIQScalingFactorDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source);

    ~AsyncIQScalingFactorDAO();

    virtual bigint get_next_id();

    virtual bool exists(const bigint id);

    virtual int save(const IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual int remove(const IQScalingFactor_ptr &p_iq_scaling_factor);

    virtual std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

private:
    struct AsyncImpl;
    AsyncImpl &pImpl;
};

}
}
