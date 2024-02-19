#pragma once

#include "model/IQScalingFactor.hpp"

namespace svr {
namespace dao { class IQScalingFactorDAO; }
namespace datamodel {
class InputQueue;

using InputQueue_ptr = std::shared_ptr<InputQueue>;

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;
}
}

namespace svr {
namespace business {

typedef std::function<double(const double)> t_iqscaler;

class IQScalingFactorService
{
    svr::dao::IQScalingFactorDAO &iq_scaling_factor_dao_;

public:
    IQScalingFactorService(svr::dao::IQScalingFactorDAO &iq_scaling_factor_dao) : iq_scaling_factor_dao_(iq_scaling_factor_dao)
    {}

    bool exists(const IQScalingFactor_ptr &iq_scaling_factor);

    int save(const IQScalingFactor_ptr &iq_scaling_factor);

    int remove(const IQScalingFactor_ptr &iq_scaling_factor);

    std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

    static std::deque<IQScalingFactor_ptr> calculate(const datamodel::InputQueue &input_queue, const size_t dataset_id, const size_t decrement);

    void prepare(datamodel::Dataset &dataset);

    void prepare(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue);

    static t_iqscaler get_scaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name);

    static t_iqscaler get_unscaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name);

    void scale(const datamodel::Dataset_ptr &p_dataset, const bool unscale = false);
};

}
}

using IQScalingTaskService_ptr = std::shared_ptr<svr::business::IQScalingFactorService>;
