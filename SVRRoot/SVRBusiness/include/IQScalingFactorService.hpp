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
    dao::IQScalingFactorDAO &iq_scaling_factor_dao_;

public:
    IQScalingFactorService(dao::IQScalingFactorDAO &iq_scaling_factor_dao) : iq_scaling_factor_dao_(iq_scaling_factor_dao)
    {}

    bool exists(const datamodel::IQScalingFactor_ptr &iq_scaling_factor);

    int save(const datamodel::IQScalingFactor_ptr &iq_scaling_factor);

    int remove(const datamodel::IQScalingFactor_ptr &iq_scaling_factor);

    std::deque<datamodel::IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

    static std::deque<datamodel::IQScalingFactor_ptr> calculate(const datamodel::InputQueue &input_queue, const size_t dataset_id, const size_t tail);

    void prepare(datamodel::Dataset &dataset, const bool save);

    void prepare(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const bool save);

    static t_iqscaler get_scaler(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const std::string &column_name);

    static t_iqscaler get_unscaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name);

    bool check(const std::deque<datamodel::IQScalingFactor_ptr> &iqsf, const std::deque<std::string> &value_columns);

    static const std::function<double(double)> C_default_scaler;
};

}
}

using IQScalingTaskService_ptr = std::shared_ptr<svr::business::IQScalingFactorService>;
