#pragma once

#include "model/IQScalingFactor.hpp"
#include "model/Dataset.hpp"

namespace svr {
    namespace dao { class IQScalingFactorDAO; }
    namespace datamodel {
    class InputQueue;
    using InputQueue_ptr = std::shared_ptr<InputQueue>;
    }
}

namespace svr {
namespace business {

class IQScalingFactorService
{
public:
    IQScalingFactorService(svr::dao::IQScalingFactorDAO& iq_scaling_factor_dao) :
        iq_scaling_factor_dao_(iq_scaling_factor_dao)
    {}

    bool exists(const IQScalingFactor_ptr& iq_scaling_factor);

    int save(const IQScalingFactor_ptr& iq_scaling_factor);
    int remove(const IQScalingFactor_ptr& iq_scaling_factor);

    std::deque<IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

    std::deque<IQScalingFactor_ptr> calculate(const datamodel::InputQueue_ptr &p_input_queue, const size_t dataset_id, const double alpha = 0.02);
    void scale(const datamodel::Dataset_ptr& p_dataset, const bool unscale = false);
private:
    svr::dao::IQScalingFactorDAO & iq_scaling_factor_dao_;
};

}
}

using IQScalingTaskService_ptr = std::shared_ptr<svr::business::IQScalingFactorService>;
