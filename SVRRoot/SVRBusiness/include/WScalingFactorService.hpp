//
// Created by zarko on 16/11/2024.
//

#ifndef SVR_WSCALINGFACTORSERVICE_HPP
#define SVR_WSCALINGFACTORSERVICE_HPP

#include "model/WScalingFactor.hpp"
#include "util/math_utils.hpp"
#include "ScalingFactorService.hpp"

namespace svr {
namespace dao { class WScalingFactorDAO; }
namespace datamodel {
class InputQueue;

using InputQueue_ptr = std::shared_ptr<InputQueue>;

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;
}
}

namespace svr {
namespace business {

class WScalingFactorService : public ScalingFactorService {
    dao::WScalingFactorDAO &w_scaling_factor_dao_;

public:
    WScalingFactorService(dao::WScalingFactorDAO &w_scaling_factor_dao) noexcept;

    bool exists(const datamodel::WScalingFactor_ptr &w_scaling_factor);

    int save(const datamodel::WScalingFactor_ptr &w_scaling_factor);

    int remove(const datamodel::WScalingFactor_ptr &w_scaling_factor);

    std::deque<datamodel::WScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id);

    static datamodel::WScalingFactor_ptr find(const std::deque<datamodel::WScalingFactor_ptr> &w_scaling_factors, const uint16_t step);

    void scale(const bigint dataset_id, arma::mat &weights);
};

}
}

#endif //SVR_WSCALINGFACTORSERVICE_HPP
