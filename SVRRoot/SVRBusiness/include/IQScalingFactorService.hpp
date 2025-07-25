#pragma once

#include "model/IQScalingFactor.hpp"
#include "util/math_utils.hpp"
#include "ScalingFactorService.hpp"

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

class IQScalingFactorService : public ScalingFactorService {
    dao::IQScalingFactorDAO &iq_scaling_factor_dao_;

public:
    IQScalingFactorService(dao::IQScalingFactorDAO &iq_scaling_factor_dao) noexcept;

    bool exists(const datamodel::IQScalingFactor_ptr &iq_scaling_factor) const;

    int save(const datamodel::IQScalingFactor_ptr &iq_scaling_factor) const;

    int remove(const datamodel::IQScalingFactor_ptr &iq_scaling_factor) const;

    std::deque<datamodel::IQScalingFactor_ptr> find_all_by_dataset_id(const bigint dataset_id) const;

    static std::deque<datamodel::IQScalingFactor_ptr> calculate(const datamodel::InputQueue &input_queue, const size_t dataset_id, const size_t use_tail);

    void prepare(datamodel::Dataset &dataset, const bool save) const;

    void prepare(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const bool save) const;

    static datamodel::t_iqscaler get_scaler(const datamodel::IQScalingFactor &sf);

    static datamodel::t_iqscaler get_scaler(const double scaling_factor, const double dc_offset);

    static datamodel::t_iqscaler get_scaler(const datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const std::string &column_name);

    static datamodel::t_iqscaler get_unscaler(const datamodel::IQScalingFactor &sf);

    static datamodel::t_iqscaler get_unscaler(const double scaling_factor, const double dc_offset);

    static datamodel::t_iqscaler get_unscaler(const datamodel::Dataset &dataset, const std::string &table_name, const std::string &column_name);

    static bool check(const std::deque<datamodel::IQScalingFactor_ptr> &iqsf, const std::deque<std::string> &value_columns);

    static const std::function<double(double)> C_default_scaler;

    template<typename T> static inline void unscale_I(const datamodel::IQScalingFactor &sf, T &v)
    {
        common::unscale_I<T>(v, sf.get_scaling_factor(), sf.get_dc_offset());
    }

    template<typename T> static inline T unscale(const datamodel::IQScalingFactor &sf, T v)
    {
        return common::unscale<T>(v, sf.get_scaling_factor(), sf.get_dc_offset());
    }
};

}
}

using IQScalingTaskService_ptr = std::shared_ptr<svr::business::IQScalingFactorService>;
