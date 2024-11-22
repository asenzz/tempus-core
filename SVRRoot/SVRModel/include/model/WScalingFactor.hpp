#pragma once

#include "Entity.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace datamodel {

class WScalingFactor : public Entity {

    PROPERTY(bigint, dataset_id, 0)

    PROPERTY(uint16_t, step, 0)

    PROPERTY(double, scaling_factor, C_double_nan)

    PROPERTY(double, dc_offset, C_double_nan)

public:
    WScalingFactor(const bigint id, const bigint dataset_id, const uint16_t step,
                   const double scaling_factor = std::numeric_limits<double>::quiet_NaN(),
                   const double dc_offset = std::numeric_limits<double>::quiet_NaN());

    virtual void init_id() override;

    bool operator==(const WScalingFactor &o) const;

    virtual std::string to_string() const override;
};


template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const WScalingFactor &i)
{
    return s << i.to_string();
}

using WScalingFactor_ptr = std::shared_ptr<svr::datamodel::WScalingFactor>;

} // namespace datamodel
} // namespace svr
