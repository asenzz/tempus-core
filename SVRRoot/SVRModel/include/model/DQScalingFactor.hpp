#pragma once

#include <execution>
#include <oneapi/tbb/concurrent_set.h>
#include "model/Entity.hpp"
#include "util/string_utils.hpp"
#include "SVRParameters.hpp"


namespace svr {
namespace datamodel {
class DQScalingFactor;

using DQScalingFactor_ptr = std::shared_ptr<DQScalingFactor>;
}
}

namespace svr {
namespace datamodel {

struct DQScalingFactorsLess;
typedef std::set<DQScalingFactor_ptr, DQScalingFactorsLess> dq_scaling_factor_container_t;

class DQScalingFactor : public Entity
{
private:
    bigint model_id_ = 0;  // TODO Replace with pointer to Dataset

    unsigned decon_level_ = C_default_svrparam_decon_level;
    unsigned step_ = C_default_svrparam_step;
    unsigned grad_depth_ = C_default_svrparam_grad_level;
    unsigned chunk_ix_ = C_default_svrparam_chunk_ix;

    double scaling_factor_features = std::numeric_limits<double>::quiet_NaN();
    double scaling_factor_labels = std::numeric_limits<double>::quiet_NaN();
    double dc_offset_features = std::numeric_limits<double>::quiet_NaN();
    double dc_offset_labels = std::numeric_limits<double>::quiet_NaN();

public:
    DQScalingFactor(
            const bigint id, const bigint model_id,
            const unsigned decon_level, const unsigned step, const unsigned grad_depth, const unsigned chunk_index,
            const double scale_feat = std::numeric_limits<double>::quiet_NaN(),
            const double scale_labels = std::numeric_limits<double>::quiet_NaN(),
            const double dc_offset_feat = std::numeric_limits<double>::quiet_NaN(),
            const double dc_offset_labels = std::numeric_limits<double>::quiet_NaN());

    bool operator^=(const DQScalingFactor &o) const;

    bool operator==(const DQScalingFactor &o) const;

    bool operator<(const DQScalingFactor &o) const;

    virtual void init_id() override;

    bigint get_model_id() const;

    void set_model_id(const bigint model_id);

    unsigned get_decon_level() const;

    void set_decon_level(const unsigned decon_level);

    unsigned get_step() const;

    void set_step(const unsigned step);

    unsigned get_grad_depth() const;

    void set_grad_depth(const unsigned grad_level);

    unsigned get_chunk_index() const;

    void set_chunk_index(const unsigned chunk_index);

    double get_features_factor() const;

    void set_features_factor(const double scaling_factor);

    double get_labels_factor() const;

    void set_labels_factor(const double label_factor);

    double get_dc_offset_features() const;

    void set_dc_offset_features(const double dc_offset_features_);

    double get_dc_offset_labels() const;

    void set_dc_offset_labels(const double dc_offset_labels_);

    std::string to_string() const override;

    bool in(const dq_scaling_factor_container_t &c);
};

template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const DQScalingFactor &d)
{
    return s << d.to_string();
}


struct DQScalingFactorsLess
{
    bool operator()(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs) const
    { return *lhs < *rhs; }
};

bool operator<(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs);

} // namespace datamodel
} // namespace svr
