#pragma once

#include "model/Dataset.hpp"

namespace svr{
namespace datamodel{

class ScalingFactorsTask : public Entity{

private:
    bigint      dataset_id;
    bool        force_recalculate_scaling_factors = false;
    int         status                  = 0; // 0 - new, 1 - in process, 2 - done, 3 - error
    double      mse                     = -1.0;

public:

    ScalingFactorsTask() : Entity() {}

    ScalingFactorsTask(const bigint id, const bigint dataset_id,
                   bool force_recalculate_scaling_factors = false,
                   int status = 0, double mse = -1.0)
            : Entity(id),
              dataset_id(dataset_id),
              force_recalculate_scaling_factors(force_recalculate_scaling_factors),
              status(status),
              mse(mse) {
    }

    bool operator == (const ScalingFactorsTask& other) const {
        return other.get_id()                       == get_id()
                && other.get_dataset_id()           == get_dataset_id()
                && other.get_force_recalculate_scaling_factors() == get_force_recalculate_scaling_factors()
                && other.get_status()               == get_status()
                && other.get_mse()                  == get_mse();
    }

    bigint get_dataset_id() const
    {
        return dataset_id;
    }

    void set_dataset_id(const bigint dataset_id)
    {
        this->dataset_id = dataset_id;
    }

    void set_force_recalculate_scaling_factors(const bool force_recalculate)
    {
        this->force_recalculate_scaling_factors = force_recalculate;
    }

    bool get_force_recalculate_scaling_factors() const
    {
        return force_recalculate_scaling_factors;
    }

    int get_status() const {
        return status;
    }

    void set_status(int status) {
        this->status = status;
    }

    double get_mse() const {
        return mse;
    }

    void set_mse(double value) {
        this->mse = value;
    }

    virtual std::string to_string() const override {
        std::stringstream ss;

        ss  << "Scaling Factors task id: "  << get_id()
            << ", Dataset id: "             << get_dataset_id()
            << ", Force recalculate scaling factors" << get_force_recalculate_scaling_factors()
            << ", Status: "                 << get_status()
            << ", MSE: "                    << get_mse();

        return ss.str();
    }
};

} /* namespace datamodel */
} /* namespace svr */

using ScalingFactorsTask_ptr = std::shared_ptr<svr::datamodel::ScalingFactorsTask>;
