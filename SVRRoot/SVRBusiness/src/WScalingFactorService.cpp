#include <string>
#include "common/constants.hpp"
#include "common/parallelism.hpp"
#include "util/time_utils.hpp"
#include "DAO/WScalingFactorDAO.hpp"
#include "model/Dataset.hpp"
#include "model/InputQueue.hpp"
#include "model/DataRow.hpp"
#include "WScalingFactorService.hpp"
#include "appcontext.hpp"


namespace svr {
namespace business {

WScalingFactorService::WScalingFactorService(dao::WScalingFactorDAO &w_scaling_factor_dao) noexcept : w_scaling_factor_dao_(w_scaling_factor_dao)
{}

bool WScalingFactorService::exists(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    return w_scaling_factor_dao_.exists(p_w_scaling_factor->get_id());
}

int WScalingFactorService::save(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    return w_scaling_factor_dao_.save(p_w_scaling_factor);
}

int WScalingFactorService::remove(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    return w_scaling_factor_dao_.remove(p_w_scaling_factor);
}

std::deque<datamodel::WScalingFactor_ptr> WScalingFactorService::find_all_by_dataset_id(const bigint dataset_id)
{
    return w_scaling_factor_dao_.find_all_by_dataset_id(dataset_id);
}

datamodel::WScalingFactor_ptr WScalingFactorService::find(const std::deque<datamodel::WScalingFactor_ptr> &w_scaling_factors, const uint16_t step)
{
    const auto res = std::find_if(C_default_exec_policy, w_scaling_factors.cbegin(), w_scaling_factors.cend(),
                                  [step](const auto &p_w_scaling_factor) { return p_w_scaling_factor->get_step() == step; });
    if (res != w_scaling_factors.cend()) return *res;
    LOG4_WARN("WScalingFactor for step " << step << " not found among " << w_scaling_factors.size() << " WScalingFactors.");
    return nullptr;
}


void WScalingFactorService::scale(const bigint dataset_id, arma::mat &weights)
{
    if (weights.empty()) {
        LOG4_ERROR("Weights for dataset ID " << dataset_id);
        return;
    }

    const auto dataset_sf = find_all_by_dataset_id(dataset_id);
    OMP_FOR_i(weights.n_cols) {
        auto sf = find(dataset_sf, i);
        if (!sf) {
            const auto [s, dc] = calc(weights.col(i));
            sf = std::make_shared<datamodel::WScalingFactor>(0, dataset_id, i, s, dc);
            save(sf);
        }
        weights.col(i) = ScalingFactorService::scale<const arma::mat>(weights.col(i), sf->get_scaling_factor(), sf->get_dc_offset());
        constexpr double C_weight_variance = 1;
        weights.col(i) = (C_weight_variance + weights.col(i)) / (C_weight_variance + 1.);
    }
}

}
}
