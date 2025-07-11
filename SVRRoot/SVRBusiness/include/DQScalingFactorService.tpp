#pragma once

//
// Created by zarko on 08/07/2025.
//

#ifndef DQSCALINGFACTORSERVICE_TPP
#define DQSCALINGFACTORSERVICE_TPP

#include "DQScalingFactorService.hpp"

namespace svr {
namespace business {

template<typename T> void DQScalingFactorService::scale_labels_I(const datamodel::DQScalingFactor &sf, arma::Mat<T> &labels)
{
    (void) scale_I(labels, sf.get_labels_factor(), sf.get_dc_offset_labels());
    if (labels.has_nonfinite()) LOG4_THROW("Scaled labels not sane, scaling factor labels " << sf << ", labels " << labels);
}

template<typename T> void DQScalingFactorService::scale_features_I( // TODO Is gradient and step really needed, since param sets are per model.
    const uint16_t chunk_ix, const uint16_t grad_level, const uint16_t step, const uint16_t lag, const datamodel::dq_scaling_factor_container_t &sf, arma::Mat<T> &features_t)
{
    assert(sf.size());
    const auto num_levels = features_t.n_rows / lag;
    OMP_FOR_i(num_levels) {
        const auto row1 = i * lag;
        const auto row2 = row1 + lag - 1;
        const auto p_sf = find(sf, 0, chunk_ix, grad_level, step, i, true, false);
        if (!p_sf) LOG4_THROW("Features scaling factor not found for chunk " << chunk_ix << ", gradient " << grad_level << ", step " << step << ", level " << i << " in " << sf);
        auto features_t_view = features_t.rows(row1, row2);
        features_t_view = common::scale<arma::Mat<T>>(features_t_view, p_sf->get_features_factor(), p_sf->get_dc_offset_features());
        if (features_t_view.has_nonfinite()) LOG4_THROW("Scaled features not sane, factors " << *p_sf << ", level " << i << ", feats " << features_t_view << ", start " << row1 <<
            ", end " << row2 << ", lag " << lag << ", chunk " << chunk_ix << ", gradient " << grad_level);
    }
}

template<typename T> datamodel::dq_scaling_factor_container_t DQScalingFactorService::calculate( // TODO Make calc and scale inplace version
    const bigint model_id, const datamodel::SVRParameters &param, const arma::Mat<T> &features_t, const arma::Mat<T> &labels)
{
    datamodel::dq_scaling_factor_container_t res;
    const auto chunk_ix = param.get_chunk_index();
    const auto dataset_id = param.get_dataset_id();
    const auto labels_level = param.get_decon_level();
    const auto labels_step = param.get_step();
    LOG4_DEBUG("Calculating scaling factors for dataset id " << dataset_id << ", model " << model_id << ", parameters " << param);

    const auto [dc_offset_labels, labels_factor] = calc(labels, common::C_input_obseg_labels);
    add(res, otr<datamodel::DQScalingFactor>(
            dataset_id, model_id, labels_level, labels_step, param.get_grad_level(), chunk_ix,
            std::numeric_limits<double>::quiet_NaN(), labels_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset_labels));

    // Features
    tbb::mutex add_sf_l;
    const auto lag = param.get_lag_count();
    const auto num_levels = features_t.n_rows / lag;
    OMP_FOR_i(num_levels) {
        const auto level_feats = features_t.rows(i * lag, (i + 1) * lag - 1);
        const auto [dc_offset, scaling_factor] = calc<T>(level_feats, common::C_input_obseg_labels);
        if (!std::isnormal(scaling_factor) || !common::isnormalz(dc_offset))
            LOG4_THROW("Scaling factors not sane, level " << i << ", lag " << lag << ", chunk " << chunk_ix << ", gradient " << param.get_grad_level() <<
                       ", step " << param.get_step() << ", scaling factor " << scaling_factor << ", DC offset " << dc_offset << ", data " << common::present(level_feats.eval()));
        const auto p_sf = otr<datamodel::DQScalingFactor>(
            dataset_id, model_id, i, labels_step, param.get_grad_level(), chunk_ix, scaling_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset);
        const tbb::mutex::scoped_lock lk(add_sf_l);
        add(res, p_sf);
    }

    return res;
}
}
}

#endif //DQSCALINGFACTORSERVICE_TPP
