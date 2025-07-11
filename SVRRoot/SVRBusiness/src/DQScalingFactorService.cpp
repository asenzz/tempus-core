#include <ipp.h>
#include <iterator>
#include <limits>
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "util/math_utils.hpp"
#include "model/DQScalingFactor.hpp"
#include "appcontext.hpp"
#include "DAO/DQScalingFactorDAO.hpp"
#include "spectral_transform.hpp"
#include "DQScalingFactorService.hpp"

#ifdef CUDA_SCALING_FACTORS
#include "dq_scaling_factors_service_impl.cuh"
#endif


namespace svr {
namespace business {
void DQScalingFactorService::add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor_ptr &p_new_sf, const bool overwrite)
{
    if (const auto [match, set] = match_n_set(sf, *p_new_sf, overwrite); match) return;
    LOG4_DEBUG("Adding " << *p_new_sf << " container of size " << sf.size());
    sf.emplace(p_new_sf);
}

void DQScalingFactorService::add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, const bool overwrite)
{
    std::ranges::copy_if(new_sf, std::inserter(sf, sf.end()), [&sf, overwrite](const auto &nf) {
        const auto [match, set] = match_n_set(sf, *nf, overwrite);
        return !match;
    });
}

bool DQScalingFactorService::exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) const
{
    return dq_scaling_factor_dao.exists(dq_scaling_factor);
}


int DQScalingFactorService::save(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor) const
{
    if (!p_dq_scaling_factor->get_id()) p_dq_scaling_factor->set_id(dq_scaling_factor_dao.get_next_id());
    return dq_scaling_factor_dao.save(p_dq_scaling_factor);
}


int DQScalingFactorService::remove(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) const
{
    return dq_scaling_factor_dao.remove(dq_scaling_factor);
}


datamodel::dq_scaling_factor_container_t DQScalingFactorService::find_all_by_model_id(const bigint model_id) const
{
    return dq_scaling_factor_dao.find_all_by_model_id(model_id);
}


datamodel::DQScalingFactor_ptr DQScalingFactorService::find(
    const datamodel::dq_scaling_factor_container_t &scaling_factors,
    const bigint model_id,
    const uint16_t chunk,
    const uint16_t gradient,
    const uint16_t step,
    const uint16_t level,
    const bool check_features,
    const bool check_labels)
{
    const auto res = std::find_if(C_default_exec_policy, scaling_factors.cbegin(), scaling_factors.cend(),
                                  [check_features, check_labels, model_id, chunk, gradient, step, level](const auto &sf) {
                                      return (!model_id || !sf->get_model_id() || sf->get_model_id() == model_id)
                                             && level == sf->get_decon_level()
                                             && step == sf->get_step()
                                             && gradient == sf->get_grad_depth()
                                             && chunk == sf->get_chunk_index()
                                             && (!check_labels || std::isnormal(sf->get_labels_factor()))
                                             && (!check_features || std::isnormal(sf->get_features_factor()))
                                             && (!check_labels || common::isnormalz(sf->get_dc_offset_labels()))
                                             && (!check_features || common::isnormalz(sf->get_dc_offset_features()));
                                  });
    if (res == scaling_factors.cend()) {
        LOG4_ERROR("Scaling factors for model " << model_id << " not found, chunk " << chunk << ", gradient " << gradient << ", step " << step << ", level " << level <<
            ", features " << check_features);
        return nullptr;
    }
    return *res;
}

void DQScalingFactorService::scale_features_I(const uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &features_t)
{
    if (features_t.empty()) return;
    const auto chunk_sf = slice(svr_model.get_scaling_factors(), chunk_ix, svr_model.get_gradient_level(), svr_model.get_step());
    scale_features_I(chunk_ix, svr_model.get_gradient_level(), svr_model.get_step(), svr_model.get_params_ptr(chunk_ix)->get_lag_count(), chunk_sf, features_t);
}

void DQScalingFactorService::scale_labels_I(
    const uint16_t chunk, const uint16_t gradient, const uint16_t step, const uint16_t level, const datamodel::dq_scaling_factor_container_t &sf, arma::mat &labels)
{
    const auto p_sf_labels = find(sf, 0, chunk, gradient, step, level, false, true);
    scale_labels_I(*p_sf_labels, labels);
}

void DQScalingFactorService::scale_labels_I(const uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &labels)
{
    if (labels.empty()) return;
    const auto p_sf_labels = find(svr_model.get_scaling_factors(), svr_model.get_model_id(), chunk_ix, svr_model.get_gradient_level(), svr_model.get_step(),
                                  svr_model.get_decon_level(), false, true);
    scale_labels_I(*p_sf_labels, labels);
}

arma::mat DQScalingFactorService::scale_labels(const datamodel::DQScalingFactor &sf, const arma::mat &labels)
{
    return scale(labels, sf.get_labels_factor(), sf.get_dc_offset_labels());
}

double DQScalingFactorService::scale_label(const datamodel::DQScalingFactor &sf, double &label)
{
    label = scale(label, sf.get_labels_factor(), sf.get_dc_offset_labels());
    if (!common::isnormalz(label))
        LOG4_THROW("Scaled labels not sane, scaling factor labels " << sf << ", label " << label);
    return label;
}


std::pair<bool, bool> DQScalingFactorService::match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, const bool overwrite)
{
    std::atomic<bool> match = false, set = false;
    OMP_FOR_i(sf.size()) {
        auto of = sf ^ i;
        if (nf ^= *of) match = true;
        else continue;
        bool factor_set = false;
        if (std::isnormal(nf.get_labels_factor()) && (overwrite || !std::isnormal(of->get_labels_factor()))) {
            of->set_labels_factor(nf.get_labels_factor());
            factor_set = true;
        }
        if (std::isnormal(nf.get_features_factor()) && (overwrite || !std::isnormal(of->get_features_factor()))) {
            of->set_features_factor(nf.get_features_factor());
            factor_set = true;
        }
        if (common::isnormalz(nf.get_dc_offset_labels()) && (overwrite || !common::isnormalz(of->get_dc_offset_labels()))) {
            of->set_dc_offset_labels(nf.get_dc_offset_labels());
            factor_set = true;
        }
        if (common::isnormalz(nf.get_dc_offset_features()) && (overwrite || !common::isnormalz(of->get_dc_offset_features()))) {
            of->set_dc_offset_features(nf.get_dc_offset_features());
            factor_set = true;
        }
        if (factor_set) {
            LOG4_TRACE("Set factor " << *of);
            set = true;
        }
    }
    return std::make_pair(match.load(), set.load());
}

void DQScalingFactorService::reset(datamodel::dq_scaling_factor_container_t &sf, const uint16_t decon, const uint16_t chunk, const uint16_t step, const uint16_t grad)
{
    OMP_FOR_i(sf.size()) {
        auto of = sf ^ i;
        if (of->get_decon_level() != decon || of->get_chunk_index() != chunk || of->get_step() != step || of->get_grad_depth() != grad) continue;
        of->set_labels_factor(std::numeric_limits<double>::quiet_NaN());
        of->set_features_factor(std::numeric_limits<double>::quiet_NaN());
        of->set_dc_offset_labels(std::numeric_limits<double>::quiet_NaN());
        of->set_dc_offset_features(std::numeric_limits<double>::quiet_NaN());
        LOG4_TRACE("Reset scaling factor " << *of);
    }
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(const datamodel::dq_scaling_factor_container_t &sf, const uint16_t chunk, const uint16_t gradient, const uint16_t step)
{
    datamodel::dq_scaling_factor_container_t res;
    std::copy_if(C_default_exec_policy, sf.cbegin(), sf.cend(), std::inserter(res, res.end()), [chunk, gradient, step](const auto &p_sf) {
        return p_sf->get_grad_depth() == gradient && p_sf->get_chunk_index() == chunk && p_sf->get_step() == step;
    });
    return res;
}
}
}
