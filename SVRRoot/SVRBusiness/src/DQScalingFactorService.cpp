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
    const auto [match, set] = match_n_set(sf, *p_new_sf, overwrite);
    if (match) return;
    LOG4_DEBUG("Adding " << *p_new_sf << " container of size " << sf.size());
    sf.emplace(p_new_sf);
}

void DQScalingFactorService::add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, const bool overwrite)
{
     std::copy_if(new_sf.cbegin(), new_sf.cend(), std::inserter(sf, sf.end()), [&sf, overwrite](const auto &nf) {
         const auto [match, set] = match_n_set(sf, *nf, overwrite);
         return !match;
     });
}

bool DQScalingFactorService::exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.exists(dq_scaling_factor);
}


int DQScalingFactorService::save(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor)
{
    if (!p_dq_scaling_factor->get_id()) p_dq_scaling_factor->set_id(dq_scaling_factor_dao.get_next_id());
    return dq_scaling_factor_dao.save(p_dq_scaling_factor);
}


int DQScalingFactorService::remove(const datamodel::DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.remove(dq_scaling_factor);
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::find_all_by_model_id(const bigint model_id)
{
    return dq_scaling_factor_dao.find_all_by_model_id(model_id);
}


datamodel::DQScalingFactor_ptr
DQScalingFactorService::find(
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


datamodel::dq_scaling_factor_container_t // TODO Make calc and scale inplace version
DQScalingFactorService::calculate(const uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, const arma::mat &features_t, const arma::mat &labels)
{
    datamodel::dq_scaling_factor_container_t res;
    auto p_params = svr_model.get_params_ptr(chunk_ix);
    const auto dataset_id = svr_model.get_dataset() ? svr_model.get_dataset()->get_id() : 0;
    const auto labels_level = p_params->get_decon_level();
    const auto labels_step = p_params->get_step();
    LOG4_DEBUG("Calculating scaling factors for dataset id " << dataset_id << ", model " << svr_model.get_model_id() << ", parameters " << *p_params);

    const auto [dc_offset_labels, labels_factor] = calc(labels, common::C_input_obseg_labels);
    add(res, otr<datamodel::DQScalingFactor>(
            dataset_id, svr_model.get_model_id(), labels_level, labels_step, svr_model.get_gradient_level(), chunk_ix,
            std::numeric_limits<double>::quiet_NaN(), labels_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset_labels));

    // Features
    tbb::mutex add_sf_l;
    const auto lag = p_params->get_lag_count();
    const auto num_levels = features_t.n_rows / lag;
    OMP_FOR_i(num_levels) {
        const auto level_feats = features_t.rows(i * lag, (i + 1) * lag - 1);
        const auto [dc_offset, scaling_factor] = calc(level_feats, common::C_input_obseg_labels);
        if (!std::isnormal(scaling_factor) || !common::isnormalz(dc_offset))
            LOG4_THROW("Scaling factors not sane, level " << i << ", lag " << lag << ", chunk " << chunk_ix << ", gradient " << svr_model.get_gradient_level() <<
                  ", step " << svr_model.get_step() << ", scaling factor " << scaling_factor << ", DC offset " << dc_offset << ", data " << common::present(arma::mat(level_feats)));
        const auto p_sf = otr<datamodel::DQScalingFactor>(
                dataset_id, svr_model.get_model_id(), i, labels_step, svr_model.get_gradient_level(), chunk_ix, scaling_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset);
        const tbb::mutex::scoped_lock lk(add_sf_l);
        add(res, p_sf);
    }

    return res;
}

void DQScalingFactorService::scale_features(const uint16_t chunk_ix, const uint16_t grad_level, const uint16_t step, const uint16_t lag,
                                            const datamodel::dq_scaling_factor_container_t &sf, arma::mat &features_t)
{
    assert(sf.size());
    const auto num_levels = features_t.n_rows / lag;
    OMP_FOR_i(num_levels) {
        const auto row1 = i * lag;
        const auto row2 = row1 + lag - 1;
        const auto p_sf = find(sf, 0, chunk_ix, grad_level, step, i, true, false);
        if (!p_sf) LOG4_THROW("Features scaling factor not found for chunk " << chunk_ix << ", gradient " << grad_level << ", step " << step << ", level " << i << " in " << sf);
        auto features_t_view = features_t.rows(row1, row2);
        features_t_view = common::scale<arma::mat>(features_t_view, p_sf->get_features_factor(), p_sf->get_dc_offset_features());
        if (features_t_view.has_nonfinite()) LOG4_THROW("Scaled features not sane, factors " << *p_sf << ", level " << i << ", feats " << features_t_view << ", start " << row1 <<
            ", end " << row2 << ", lag " << lag << ", chunk " << chunk_ix << ", gradient " << grad_level);
    }
}

void DQScalingFactorService::scale_features(const uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &features_t)
{
    if (features_t.empty()) return;
    const auto chunk_sf = slice(svr_model.get_scaling_factors(), chunk_ix, svr_model.get_gradient_level(), svr_model.get_step());
    scale_features(chunk_ix, svr_model.get_gradient_level(), svr_model.get_step(), svr_model.get_params_ptr(chunk_ix)->get_lag_count(), chunk_sf, features_t);
}

void
DQScalingFactorService::scale_labels(const uint16_t chunk, const uint16_t gradient, const uint16_t step, const uint16_t level, const datamodel::dq_scaling_factor_container_t &sf,
                                     arma::mat &labels)
{
    const auto p_sf_labels = find(sf, 0, chunk, gradient, step, level, false, true);
    scale_labels(*p_sf_labels, labels);
}

void DQScalingFactorService::scale_labels(const uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &labels)
{
    if (labels.empty()) return;
    const auto p_sf_labels = find(svr_model.get_scaling_factors(), svr_model.get_model_id(), chunk_ix, svr_model.get_gradient_level(), svr_model.get_step(),
                                  svr_model.get_decon_level(), false, true);
    scale_labels(*p_sf_labels, labels);
}

void DQScalingFactorService::scale_labels(const datamodel::DQScalingFactor &sf, arma::mat &labels)
{
    (void) scale_I(labels, sf.get_labels_factor(), sf.get_dc_offset_labels());
    if (labels.has_nonfinite()) LOG4_THROW("Scaled labels not sane, scaling factor labels " << sf << ", labels " << labels);
}

double DQScalingFactorService::scale_label(const datamodel::DQScalingFactor &sf, double &label)
{
    label = scale(label, sf.get_labels_factor(), sf.get_dc_offset_labels());
    if (!common::isnormalz(label)) LOG4_THROW("Scaled labels not sane, scaling factor labels " << sf << ", label " << label);
    return label;
}


std::pair<bool, bool> DQScalingFactorService::match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, const bool overwrite)
{
    bool match = false;
    const auto set = std::any_of(C_default_exec_policy, sf.begin(), sf.end(), [&nf, &match, overwrite](auto &of) {
            if (nf ^= *of) match = true;
            else return false;

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
            if (factor_set) LOG4_TRACE("Set factor " << *of);
            return factor_set;
        });
    return std::make_pair(match, set);
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
