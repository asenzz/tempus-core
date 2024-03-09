#include "common/constants.hpp"
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
    if (match_n_set(sf, *p_new_sf, overwrite)) return;
    LOG4_DEBUG("Adding " << *p_new_sf << " container of size " << sf.size());
    sf.emplace(p_new_sf);
}

void DQScalingFactorService::add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, const bool overwrite)
{
    std::for_each(std::execution::par_unseq, new_sf.cbegin(), new_sf.cend(), [&](const auto &nf) { if (!match_n_set(sf, *nf, overwrite)) sf.emplace(nf); });
}

bool DQScalingFactorService::exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.exists(dq_scaling_factor);
}


int DQScalingFactorService::save(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor)
{
    if (!p_dq_scaling_factor->get_id())
        p_dq_scaling_factor->set_id(dq_scaling_factor_dao.get_next_id());
    return dq_scaling_factor_dao.save(p_dq_scaling_factor);
}


int DQScalingFactorService::remove(const datamodel::DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.remove(dq_scaling_factor);
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::find_all_by_dataset_id(const bigint dataset_id)
{
    return dq_scaling_factor_dao.find_all_by_dataset_id(dataset_id);
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue)
{
    return slice(p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), p_decon_queue->get_input_queue_column_name());
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const size_t dataset_id,
        const std::string &input_queue_column_name)
{
    LOG4_DEBUG("Slicing decon queue scaling factors with dataset id " << dataset_id << " input queue column name " << input_queue_column_name);

    datamodel::dq_scaling_factor_container_t result;
    std::for_each(std::execution::par_unseq, scaling_factors.cbegin(), scaling_factors.cend(),
                 [&input_queue_column_name, &dataset_id, &result] (const auto &sf) {
        if ((!dataset_id || sf->get_dataset_id() == dataset_id) && sf->get_input_queue_column_name() == input_queue_column_name)
            result.emplace(sf);
    });

    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
    return result;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const size_t dataset_id,
        const std::string &input_queue_column_name,
        const std::set<size_t> &levels,
        const bool match_missing,
        const bool check_features_only)
{
//    LOG4_TRACE("Slicing scaling factors " << scaling_factors.size() << " with dataset id " << dataset_id << ", input queue column name " << input_queue_column_name << ", levels " << levels);

    datamodel::dq_scaling_factor_container_t result;
    std::for_each(std::execution::par_unseq, scaling_factors.cbegin(), scaling_factors.cend(),
          [&check_features_only, &match_missing, &result, &dataset_id, &levels, &input_queue_column_name](const auto &sf) {
              if ((!dataset_id || !sf->get_dataset_id() || sf->get_dataset_id() == dataset_id)
                  && sf->get_input_queue_column_name() == input_queue_column_name
                  && (levels.empty() || levels.count(sf->get_decon_level())))
                  if (match_missing || (
                          (check_features_only || std::isnormal(sf->get_labels_factor()))
                          && std::isnormal(sf->get_features_factor())
                          && (check_features_only || common::isnormalz(sf->get_dc_offset_labels()))
                          && common::isnormalz(sf->get_dc_offset_features())))
        result.emplace(sf);
    });

//    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
    return result;
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const size_t dataset_id,
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const std::set<size_t> &feat_levels)
{
    datamodel::dq_scaling_factor_container_t r;
    OMP_LOCK(slice_insert_l)
#pragma omp parallel for num_threads(adj_threads(decon_queues.size()))
    for (const auto &d: decon_queues) {
        const auto dr = slice(scaling_factors, dataset_id, d->get_input_queue_column_name(), feat_levels);
        omp_set_lock(&slice_insert_l);
        r.insert(dr.cbegin(), dr.cend());
        omp_unset_lock(&slice_insert_l);
    }
    return r;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::check(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const datamodel::SVRParameters &head_params,
        const std::set<size_t> feat_levels,
        const datamodel::dq_scaling_factor_container_t &scaling_factors)
{
    const auto dataset_id = head_params.get_dataset_id();
    datamodel::dq_scaling_factor_container_t missing;
    if (slice(scaling_factors, dataset_id, head_params.get_input_queue_column_name(), std::set{head_params.get_decon_level()}).empty())
        missing.emplace(ptr<datamodel::DQScalingFactor>(
                0, dataset_id, decon_queues.front()->get_input_queue_table_name(), head_params.get_input_queue_column_name(), head_params.get_decon_level()));

#pragma omp parallel for num_threads(adj_threads(decon_queues.size())) schedule(static, 1)
    for (const auto &dq: decon_queues) {
        const auto found_sf = slice(scaling_factors, dataset_id, dq->get_input_queue_column_name(), feat_levels, false, true);
        if (found_sf.size() == feat_levels.size()) continue;
        if (found_sf.empty())
            for (const auto fl: feat_levels)
                missing.emplace(ptr<datamodel::DQScalingFactor>(
                        0, dataset_id, dq->get_input_queue_table_name(), dq->get_input_queue_column_name(), fl));
        else
            if (found_sf.size() != feat_levels.size())
                for (const auto fl: feat_levels)
                    if (slice(found_sf, dataset_id, dq->get_input_queue_column_name(), std::set{fl}, false, true).empty())
                        missing.emplace(ptr<datamodel::DQScalingFactor>(
                            0, dataset_id, dq->get_input_queue_table_name(), dq->get_input_queue_column_name(), fl));
    }
    return missing;
}


double DQScalingFactorService::calc_dc_offset(const arma::mat &v)
{
    return arma::mean(arma::vectorise(v));
}


double DQScalingFactorService::calc_scaling_factor(const arma::mat &v)
{
    return arma::median(arma::vectorise(arma::abs(v))) / common::C_input_obseg_labels;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues, // Labels decon is first in queue, as order of features in row, as order of value columns in input queue
        const datamodel::SVRParameters &params,
        const std::set<size_t> &feat_levels, // feature levels
        const arma::mat &features,
        const arma::mat &labels,
        const datamodel::dq_scaling_factor_container_t &req_factors)
{
    datamodel::dq_scaling_factor_container_t res;
    if (req_factors.empty()) return res;

    const auto labels_level = params.get_decon_level();
    const auto lag = params.get_lag_count();
    LOG4_DEBUG("Calculating " << req_factors.size() << " scaling factors for " << decon_queues.size() << " decon queues, labels level " << labels_level);

    // Labels
    if (slice(req_factors, params.get_dataset_id(), params.get_input_queue_column_name(), std::set{labels_level}, true).size()) {
        const auto dc_offset_labels = calc_dc_offset(labels);
        const auto labels_factor = calc_scaling_factor(labels - dc_offset_labels);
        add(res, ptr<datamodel::DQScalingFactor>(
                0, params.get_dataset_id(), decon_queues.front()->get_input_queue_table_name(), params.get_input_queue_column_name(),
                labels_level, std::numeric_limits<double>::quiet_NaN(), labels_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset_labels));
    }

    if (features.empty()) return res;

    // Features
    const size_t dqct = decon_queues.size();
    const size_t row_len = lag * feat_levels.size();
#pragma omp parallel for num_threads(adj_threads(dqct * feat_levels.size())) collapse(2) schedule(static, 1)
    for (size_t i = 0; i < dqct; ++i) {
        for (size_t j = 0; j < feat_levels.size(); ++j) {
            const auto p_decon_queue = decon_queues[i];
            const size_t feat_level = feat_levels ^ j;
            if (slice(req_factors, params.get_dataset_id(), p_decon_queue->get_input_queue_column_name(), std::set{feat_level}, true).empty()) continue;

            const auto level_feats = features.cols(i * row_len + j * lag, i * row_len + (j + 1) * lag - 1);
            const auto dc_offset = calc_dc_offset(level_feats);
            const auto scaling_factor = calc_scaling_factor(level_feats - dc_offset);
            add(res, ptr<datamodel::DQScalingFactor>(
                    0, params.get_dataset_id(), p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name(), feat_level,
                    scaling_factor, std::numeric_limits<double>::quiet_NaN(), dc_offset));
        }
    }

    return res;
}


void
DQScalingFactorService::scale(
        datamodel::Dataset &dataset,
        const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
        const datamodel::SVRParameters &head_params,
        arma::mat &features,
        arma::mat &labels,
        arma::vec &last_known)
{
    const std::set<size_t> feat_levels = common::get_adjacent_indexes(
            head_params.get_decon_level(), head_params.get_svr_adjacent_levels_ratio(), dataset.get_transformation_levels());
    LOG4_DEBUG(
            "Scaling features " << common::present(features) << ", labels " << common::present(labels) << ", last-known " << common::present(last_known) << ", dataset "
                                << dataset.get_id() << ", decon queue " << *aux_decon_queues.front() << ", params " << head_params << ", feat levels " << feat_levels.size());
    auto missing_factors = check(aux_decon_queues, head_params, feat_levels, dataset.get_dq_scaling_factors());
    if (missing_factors.size() && dataset.get_id()) {
        LOG4_INFO("Dataset doesn't have scaling factors, loading from database.");
        dataset.set_dq_scaling_factors(find_all_by_dataset_id(dataset.get_id()));
        missing_factors = check(aux_decon_queues, head_params, feat_levels, dataset.get_dq_scaling_factors());
    }

    scale(dataset, aux_decon_queues, head_params, missing_factors, features, labels, last_known);

    if (dataset.get_id())
        for_each(std::execution::par_unseq, dataset.get_dq_scaling_factors().cbegin(), dataset.get_dq_scaling_factors().cend(),
                 [&](const auto &s) { if (exists(s)) remove(s); save(s); });
}

void
DQScalingFactorService::scale(
        datamodel::Dataset &dataset,
        const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
        const datamodel::SVRParameters &head_params,
        const datamodel::dq_scaling_factor_container_t &missing_factors,
        arma::mat &features,
        arma::mat &labels,
        arma::vec &last_known)
{
    const auto feat_levels = common::get_adjacent_indexes(
            head_params.get_decon_level(), head_params.get_svr_adjacent_levels_ratio(), dataset.get_transformation_levels());
    if (missing_factors.empty()) goto __scale;

    LOG4_INFO("Recalculating " << missing_factors << " scaling factors for dataset " << dataset.get_id());
#ifdef INTEGRATION_TEST
    dataset.set_dq_scaling_factors(calculate(aux_decon_queues, head_params, feat_levels,
                                             features.rows(0, features.n_rows - INTEGRATION_TEST_VALIDATION_WINDOW - 1),
                                             labels.rows(0, labels.n_rows - INTEGRATION_TEST_VALIDATION_WINDOW - 1),
                                             missing_factors));
#else
    dataset.set_dq_scaling_factors(calculate(aux_decon_queues, head_params, feat_levels, features, labels, missing_factors));
#endif

__scale:
    if (check(aux_decon_queues, head_params, feat_levels, dataset.get_dq_scaling_factors()).size())
        LOG4_THROW("Could not prepare scaling factors for " << head_params);

    // Labels
    const auto sf_labels = slice(dataset.get_dq_scaling_factors(), dataset.get_id(), head_params.get_input_queue_column_name(), std::set{head_params.get_decon_level()});
    labels = (labels - (**sf_labels.cbegin()).get_dc_offset_labels()) / (**sf_labels.cbegin()).get_labels_factor();
    last_known = (last_known - (**sf_labels.cbegin()).get_dc_offset_labels()) / (**sf_labels.cbegin()).get_labels_factor();
    if (!common::sane(labels) || !common::sane(last_known))
        LOG4_THROW("Scaled labels not sane, scaling factor labels " << **sf_labels.cbegin() << ", parameters " << head_params << ", labels " << labels << ", last-knowns " << last_known);

    if (features.empty()) return;

    scale_features(dataset, aux_decon_queues, head_params, features, dataset.get_dq_scaling_factors(), feat_levels);
}

void DQScalingFactorService::scale_features(
        const datamodel::Dataset &dataset, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues,
        const datamodel::SVRParameters &head_params, arma::mat &features,
        const datamodel::dq_scaling_factor_container_t &dq_scaling_factors, const std::set<size_t> &feat_levels)
{// Features
    const auto rowlen = feat_levels.size() * head_params.get_lag_count();
#pragma omp parallel for collapse(2) num_threads(adj_threads(feat_levels.size() * aux_decon_queues.size()))
    for (size_t dqix = 0; dqix < aux_decon_queues.size(); ++dqix) {
        for (size_t i = 0; i < feat_levels.size(); ++i) {
            const size_t l = feat_levels ^ i;
            const size_t col_start = dqix * rowlen + i * head_params.get_lag_count();
            const size_t col_end = dqix * rowlen + (i + 1) * head_params.get_lag_count() - 1;
            const auto sf_features = slice(dq_scaling_factors, dataset.get_id(), head_params.get_input_queue_column_name(), std::set{l}, false, true);
            if (sf_features.empty())
                LOG4_THROW("Couldn't find features scaling factors for dataset id " << dataset.get_id() << ", column name " <<
                                                                                head_params.get_input_queue_column_name() << ", level " << l);
            const auto p_sf = *sf_features.cbegin();
            features.cols(col_start, col_end) = (features.cols(col_start, col_end) - p_sf->get_dc_offset_features()) / p_sf->get_features_factor();
            if (!common::sane<double>(features.cols(col_start, col_end)))
                LOG4_THROW("Scaled features error, scaling factor features " << *p_sf << ", index " << i << ", level " << l << ", feats " << arma::size(features) <<
                                                                             ", start col " << col_start << ", end col " << col_end);
        }
    }
}

bool DQScalingFactorService::match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, const bool overwrite)
{
    auto found_sf = slice(sf, nf.get_dataset_id(), nf.get_input_queue_column_name(), std::set{nf.get_decon_level()}, true);
    if (found_sf.empty()) return false;
    return std::any_of(std::execution::par_unseq, found_sf.begin(), found_sf.end(), [&nf, &overwrite](auto &of) {
        LOG4_TRACE("Setting factor " << *of << " to " << nf);
        bool factor_set = false;
        if (std::isnormal(nf.get_labels_factor()) || overwrite || !std::isnormal(of->get_labels_factor())) {
            of->set_labels_factor(nf.get_labels_factor());
            factor_set = true;
        }
        if (std::isnormal(nf.get_features_factor()) || overwrite || !std::isnormal(of->get_features_factor())) {
            of->set_features_factor(nf.get_features_factor());
            factor_set = true;
        }
        if (common::isnormalz(nf.get_dc_offset_labels()) || overwrite || !common::isnormalz(of->get_dc_offset_labels())) {
            of->set_dc_offset_labels(nf.get_dc_offset_labels());
            factor_set = true;
        }
        if (common::isnormalz(nf.get_dc_offset_features()) || overwrite || !common::isnormalz(of->get_dc_offset_features())) {
            of->set_dc_offset_features(nf.get_dc_offset_features());
            factor_set = true;
        }
        return factor_set;
    });
}

}
}
