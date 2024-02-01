#include "appcontext.hpp"
#include "DAO/DQScalingFactorDAO.hpp"
#include "spectral_transform.hpp"
#include "DQScalingFactorService.hpp"

#ifdef CUDA_SCALING_FACTORS
#include "dq_scaling_factors_service_impl.cuh"
#endif


namespace svr {
namespace business {

bool DQScalingFactorService::exists(const DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.exists(dq_scaling_factor);
}


int DQScalingFactorService::save(const DQScalingFactor_ptr &p_dq_scaling_factor)
{
    if (!p_dq_scaling_factor->get_id())
        p_dq_scaling_factor->set_id(dq_scaling_factor_dao.get_next_id());
    return dq_scaling_factor_dao.save(p_dq_scaling_factor);
}


int DQScalingFactorService::remove(const DQScalingFactor_ptr &dq_scaling_factor)
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
    return slice(p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const size_t dataset_id,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    LOG4_DEBUG("Slicing decon queue scaling factors with dataset id " << dataset_id << " input queue table name "
                                                                      << input_queue_table_name << " input queue column name " << input_queue_column_name);

    datamodel::dq_scaling_factor_container_t result;
    for_each(std::execution::par_unseq, scaling_factors.begin(), scaling_factors.end(), [&](const auto sf) {
        if ((!dataset_id or sf->get_dataset_id() == dataset_id) and
            sf->get_input_queue_table_name() == input_queue_table_name and
            sf->get_input_queue_column_name() == input_queue_column_name) {
            result.emplace(sf);
        }
    });

    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
    return result;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::slice(
        const datamodel::dq_scaling_factor_container_t &scaling_factors,
        const size_t dataset_id,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name,
        const std::set<size_t> &feat_levels)
{
    LOG4_DEBUG("Slicing decon queue scaling factors with dataset id " << dataset_id << ", input queue table name " << input_queue_table_name <<
                                                                      ", input queue column name " << input_queue_column_name);

    datamodel::dq_scaling_factor_container_t result;
    for_each(std::execution::par_unseq, scaling_factors.begin(), scaling_factors.end(), [&](const auto &sf) {
        if ((!dataset_id or sf->get_dataset_id() == dataset_id) &&
            sf->get_input_queue_table_name() == input_queue_table_name &&
            sf->get_input_queue_column_name() == input_queue_column_name &&
            feat_levels.count(sf->get_decon_level()) &&
            std::isnormal(sf->get_labels_factor()) && std::isnormal(sf->get_features_factor())) {
            result.emplace(sf);
        }
    });

    if (not feat_levels.count(0)) goto __bail;

    if (std::none_of(std::execution::par_unseq, scaling_factors.begin(), scaling_factors.end(), [&](const auto &sf) {
        if (sf->get_decon_level() == DC_INDEX && common::isnormalz(sf->get_labels_factor()) && common::isnormalz(sf->get_features_factor())) {
            result.emplace(sf);
            return true;
        } else
            return false;
    }))
    LOG4_WARN("Could not find DC offset for " << common::to_string(feat_levels) << ", dataset id " << dataset_id <<
                                              ", input queue table name " << input_queue_table_name << ", input queue column name " << input_queue_column_name);

__bail:
    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
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
#pragma omp parallel for
    for (const auto &d: decon_queues) {
        const auto dr = slice(scaling_factors, dataset_id, d->get_input_queue_table_name(), d->get_input_queue_column_name(), feat_levels);
#pragma omp critical
        r.insert(dr.begin(), dr.end());
    }
    return r;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::check(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const datamodel::SVRParameters_ptr &p_head_params,
        const std::set<size_t> feat_levels,
        const datamodel::dq_scaling_factor_container_t &scaling_factors)
{
    datamodel::dq_scaling_factor_container_t missing;
    if (std::none_of(std::execution::par_unseq, scaling_factors.begin(), scaling_factors.end(), [&](const auto &s) {
        return s->get_input_queue_table_name() == decon_queues.front()->get_input_queue_table_name()
               && s->get_input_queue_column_name() == p_head_params->get_input_queue_column_name()
               && s->get_decon_level() == p_head_params->get_decon_level()
               && common::isnormalz(s->get_labels_factor());
    })) missing.emplace(std::make_shared<datamodel::DQScalingFactor>(
                0, p_head_params->get_dataset_id(), decon_queues.front()->get_input_queue_table_name(), p_head_params->get_input_queue_column_name(), p_head_params->get_decon_level()));

#pragma omp parallel for
    for (const auto &dq: decon_queues) {
        tbb::concurrent_set<size_t> found_feat_levels;
        for_each(std::execution::par_unseq, scaling_factors.begin(), scaling_factors.end(), [&](const auto &s) {
            for (const auto fl: feat_levels)
                if (s->get_input_queue_table_name() == dq->get_input_queue_table_name()
                    && s->get_input_queue_column_name() == dq->get_input_queue_column_name()
                    && s->get_decon_level() == fl
                    && common::isnormalz(s->get_features_factor()))
                    found_feat_levels.emplace(fl);
        });

        if (feat_levels.size() == found_feat_levels.size()) continue;

        for_each(std::execution::par_unseq, feat_levels.begin(), feat_levels.end(), [&](const auto f) {
            if (std::none_of(found_feat_levels.begin(), found_feat_levels.end(), [f](const auto p) { return p == f; })) {
                missing.emplace(std::make_shared<datamodel::DQScalingFactor>(
                        0, dq->get_dataset_id(), dq->get_input_queue_table_name(), dq->get_input_queue_column_name(), f,
                        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));
            }
        });
    }
    return missing;
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const datamodel::SVRParameters_ptr &p_params,
        const std::set<size_t> &feat_levels, // feat_levels need to contain level
        const arma::mat &features,
        const arma::mat &labels,
        const datamodel::dq_scaling_factor_container_t &req_factors)
{
    if (req_factors.empty()) return {};
    const auto labels_level = p_params->get_decon_level();
    const auto lag = p_params->get_lag_count();
    LOG4_DEBUG("Calculating " << req_factors.size() << " scaling factors for " << decon_queues.size() << " decon queues, labels level " << labels_level);

    datamodel::dq_scaling_factor_container_t re;
    // Labels
    double labels_factor = std::numeric_limits<double>::quiet_NaN();
    if (!p_params->get_decon_level()) {
        arma::colvec labels_copy = arma::vectorise(labels);
        const auto dc_offset_labels = arma::mean(labels_copy);
        std::make_shared<datamodel::DQScalingFactor>(
                0, p_params->get_dataset_id(), decon_queues.front()->get_input_queue_table_name(), p_params->get_input_queue_column_name(),
                DC_INDEX, std::numeric_limits<double>::quiet_NaN(), dc_offset_labels)->add(re);
        labels_copy -= dc_offset_labels;
        labels_factor = arma::median(arma::abs(labels_copy)) / common::C_input_obseg_labels;
    } else {
        labels_factor = arma::median(arma::abs(arma::vectorise(labels))) / common::C_input_obseg_labels;
    }
    datamodel::DQScalingFactor(
            0, p_params->get_dataset_id(), decon_queues.front()->get_input_queue_table_name(), p_params->get_input_queue_column_name(),
            p_params->get_decon_level(), std::numeric_limits<double>::quiet_NaN(), labels_factor).add(re);

    // Features
    const size_t dqct = decon_queues.size();
    const size_t row_len = lag * feat_levels.size();
    std::deque<arma::colvec> feats_level_0(dqct);
    std::deque<double> dc_offset_feats(dqct, std::numeric_limits<double>::quiet_NaN());
    if (feat_levels.count(0)) {
#pragma omp parallel for
        for (size_t i = 0; i < dqct; ++i) {
            feats_level_0[i] = arma::vectorise(features.cols(i * row_len, i * row_len + row_len - 1));
            dc_offset_feats[i] = arma::mean(feats_level_0[i]);
            feats_level_0[i] -= dc_offset_feats[i];
        }
    }
    std::deque<std::map<size_t, double>> feat_factors(dqct);
#pragma omp parallel for
    for (size_t i = 0; i < dqct; ++i) {
#pragma omp parallel for
        for (size_t j = 0; j < feat_levels.size(); ++j) {
            const size_t l = feat_levels ^ j;
            const double f_factor =
                    arma::median(arma::abs(l ? arma::vectorise(features.cols(i * row_len + j * lag, i * row_len + j * lag + lag - 1)) : feats_level_0[i])) /
                    common::C_input_obseg_features;
#pragma omp critical
            feat_factors[i][l] = f_factor;
        }

        if (common::isnormalz(dc_offset_feats[i]))
#pragma omp critical
            datamodel::DQScalingFactor(
                    0, p_params->get_dataset_id(), decon_queues[i]->get_input_queue_table_name(), decon_queues[i]->get_input_queue_column_name(),
                    DC_INDEX, dc_offset_feats[i], std::numeric_limits<double>::quiet_NaN()).add(re);
        for (const size_t l: feat_levels)
#pragma omp critical
            datamodel::DQScalingFactor(
                    0, p_params->get_dataset_id(), decon_queues[i]->get_input_queue_table_name(), decon_queues[i]->get_input_queue_column_name(), l, feat_factors[i][l],
                    std::numeric_limits<double>::quiet_NaN()).add(re);
    }

    return re;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const datamodel::Dataset_ptr &p_dataset,
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const datamodel::SVRParameters_ptr &p_params,
        const arma::mat &features,
        const arma::mat &labels,
        std::set<size_t> feat_levels,
        datamodel::dq_scaling_factor_container_t req_factors)
{
    if (feat_levels.empty()) feat_levels = common::get_adjacent_indexes(p_params->get_decon_level(), p_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
    LOG4_DEBUG("Scaling features " << common::present(features) << ", labels " << common::present(labels) << ", dataset " <<
                                   *p_dataset << ", decon queue " << *decon_queues.front() << ", params " << *p_params << ", feat levels " << feat_levels.size());

    const std::scoped_lock lk(p_dataset->get_dq_scaling_factors_calc_mutex());
    datamodel::dq_scaling_factor_container_t dq_scaling_factors;
    if (req_factors.empty()) {
        dq_scaling_factors = slice(p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), decon_queues, feat_levels);
        req_factors = check(decon_queues, p_params, feat_levels, dq_scaling_factors);
    }

    if (req_factors.empty()) {
        LOG4_DEBUG("Scaling factors for labels level " << p_params->get_decon_level() << ", feat levels " << common::to_string(feat_levels) << " already calculated.");
        return dq_scaling_factors;
    }
    LOG4_INFO("Calculating scaling factors for dataset " << p_dataset->get_id());
    dq_scaling_factors = calculate(decon_queues, p_params, feat_levels, features, labels, req_factors);
    p_dataset->add_dq_scaling_factors(dq_scaling_factors);

    if (p_dataset->get_id())
        for_each(std::execution::par_unseq, p_dataset->get_dq_scaling_factors().begin(), p_dataset->get_dq_scaling_factors().end(),
                 [&](const auto &s) { remove(s); save(s); });
    else
        LOG4_WARN("Not saving, dataset does not exist in database.");

    LOG4_DEBUG("Calculated scaling factors " << dq_scaling_factors);

    return dq_scaling_factors;
}


void
DQScalingFactorService::scale(
        const datamodel::Dataset_ptr &p_dataset,
        const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
        const datamodel::SVRParameters_ptr &p_head_params,
        arma::mat &features,
        arma::mat &labels,
        arma::mat &last_known)
{
    const std::set<size_t> feat_levels = common::get_adjacent_indexes(
            p_head_params->get_decon_level(), p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
    LOG4_DEBUG(
            "Scaling features " << common::present(features) << ", labels " << common::present(labels) << ", last known " << common::present(last_known) << ", dataset "
                                << *p_dataset << ", decon queue " << *aux_decon_queues.front() << ", params " << *p_head_params << ", feat levels " << feat_levels.size());
    datamodel::dq_scaling_factor_container_t missing_factors, dq_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(),
            p_dataset->get_id(),
            aux_decon_queues,
            feat_levels);
    if (check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors).empty()) goto __scale;

    LOG4_INFO("Dataset doesn't have scaling factors, loading from database.");
    dq_scaling_factors = slice(
            find_all_by_dataset_id(p_dataset->get_id()), // Load from database if existing
            p_dataset->get_id(), aux_decon_queues, feat_levels);

    missing_factors = check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors);
    if (missing_factors.empty()) goto __scale;

    LOG4_INFO("Recalculating " << missing_factors.size() << " scaling factors.");
    // otherwise calculate the needed ones and save to database
    datamodel::DQScalingFactor::add(dq_scaling_factors, calculate(
            p_dataset, aux_decon_queues, p_head_params,
            features.rows(0, features.n_rows - 1 - MANIFOLD_TEST_VALIDATION_WINDOW),
            labels.rows(0, labels.n_rows - 1 - MANIFOLD_TEST_VALIDATION_WINDOW),
            feat_levels, missing_factors));

__scale:
    if (!check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors).empty())
        LOG4_THROW("Could not prepare scaling factors for " << *p_head_params);

    std::atomic<double> labels_scaling_factor, dc_offset_labels;
    tbb::concurrent_map<size_t, double> dc_offset_features;
    tbb::concurrent_map<std::pair<size_t, size_t>, double> features_scaling_factors;
#pragma omp parallel for collapse(2)
    for (size_t dqix = 0; dqix < aux_decon_queues.size(); ++dqix) {
        for (size_t sfix = 0; sfix < dq_scaling_factors.size(); ++sfix) {
            const auto sf = dq_scaling_factors ^ sfix;

            if (sf->get_input_queue_column_name() != aux_decon_queues[dqix]->get_input_queue_column_name() ||
                sf->get_input_queue_table_name() != aux_decon_queues[dqix]->get_input_queue_table_name() ||
                sf->get_dataset_id() != p_head_params->get_dataset_id() ||
                sf->get_dataset_id() != p_dataset->get_id())
                continue;

            if (std::isnormal(sf->get_labels_factor()) && sf->get_decon_level() == p_head_params->get_decon_level())
                labels_scaling_factor.store(sf->get_labels_factor(), std::memory_order_relaxed);

            if (std::isnormal(sf->get_features_factor()) && feat_levels.count(sf->get_decon_level()))
                features_scaling_factors[{dqix, sf->get_decon_level()}] = sf->get_features_factor();

            if (common::isnormalz(sf->get_features_factor()) && sf->get_decon_level() == DC_INDEX)
                dc_offset_features[dqix] = sf->get_features_factor();

            if (common::isnormalz(sf->get_labels_factor()) && sf->get_decon_level() == DC_INDEX)
                dc_offset_labels.store(sf->get_labels_factor(), std::memory_order_relaxed);

        }
    }

    if (p_head_params->get_decon_level() == 0) {
        if (common::isnormalz(dc_offset_labels.load(std::memory_order_relaxed))) {
            labels -= dc_offset_labels;
            last_known -= dc_offset_labels;
        } else
            LOG4_THROW("Labels DC offset not found for " << *p_head_params);
    }

    labels /= labels_scaling_factor;
    last_known /= labels_scaling_factor;
    if (!common::sane(labels) || !common::sane(last_known))
        LOG4_ERROR("Scaled labels not sane, scaling factor labels " << labels_scaling_factor << ", level " << p_head_params->get_decon_level() << ", offset " << dc_offset_labels);

    scale(aux_decon_queues, p_head_params, features, feat_levels, dc_offset_features, features_scaling_factors);
}


void DQScalingFactorService::scale(
        const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, const svr::datamodel::SVRParameters_ptr &p_head_params, arma::mat &features,
        const std::set<size_t> &feat_levels, const tbb::concurrent_map<size_t, double> &dc_offset_features,
        const tbb::concurrent_map<std::pair<size_t, size_t>, double> &features_scaling_factors)
{
    const auto rowlen = feat_levels.size() * p_head_params->get_lag_count();
#pragma omp parallel for collapse(2)
    for (size_t dqix = 0; dqix < aux_decon_queues.size(); ++dqix) {
        for (size_t i = 0; i < feat_levels.size(); ++i) {
            const size_t l = feat_levels ^ i;
            const size_t col_start = dqix * rowlen + i * p_head_params->get_lag_count();
            const size_t col_end = dqix * rowlen + (i + 1) * p_head_params->get_lag_count() - 1;
            if (l == 0) {
                if (common::isnormalz(dc_offset_features.at(dqix)))
                    features.cols(col_start, col_end) -= dc_offset_features.at(dqix);
                else
                    LOG4_THROW("Features DC offset not found for " << *p_head_params);
            }
            const auto sf_feats = features_scaling_factors.at({dqix, l});
            if (!std::isnormal(sf_feats)) LOG4_ERROR("Features scaling factor for level " << l << " not normal " << sf_feats);
            features.cols(col_start, col_end) /= sf_feats;
            if (!common::sane<double>(features.cols(col_start, col_end)))
                LOG4_ERROR("Scaled features error, scaling factor features " << sf_feats << ", index " << i << ", level " << l << ", offset " << dc_offset_features.at(dqix));
        }
    }
}


void
DQScalingFactorService::scale(
        const datamodel::Dataset_ptr &p_dataset,
        const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
        const datamodel::SVRParameters_ptr &p_head_params,
        arma::mat &features)
{
    const std::set<size_t> feat_levels = common::get_adjacent_indexes(
            p_head_params->get_decon_level(), p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
    LOG4_DEBUG(
            "Scaling features " << common::present(features) << ", dataset " << *p_dataset << ", decon queue " << *aux_decon_queues.front() << ", params " << *p_head_params << ", feat levels " << feat_levels.size());
    datamodel::dq_scaling_factor_container_t missing_factors, dq_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(),
            p_dataset->get_id(),
            aux_decon_queues,
            feat_levels);
    if (check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors).empty()) goto __scale;

    LOG4_INFO("Dataset doesn't have scaling factors, loading from database.");
    dq_scaling_factors = slice(
            find_all_by_dataset_id(p_dataset->get_id()), // Load from database if existing
            p_dataset->get_id(), aux_decon_queues, feat_levels);

    missing_factors = check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors);
    if (missing_factors.empty()) goto __scale;

    LOG4_THROW("Missing factors for " << *p_head_params);

__scale:
    if (!check(aux_decon_queues, p_head_params, feat_levels, dq_scaling_factors).empty())
        LOG4_THROW("Could not prepare scaling factors for " << *p_head_params);

    tbb::concurrent_map<size_t, double> dc_offset_features;
    tbb::concurrent_map<std::pair<size_t, size_t>, double> features_scaling_factors;
#pragma omp parallel for collapse(2)
    for (size_t dqix = 0; dqix < aux_decon_queues.size(); ++dqix) {
        for (size_t sfix = 0; sfix < dq_scaling_factors.size(); ++sfix) {
            const auto sf = dq_scaling_factors ^ sfix;

            if (sf->get_input_queue_column_name() != aux_decon_queues[dqix]->get_input_queue_column_name() ||
                sf->get_input_queue_table_name() != aux_decon_queues[dqix]->get_input_queue_table_name() ||
                sf->get_dataset_id() != p_head_params->get_dataset_id() ||
                sf->get_dataset_id() != p_dataset->get_id())
                continue;

            if (std::isnormal(sf->get_features_factor()) && feat_levels.count(sf->get_decon_level()))
                features_scaling_factors[{dqix, sf->get_decon_level()}] = sf->get_features_factor();

            if (common::isnormalz(sf->get_features_factor()) && sf->get_decon_level() == DC_INDEX)
                dc_offset_features[dqix] = sf->get_features_factor();
        }
    }

    scale(aux_decon_queues, p_head_params, features, feat_levels, dc_offset_features, features_scaling_factors);
}


}
}
