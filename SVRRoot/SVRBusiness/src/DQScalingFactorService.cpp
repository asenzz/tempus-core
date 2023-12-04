#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"
#include "DAO/DQScalingFactorDAO.hpp"
#include "spectral_transform.hpp"
#include "fast_cvmd.hpp"
#include "online_emd.hpp"

#ifdef CUDA_SCALING_FACTORS
#include "dq_scaling_factors_service_impl.cuh"
#endif

using namespace svr::common;
using namespace svr::context;

// TODO Rewrite, remove scale input queue and the calling scale dataset

namespace svr {
namespace business {

bool DQScalingFactorService::exists(const DQScalingFactor_ptr &dq_scaling_factor)
{
    return dq_scaling_factor_dao.exists(dq_scaling_factor);
}


int DQScalingFactorService::save(const DQScalingFactor_ptr &p_dq_scaling_factor)
{
    if (not p_dq_scaling_factor->get_id())
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
DQScalingFactorService::slice(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue)
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
    LOG4_DEBUG(
            "Slicing decon queue scaling factors with dataset id " << dataset_id << " input queue table name "
                                                                   << input_queue_table_name << " input queue column name " << input_queue_column_name);

    datamodel::dq_scaling_factor_container_t result;
    for (auto &scaling_factor: scaling_factors)
        if ((!dataset_id or scaling_factor->get_dataset_id() == dataset_id) and
            scaling_factor->get_input_queue_table_name() == input_queue_table_name and
            scaling_factor->get_input_queue_column_name() == input_queue_column_name)
            result.emplace(scaling_factor);

    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
    return result;
}


std::pair<std::vector<double>, std::vector<double>>
DQScalingFactorService::do_calculate(const datamodel::DataRow::container::const_iterator &begin, const size_t rows_size)
{
    LOG4_THROW("Not used!");
#if 0
    if (rows_size < 1) return {{}, {}};

    auto get_coefs = [&](arma::mat &scale_data, const size_t q) -> arma::mat {
#ifdef EMO_DIFF
        if (q == MAIN_DECON_QUEUE_RES_SECS) {
            const size_t diff_offset = q * OFFSET_PRED_MUL;
#pragma omp parallel for default(shared)
            for (size_t row_ix = diff_offset; row_ix < scale_data.n_rows - q; row_ix += q)
                scale_data.row(row_ix / q) = arma::mean(scale_data.rows(row_ix, row_ix + q - 1)) - scale_data.row(row_ix - diff_offset);
        } else
#endif
        for (size_t row_ix = 0; row_ix < scale_data.n_rows - q; row_ix += q)
            scale_data.row(row_ix / q) = arma::mean(scale_data.rows(row_ix, row_ix + q - 1));
        scale_data.shed_rows(scale_data.n_rows / q, scale_data.n_rows - 1);
#ifdef EMO_DIFF
        if (q == QUANTIZE_FIXED) {
#pragma omp parallel for default(shared)
            for (size_t row_ix = 1; row_ix < scale_data.n_rows; ++row_ix)
                scale_data.row(row_ix) = scale_data.row(row_ix) - scale_data.row(row_ix - 1);
            scale_data.shed_row(0);
        }
#endif
        const double dc_offset = arma::mean(scale_data.col(0));
        scale_data.col(0) -= dc_offset;
#if 1
        scale_data = arma::stddev(scale_data) / C_input_obseg_labels;
#else
        scale_data = arma::sort(arma::abs(scale_data));
        const size_t trim_rows = std::round((scale_data.n_rows - 1.) * PROPS.get_scaling_alpha());
        LOG4_DEBUG("Shaving off " << trim_rows << " rows of " << arma::size(scale_data) << ", scaling alpha " << PROPS.get_scaling_alpha() << ", scaling divisor " << C_input_obseg_labels <<
                    ", scale data size " << arma::size(scale_data) << ", max " << arma::max(scale_data) << ", mean " << arma::mean(scale_data) << ", min " << arma::min(scale_data));

        // scale_data = scale_data.row(scale_data.n_rows - 1 - trim_rows) / C_input_obseg_labels;
        scale_data = arma::mean(scale_data) / C_input_obseg_labels;
#endif
        scale_data = arma::join_rows(arma::mat(1, 1, arma::fill::value(dc_offset)), scale_data);
        return scale_data;
    };

    arma::mat scale_data_features(rows_size, begin->get()->get_values().size());
#pragma omp parallel for default(shared)
    for (size_t row_ix = 0; row_ix < rows_size; ++row_ix)
        scale_data_features.row(row_ix) = arma::conv_to<arma::rowvec>::from((begin + row_ix)->get()->get_values());
    arma::mat scale_data_labels = scale_data_features;
    return {arma::conv_to<std::vector<double>>::from(get_coefs(scale_data_features, QUANTIZE_FIXED)),
            arma::conv_to<std::vector<double>>::from(get_coefs(scale_data_labels, MAIN_DECON_QUEUE_RES_SECS))};
#endif
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const datamodel::DataRow::container::const_iterator &begin,
        const datamodel::DataRow::container::const_iterator &end,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name,
        const size_t dataset_id,
        const size_t levels_ct)
{
    LOG4_BEGIN();
    const auto ct = std::distance(begin, end);
    if (ct < 1) {
        LOG4_ERROR("Decon queue is empty");
        return {};
    }
    LOG4_DEBUG(
            "Calculating truncated mean for scaling " << input_queue_table_name << " " << input_queue_column_name << " data from " <<
            begin->get()->get_value_time() << " until " << std::prev(end)->get()->get_value_time());

    datamodel::dq_scaling_factor_container_t result;
    const auto calced_sf = do_calculate(begin, ct);
    if (calced_sf.first.size() != levels_ct + 1 || calced_sf.first.size() != calced_sf.second.size())
        LOG4_THROW("Scaling factors not calculated correctly!");

    result.insert(
            std::make_shared<datamodel::DQScalingFactor>(
                    0, dataset_id, input_queue_table_name, input_queue_column_name, DC_DQ_SCALING_FACTOR, calced_sf.first.front(), calced_sf.second.front()));
    for (size_t l = 0; l < levels_ct; ++l)
        result.insert(
                std::make_shared<datamodel::DQScalingFactor>(
                        0, dataset_id, input_queue_table_name, input_queue_column_name, l, calced_sf.first[l + 1], calced_sf.second[l + 1]));

    return result;
}

std::mutex mx_calc_factors;

// Calculate scaling factors related to decon queue
datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(const Dataset_ptr &p_dataset)
{
    const std::scoped_lock lg_calc_factors(mx_calc_factors);
    const auto level_ct = p_dataset->get_transformation_levels() + 1;
    const auto decon_queues = p_dataset->get_decon_queues();
    if (p_dataset->get_dq_scaling_factors().size() == decon_queues.size() * level_ct) {
        LOG4_DEBUG("Scaling factors already calculated.");
        return p_dataset->get_dq_scaling_factors();
    }

    LOG4_INFO("Calculating scaling factors for dataset " << p_dataset->get_id());
    datamodel::dq_scaling_factor_container_t dq_scaling_factors;
    std::mutex mx;
    __pxt_pfor_i(0, decon_queues.size(),
        const auto p_decon_queue = std::next(decon_queues.begin(), i)->second;
        const auto new_dataset_scaling_factors = calculate(
                p_decon_queue->get_data().begin() + (p_decon_queue->get_data().size() < p_dataset->get_maxpos_residuals_count() ? 0 : p_dataset->get_maxpos_residuals_count()),
                p_decon_queue->get_data().end(),
                p_decon_queue->get_input_queue_table_name(),
                p_decon_queue->get_input_queue_column_name(),
                p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        for (const auto &p_scaling_factor: new_dataset_scaling_factors) {
            const std::scoped_lock l(mx);
            dq_scaling_factors.emplace(p_scaling_factor);
        }
    )

    if (p_dataset->get_id()) {
        auto existing_scaling_factors = find_all_by_dataset_id(p_dataset->get_id());
        for (auto &existing_scaling_factor: existing_scaling_factors)
            remove(existing_scaling_factor);
        for (const auto &scaling_factor: dq_scaling_factors)
            save(scaling_factor);
    } else {
        LOG4_WARN("Not saving, dataset does not exist in database.");
    }
    p_dataset->set_dq_scaling_factors(dq_scaling_factors);
    LOG4_END();

    return dq_scaling_factors;
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::prepare_decon_queue_scaling_factors(
        const Dataset_ptr &p_dataset,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    const size_t all_scale_levels = p_dataset->get_transformation_levels() + 1;
    datamodel::dq_scaling_factor_container_t decon_queue_scaling_factors;
    decon_queue_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(),
            p_dataset->get_id(),
            input_queue_table_name,
            input_queue_column_name);
    if (decon_queue_scaling_factors.size() == all_scale_levels) goto __bail;

    LOG4_WARN("Dataset doesn't have scaling factors " << decon_queue_scaling_factors.size() << ", loading from database.");
    p_dataset->set_dq_scaling_factors(find_all_by_dataset_id(p_dataset->get_id()));

    // Scale main decon queue
    decon_queue_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(),
            p_dataset->get_id(),
            input_queue_table_name,
            input_queue_column_name);

    if (decon_queue_scaling_factors.size() == all_scale_levels) goto __bail;
    LOG4_DEBUG("Recalculating scaling factors.");

    // otherwise calculate them and save to the DB
    if (decon_queue_scaling_factors.size() != all_scale_levels) {
        calculate(p_dataset);
        decon_queue_scaling_factors = slice(
                p_dataset->get_dq_scaling_factors(),
                p_dataset->get_id(),
                input_queue_table_name,
                input_queue_column_name);
        LOG4_DEBUG("Calculated scaling factors count " << decon_queue_scaling_factors.size());
    }

__bail:
    return decon_queue_scaling_factors;
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::prepare_decon_queue_scaling_factors(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue)
{
    return prepare_decon_queue_scaling_factors(p_dataset, p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
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
    for (const auto &sf: scaling_factors)
        if ((!dataset_id or sf->get_dataset_id() == dataset_id) &&
            sf->get_input_queue_table_name() == input_queue_table_name &&
            sf->get_input_queue_column_name() == input_queue_column_name &&
            feat_levels.count(sf->get_decon_level()) &&
            std::isnormal(sf->get_labels_factor()) && std::isnormal(sf->get_features_factor()))
                result.emplace(sf);

    if (feat_levels.count(0)) {
        bool sf_set = false;
        for (auto &sf: scaling_factors) {
            if (sf->get_decon_level() == DC_DQ_SCALING_FACTOR && common::isnormalz(sf->get_labels_factor()) && common::isnormalz(sf->get_features_factor())) {
                sf_set = true;
                result.emplace(sf);
            }
        }
        if (!sf_set) LOG4_WARN(
                "Could not find DC offset for " << common::deep_to_string(feat_levels) << ", dataset id " << dataset_id <<
                ", input queue table name " << input_queue_table_name << ", input queue column name " << input_queue_column_name);
    }
    LOG4_DEBUG("Input decon factors size is " << scaling_factors.size() << " filtered result size is " << result.size());
    return result;
}

bool in(const datamodel::dq_scaling_factor_container_t &cont, const DQScalingFactor_ptr &sf)
{
    for (const auto &p_dqsf: cont) {
        if (sf->get_decon_level() == p_dqsf->get_decon_level() &&
            sf->get_dataset_id() == p_dqsf->get_dataset_id() &&
            sf->get_input_queue_column_name() == p_dqsf->get_input_queue_column_name() &&
            sf->get_input_queue_table_name() == p_dqsf->get_input_queue_table_name()) return true;
    }
    return false;
}

datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name,
        const size_t dataset_id,
        const std::set<size_t> &feat_levels, // feat_levels need to contain level
        const size_t level,
        const size_t lag,
        const arma::mat &features,
        const arma::mat &labels)
{
    LOG4_DEBUG("Calculating scaling factors for " << input_queue_table_name << " " << input_queue_column_name << ", level " << level);

    datamodel::dq_scaling_factor_container_t result;
    // Labels
    double dc_offset_labels = std::numeric_limits<double>::quiet_NaN();
    double labels_factor = std::numeric_limits<double>::quiet_NaN();
    if (!level) {
        arma::colvec labels_copy = arma::vectorise(labels);
        dc_offset_labels = arma::mean(labels_copy);
        labels_copy -= dc_offset_labels;
        labels_factor = arma::median(arma::abs(labels_copy)) / common::C_input_obseg_labels;
    } else {
        labels_factor = arma::median(arma::abs(arma::vectorise(labels))) / common::C_input_obseg_labels;
    }

    // Features
    double dc_offset_feats = std::numeric_limits<double>::quiet_NaN();
    arma::colvec feats_level_0;
    if (feat_levels.count(0)) {
        feats_level_0 = arma::vectorise(features.cols(0, lag - 1));
        dc_offset_feats = arma::mean(feats_level_0);
        feats_level_0 -= dc_offset_feats;
    }
    std::map<size_t, double> feats_factors;
#pragma omp parallel for
    for (size_t i = 0; i < feat_levels.size(); ++i) {
        const size_t l = *std::next(feat_levels.begin(), i);
        const double f_factor = arma::median(arma::abs(l ? arma::vectorise(features.cols(i * lag, i * lag + lag - 1)) : feats_level_0)) / common::C_input_obseg_features;
#pragma omp critical(dq_calculate)
        feats_factors[l] = f_factor;
    }

    if (feat_levels.size() != feats_factors.size()) LOG4_THROW("Scaling factors not calculated correctly!");

    if (common::isnormalz(dc_offset_feats) || common::isnormalz(dc_offset_labels))
        result.insert(
                std::make_shared<datamodel::DQScalingFactor>(
                        0, dataset_id, input_queue_table_name, input_queue_column_name, DC_DQ_SCALING_FACTOR, dc_offset_feats, dc_offset_labels));
    for (const size_t l: feat_levels)
        result.insert(
                std::make_shared<datamodel::DQScalingFactor>(
                        0, dataset_id, input_queue_table_name, input_queue_column_name, l, feats_factors[l], l == level ? labels_factor : std::numeric_limits<double>::quiet_NaN()));
    return result;
}


datamodel::dq_scaling_factor_container_t
DQScalingFactorService::calculate(
        const Dataset_ptr &p_dataset,
        const DeconQueue_ptr &p_decon_queue,
        const SVRParameters_ptr &p_params,
        const std::set<size_t> &feat_levels,
        const size_t expected_factors_ct,
        const arma::mat &features,
        const arma::mat &labels)
{
    const std::scoped_lock lk(mx_calc_factors);
    auto dq_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name(), feat_levels);
    if (dq_scaling_factors.size() == expected_factors_ct) {
        LOG4_DEBUG("Scaling factors already calculated.");
        return dq_scaling_factors;
    }
    LOG4_INFO("Calculating scaling factors for dataset " << p_dataset->get_id());
    dq_scaling_factors = calculate(
            p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name(), p_dataset->get_id(), feat_levels, p_params->get_decon_level(), p_params->get_lag_count(), features, labels);
    p_dataset->add_dq_scaling_factors(dq_scaling_factors);

    if (p_dataset->get_id()) {
        for (const auto &scaling_factor: p_dataset->get_dq_scaling_factors()) {
            remove(scaling_factor);
            save(scaling_factor);
        }
    } else
        LOG4_WARN("Not saving, dataset does not exist in database.");

    LOG4_DEBUG("Calculated scaling factors " << common::deep_to_string(dq_scaling_factors));

    return dq_scaling_factors;
}


void
DQScalingFactorService::scale(
        const Dataset_ptr &p_dataset,
        const DeconQueue_ptr &p_decon_queue,
        const SVRParameters_ptr &p_params,
        const std::set<size_t> &feat_levels,
        arma::mat &features,
        arma::mat &labels,
        arma::mat &last_known)
{
    LOG4_DEBUG("Scaling features " << common::present(features) << ", labels " << common::present(labels) << ", last known " << common::present(last_known) << ", dataset " <<
                   p_dataset->to_string() << ", decon queue " << p_decon_queue->to_string() << ", params " << p_params->to_string() << ", feat levels " << feat_levels.size());
    const size_t expected_factors_ct = feat_levels.size() + (feat_levels.count(0) || !p_params->get_decon_level());
    datamodel::dq_scaling_factor_container_t decon_queue_scaling_factors = slice(
            p_dataset->get_dq_scaling_factors(),
            p_dataset->get_id(),
            p_decon_queue->get_input_queue_table_name(),
            p_decon_queue->get_input_queue_column_name(),
            feat_levels);
    if (decon_queue_scaling_factors.size() == expected_factors_ct) goto __bail;

    LOG4_INFO("Dataset doesn't have scaling factors, loading from database.");
    // Load from database if existing
    decon_queue_scaling_factors = slice(
            find_all_by_dataset_id(p_dataset->get_id()),
            p_dataset->get_id(),
            p_decon_queue->get_input_queue_table_name(),
            p_decon_queue->get_input_queue_column_name(),
            feat_levels);
    if (decon_queue_scaling_factors.size() == expected_factors_ct) goto __bail;
    LOG4_INFO("Recalculating scaling factors.");

    // otherwise calculate the needed ones and save to database
    decon_queue_scaling_factors = calculate(p_dataset, p_decon_queue, p_params, feat_levels, expected_factors_ct, features.rows(0, features.n_rows - 1 - MANIFOLD_TEST_VALIDATION_WINDOW), labels.rows(0, labels.n_rows - 1 - MANIFOLD_TEST_VALIDATION_WINDOW));

__bail:
    if (decon_queue_scaling_factors.size() != expected_factors_ct) {
        LOG4_ERROR("Could not prepare scaling factors for " << p_params->to_string());
        return;
    }
    double labels_scaling_factor = std::numeric_limits<double>::quiet_NaN(), dc_offset_labels = std::numeric_limits<double>::quiet_NaN(), dc_offset_features = std::numeric_limits<double>::quiet_NaN();
    std::map<size_t, double> features_scaling_factors;
    for (const auto &dq_sf: decon_queue_scaling_factors) {
        if (dq_sf->get_input_queue_column_name() != p_decon_queue->get_input_queue_column_name() || dq_sf->get_input_queue_table_name() != p_decon_queue->get_input_queue_table_name() ||
                dq_sf->get_dataset_id() != p_params->get_dataset_id() || dq_sf->get_dataset_id() != p_dataset->get_id()) continue;

        if (std::isnormal(dq_sf->get_labels_factor()) && dq_sf->get_decon_level() == p_params->get_decon_level())
            labels_scaling_factor = dq_sf->get_labels_factor();
        if (std::isnormal(dq_sf->get_features_factor()) && feat_levels.count(dq_sf->get_decon_level()))
            features_scaling_factors[dq_sf->get_decon_level()] = dq_sf->get_features_factor();
        if (dq_sf->get_decon_level() == DC_DQ_SCALING_FACTOR && (common::isnormalz(dq_sf->get_features_factor()) || common::isnormalz(dq_sf->get_labels_factor()))) {
            dc_offset_features = dq_sf->get_features_factor();
            dc_offset_labels = dq_sf->get_labels_factor();
        }
    }
    if (!p_params->get_decon_level()) {
        if (common::isnormalz(dc_offset_labels)) {
            labels -= dc_offset_labels;
            last_known -= dc_offset_labels;
        } else
            LOG4_THROW("Labels DC offset not found for " << p_params->to_string());
    }
    labels /= labels_scaling_factor;
    last_known /= labels_scaling_factor;
    if (!common::sane(labels) || !common::sane(last_known))
        LOG4_ERROR("Scaled labels not sane, scaling factor labels " << labels_scaling_factor << ", level " << p_params->get_decon_level() << ", offset " << dc_offset_labels);
#pragma omp parallel for
    for (size_t i = 0; i < feat_levels.size(); ++i) {
        const size_t l = *std::next(feat_levels.begin(), i);
        if (!l) {
            if (common::isnormalz(dc_offset_features))
                features.cols(i * p_params->get_lag_count(), (i + 1) * p_params->get_lag_count() - 1) -= dc_offset_features;
            else
                LOG4_THROW("Features DC offset not found for " << p_params->to_string());
        }
        const auto sf_feats = features_scaling_factors[l];
        if (!std::isnormal(sf_feats)) LOG4_ERROR("Features scaling factor for level " << l << " not normal " << sf_feats);
        features.cols(i * p_params->get_lag_count(), (i + 1) * p_params->get_lag_count() - 1) /= sf_feats;
        if (!common::sane<double>(features.cols(i * p_params->get_lag_count(), (i + 1) * p_params->get_lag_count() - 1)))
            LOG4_ERROR("Scaled features error, scaling factor features " << sf_feats << ", index " << i << ", level " << l << ", offset " << dc_offset_features);
    }
}


}
}
