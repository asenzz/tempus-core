#pragma once

#include <armadillo>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"

// #define CUDA_SCALING_FACTORS
#define DC_DQ_SCALING_FACTOR 10000

namespace svr {
    namespace dao { class DQScalingFactorDAO; }
    namespace datamodel { class InputQueue; class DeconQueue; class Dataset; class Ensemble; class SVRParameters; }
}

using InputQueue_ptr = std::shared_ptr<svr::datamodel::InputQueue>;
using DeconQueue_ptr = std::shared_ptr<svr::datamodel::DeconQueue>;
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;
using Ensemble_ptr = std::shared_ptr<svr::datamodel::Ensemble>;
using SVRParameters_ptr = std::shared_ptr<svr::datamodel::SVRParameters>;

class DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

namespace svr::business {


class DQScalingFactorService
{
    friend class ::DaoTestFixture_DQScalingFactorScalingUnscaling_Test;
public:
    explicit DQScalingFactorService(svr::dao::DQScalingFactorDAO& dq_scaling_factor_dao) :
        dq_scaling_factor_dao(dq_scaling_factor_dao)
    {}

    bool exists(const DQScalingFactor_ptr &dq_scaling_factor);
    // bool exists_by_dataset_id(const bigint dataset_id); // TODO Implement as needed!

    int save(const DQScalingFactor_ptr& p_dq_scaling_factor);
    int remove(const DQScalingFactor_ptr& p_dq_scaling_factor);

    static svr::datamodel::dq_scaling_factor_container_t slice(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue);
    svr::datamodel::dq_scaling_factor_container_t calculate(const Dataset_ptr &p_dataset);
    svr::datamodel::dq_scaling_factor_container_t prepare_decon_queue_scaling_factors(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue);
    svr::datamodel::dq_scaling_factor_container_t prepare_decon_queue_scaling_factors(const Dataset_ptr &p_dataset, const std::string &input_queue_table_name, const std::string &input_queue_column_name);
private:
    svr::dao::DQScalingFactorDAO &dq_scaling_factor_dao;

    static svr::datamodel::dq_scaling_factor_container_t
    slice(
            const svr::datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::string& input_queue_table_name,
            const std::string& input_queue_column_name);

    static std::pair<std::vector<double>, std::vector<double>>
    do_calculate(const svr::datamodel::DataRow::container::const_iterator &begin, const size_t rows_size);

    static svr::datamodel::dq_scaling_factor_container_t calculate(
            const svr::datamodel::DataRow::container::const_iterator &begin,
            const svr::datamodel::DataRow::container::const_iterator &end,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const size_t dataset_id,
            const size_t levels_ct);

public:
    static datamodel::dq_scaling_factor_container_t slice(
            const svr::datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const std::set<size_t> &feat_levels);

    static datamodel::dq_scaling_factor_container_t calculate(
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const size_t dataset_id,
            const std::set<size_t> &feat_levels,
            const size_t level,
            const size_t lag,
            const arma::mat &features,
            const arma::mat &labels);

    datamodel::dq_scaling_factor_container_t calculate(
            const Dataset_ptr &p_dataset,
            const DeconQueue_ptr &p_decon_queue,
            const SVRParameters_ptr &p_params,
            const std::set<size_t> &feat_levels,
            const size_t expected_factors_ct,
            const arma::mat &features,
            const arma::mat &labels);

    void scale(
            const Dataset_ptr &p_dataset,
            const DeconQueue_ptr &p_decon_queue,
            const SVRParameters_ptr &p_params,
            const std::set<size_t> &feature_levels,
            arma::mat &features,
            arma::mat &labels,
            arma::mat &last_known);

    datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id);
    template<typename T> static T unscale_decon_prediction( // TODO optimize
            T prediction, const size_t level_idx, const svr::datamodel::dq_scaling_factor_container_t &decon_queue_scaling_factors)
    {
        LOG4_BEGIN();
        std::map<size_t, double> label_scales;
        for (const auto &p_scaling_factor: decon_queue_scaling_factors)
            label_scales[p_scaling_factor->get_decon_level()] = p_scaling_factor->get_labels_factor();
        LOG4_TRACE("Decon scaling factor for level " << level_idx << ", label scale " << label_scales[level_idx]);
        if (!level_idx) prediction += label_scales[DC_DQ_SCALING_FACTOR];
        prediction *= label_scales[level_idx];
        return prediction;
        LOG4_END();
    }
};

template<typename T> inline T // TODO Optimize
scale(const T &val, const size_t level, const svr::datamodel::dq_scaling_factor_container_t &scaling_factors, const bool scale_labels = false)
{
    double dc_offset = 0;
    if (!level) {
        for (const auto &sf: scaling_factors)
            if (sf->get_decon_level() == DC_DQ_SCALING_FACTOR)
                dc_offset = scale_labels ? sf->get_labels_factor() : sf->get_features_factor();
        if (!dc_offset) LOG4_ERROR("DC offset not found!");
    }
    for (const auto &sf: scaling_factors)
        if (sf->get_decon_level() == level)
            return (val - dc_offset) / (scale_labels ? sf->get_labels_factor() : sf->get_features_factor());
    throw std::runtime_error("Scaling factor for level " + std::to_string(level) + " not found!");
}

}

using DQScalingTaskService_ptr = std::shared_ptr<svr::business::DQScalingFactorService>;
