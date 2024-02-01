#pragma once

#include <armadillo>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/concurrent_map.h>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"
#include "model/Dataset.hpp"
#include "model/DeconQueue.hpp"
#include "model/SVRParameters.hpp"

class DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

namespace svr {

namespace dao { class DQScalingFactorDAO; }

namespace business {

constexpr unsigned DC_INDEX = 10000;

class DQScalingFactorService
{
    friend class ::DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

public:
    explicit DQScalingFactorService(dao::DQScalingFactorDAO &dq_scaling_factor_dao) :
            dq_scaling_factor_dao(dq_scaling_factor_dao)
    {}

    bool exists(const DQScalingFactor_ptr &dq_scaling_factor);
    // bool exists_by_dataset_id(const bigint dataset_id); // TODO Implement as needed!

    int save(const DQScalingFactor_ptr &p_dq_scaling_factor);

    int remove(const DQScalingFactor_ptr &p_dq_scaling_factor);

    static datamodel::dq_scaling_factor_container_t slice(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue);

private:
    dao::DQScalingFactorDAO &dq_scaling_factor_dao;

    static datamodel::dq_scaling_factor_container_t
    slice(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name);

public:
    datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id);

    static datamodel::dq_scaling_factor_container_t
    slice(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const std::set<size_t> &feat_levels);

    static datamodel::dq_scaling_factor_container_t
    slice(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::deque <datamodel::DeconQueue_ptr> &decon_queues,
            const std::set<size_t> &feat_levels);

    static datamodel::dq_scaling_factor_container_t
    check(const std::deque <datamodel::DeconQueue_ptr> &decon_queues, const datamodel::SVRParameters_ptr &p_head_params, const std::set<size_t> feat_levels,
          const datamodel::dq_scaling_factor_container_t &scaling_factors);

    static datamodel::dq_scaling_factor_container_t
    calculate(
            const std::deque <datamodel::DeconQueue_ptr> &decon_queues,
            const datamodel::SVRParameters_ptr &p_params,
            const std::set<size_t> &feat_levels,
            const arma::mat &features,
            const arma::mat &labels,
            const datamodel::dq_scaling_factor_container_t &req_factors);

    datamodel::dq_scaling_factor_container_t
    calculate(
            const datamodel::Dataset_ptr &p_dataset,
            const std::deque <datamodel::DeconQueue_ptr> &decon_queues,
            const datamodel::SVRParameters_ptr &p_params,
            const arma::mat &features,
            const arma::mat &labels,
            std::set<size_t> feat_levels = {},
            datamodel::dq_scaling_factor_container_t req_factors = {});

    void
    scale(const datamodel::Dataset_ptr &p_dataset, const std::deque <datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
          const datamodel::SVRParameters_ptr &p_head_params, arma::mat &features, arma::mat &labels, arma::mat &last_known);

    void scale(const datamodel::Dataset_ptr &p_dataset, const std::deque <datamodel::DeconQueue_ptr> &aux_decon_queues, const datamodel::SVRParameters_ptr &p_head_params, arma::mat &features);

    static void scale(
            const std::deque <datamodel::DeconQueue_ptr> &aux_decon_queues, const svr::datamodel::SVRParameters_ptr &p_head_params, arma::mat &features,
            const std::set<size_t> &feat_levels, const tbb::concurrent_map<size_t, double> &dc_offset_features,
            const tbb::concurrent_map<std::pair<size_t, size_t>, double> &features_scaling_factors);

    template<typename T> static T
    unscale(T prediction, const size_t levix, const datamodel::dq_scaling_factor_container_t &scaling_factors)
    {
        LOG4_BEGIN();
        std::map<size_t, double> label_scales;
        for (const auto &p_scaling_factor: scaling_factors)
            label_scales[p_scaling_factor->get_decon_level()] = p_scaling_factor->get_labels_factor();
        LOG4_TRACE("Decon scaling factor for level " << levix << ", label scale " << label_scales[levix]);
        if (!levix) prediction += label_scales[DC_INDEX];
        prediction *= label_scales[levix];
        return prediction;
    }
};


}
}