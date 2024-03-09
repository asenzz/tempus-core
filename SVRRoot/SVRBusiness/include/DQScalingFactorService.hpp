#pragma once

#include <armadillo>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/concurrent_map.h>
#include <set>
#include <execution>
#include <algorithm>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"
#include "model/Dataset.hpp"
#include "model/DeconQueue.hpp"
#include "model/SVRParameters.hpp"
#include "model/Entity.hpp"
#include "util/string_utils.hpp"

class DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

namespace svr {

namespace dao { class DQScalingFactorDAO; }

namespace business {


class DQScalingFactorService
{
    friend class ::DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

    dao::DQScalingFactorDAO &dq_scaling_factor_dao;

public:
    explicit DQScalingFactorService(dao::DQScalingFactorDAO &dq_scaling_factor_dao) :
            dq_scaling_factor_dao(dq_scaling_factor_dao)
    {}

    bool exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor);
    // bool exists_by_dataset_id(const bigint dataset_id); // TODO Implement as needed!

    int save(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor);

    int remove(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor);

    static datamodel::dq_scaling_factor_container_t slice(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue);

    static bool match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, const bool overwrite);
    datamodel::dq_scaling_factor_container_t find_all_by_dataset_id(const bigint dataset_id);

    static datamodel::dq_scaling_factor_container_t
    slice(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::string &input_queue_column_name,
            const std::set<size_t> &levels,
            const bool match_missing = false,
            const bool check_features_only = false);

    static datamodel::dq_scaling_factor_container_t
    slice(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t dataset_id,
            const std::deque <datamodel::DeconQueue_ptr> &decon_queues,
            const std::set<size_t> &feat_levels);

    static datamodel::dq_scaling_factor_container_t
    check(const std::deque <datamodel::DeconQueue_ptr> &decon_queues, const datamodel::SVRParameters &head_params, const std::set<size_t> feat_levels,
          const datamodel::dq_scaling_factor_container_t &scaling_factors);

    static double calc_scaling_factor(const arma::mat &v);
    static double calc_dc_offset(const arma::mat &v);

    static datamodel::dq_scaling_factor_container_t
    calculate(
            const std::deque <datamodel::DeconQueue_ptr> &decon_queues,
            const datamodel::SVRParameters &params,
            const std::set<size_t> &feat_levels,
            const arma::mat &features,
            const arma::mat &labels,
            const datamodel::dq_scaling_factor_container_t &req_factors);

    void scale(datamodel::Dataset &dataset, const std::deque <datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
          const datamodel::SVRParameters &head_params, arma::mat &features, arma::mat &labels, arma::vec &last_known);

    static void scale(
            datamodel::Dataset &dataset,
            const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, // labels and features are created from aux decon data
            const datamodel::SVRParameters &head_params,
            const datamodel::dq_scaling_factor_container_t &missing_factors,
            arma::mat &features,
            arma::mat &labels,
            arma::vec &last_known);

    static void
    scale_features(const datamodel::Dataset &dataset, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues,
                   const datamodel::SVRParameters &head_params,
                   arma::mat &features, const datamodel::dq_scaling_factor_container_t &dq_scaling_factors, const std::set<size_t> &feat_levels);

    template<typename T> static inline T
    unscale(const T &labels, const datamodel::DQScalingFactor &sf)
    {
        return labels * sf.get_labels_factor() + sf.get_dc_offset_labels();
    }

    template<typename T> static T
    unscale(const T &labels, const size_t levix, const std::string &input_queue_column_name, const datamodel::dq_scaling_factor_container_t &scaling_factors)
    {
        const auto found_sf = slice(scaling_factors, 0, input_queue_column_name, std::set{levix});
        if (found_sf.empty()) LOG4_THROW("Could not find scaling factor for " << levix << " " << input_queue_column_name);
        return unscale(labels, **found_sf.cbegin());
    }

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, const bool overwrite = false);

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor_ptr &p_new_sf, const bool overwrite = false);

    static datamodel::dq_scaling_factor_container_t
    slice(const datamodel::dq_scaling_factor_container_t &scaling_factors, const size_t dataset_id, const std::string &input_queue_column_name);
};


}
}