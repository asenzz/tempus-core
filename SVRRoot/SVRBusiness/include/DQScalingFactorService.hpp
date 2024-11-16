#pragma once

#include <armadillo>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/concurrent_map.h>
#include <set>
#include <execution>
#include <algorithm>
#include <oneapi/tbb/concurrent_set.h>
#include <unordered_set>
#include "model/DQScalingFactor.hpp"
#include "model/DataRow.hpp"
#include "model/Dataset.hpp"
#include "model/DeconQueue.hpp"
#include "model/SVRParameters.hpp"
#include "model/Entity.hpp"
#include "util/string_utils.hpp"
#include "util/math_utils.hpp"
#include "ScalingFactorService.hpp"

class DaoTestFixture_DQScalingFactorScalingUnscaling_Test;

namespace svr {

namespace dao { class DQScalingFactorDAO; }

namespace business {


class DQScalingFactorService : public ScalingFactorService {
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

    static bool match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, const bool overwrite);

    datamodel::dq_scaling_factor_container_t find_all_by_model_id(const bigint model_id);

    static datamodel::DQScalingFactor_ptr find(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const size_t model_id, const unsigned chunk, const unsigned gradient, const unsigned step, const unsigned level,
            const bool check_features, const bool check_labels);

    static datamodel::dq_scaling_factor_container_t slice(const datamodel::dq_scaling_factor_container_t &sf, const unsigned chunk, const unsigned gradient, const unsigned step);

    static datamodel::dq_scaling_factor_container_t calculate(const unsigned chunk_ix, const datamodel::OnlineMIMOSVR &svr_model, const arma::mat &features_t, const arma::mat &labels);

    static void scale_features(const unsigned chunk_ix, const unsigned grad_level, const unsigned step, const unsigned lag,
                               const datamodel::dq_scaling_factor_container_t &sf, arma::mat &features_t);
    static void scale_features(const unsigned chunk_ix, const datamodel::OnlineMIMOSVR &svr_model, arma::mat &features_t);
    static void scale_labels(const unsigned chunk, const unsigned gradient, const unsigned step, const unsigned level, const datamodel::dq_scaling_factor_container_t &sf,
                             arma::mat &labels);
    static double scale_label(const datamodel::DQScalingFactor &sf, double &label);
    static void scale_labels(const unsigned chunk_ix, const datamodel::OnlineMIMOSVR &svr_model, arma::mat &labels);
    static void scale_labels(const datamodel::DQScalingFactor &sf, arma::mat &labels);
    template<typename T> static inline void unscale_labels(const datamodel::DQScalingFactor &sf, T &labels)
    {
        common::unscale_I<T>(labels, sf.get_labels_factor(), sf.get_dc_offset_labels());
    }

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, const bool overwrite = false);

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor_ptr &p_new_sf, const bool overwrite = false);
};


}
}