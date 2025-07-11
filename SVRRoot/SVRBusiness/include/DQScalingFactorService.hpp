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

    bool exists(const datamodel::DQScalingFactor_ptr &dq_scaling_factor) const;
    // bool exists_by_dataset_id(bigint dataset_id); // TODO Implement as needed!

    int save(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor) const;

    int remove(const datamodel::DQScalingFactor_ptr &p_dq_scaling_factor) const;

    static std::pair<bool, bool> match_n_set(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor &nf, bool overwrite);

    static void reset(datamodel::dq_scaling_factor_container_t &sf, const uint16_t decon, const uint16_t chunk, const uint16_t step, const uint16_t grad);

    datamodel::dq_scaling_factor_container_t find_all_by_model_id(bigint model_id) const;

    static datamodel::DQScalingFactor_ptr find(
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            bigint model_id, uint16_t chunk, uint16_t gradient, uint16_t step, uint16_t level,
            bool check_features, bool check_labels);

    static datamodel::dq_scaling_factor_container_t slice(const datamodel::dq_scaling_factor_container_t &sf, uint16_t chunk, uint16_t gradient, uint16_t step);

    template<typename T> static datamodel::dq_scaling_factor_container_t calculate(bigint model_id, const datamodel::SVRParameters &param, const arma::Mat<T> &features_t, const arma::Mat<T> &labels);
    template<typename T> static void scale_features_I(uint16_t chunk_ix, uint16_t grad_level, uint16_t step, uint16_t lag, const datamodel::dq_scaling_factor_container_t &sf, arma::Mat<T> &features_t);
    static void scale_features_I(uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &features_t);
    static void scale_labels_I(uint16_t chunk, uint16_t gradient, uint16_t step, uint16_t level, const datamodel::dq_scaling_factor_container_t &sf, arma::mat &labels);
    static double scale_label(const datamodel::DQScalingFactor &sf, double &label);
    static void scale_labels_I(uint16_t chunk_ix, const datamodel::OnlineSVR &svr_model, arma::mat &labels);
    static arma::mat scale_labels(const datamodel::DQScalingFactor &sf, const arma::mat &labels);
    template<typename T> static void scale_labels_I(const datamodel::DQScalingFactor &sf, arma::Mat<T> &labels);
    template<typename T> static inline void unscale_labels_I(const datamodel::DQScalingFactor &sf, T &labels)
    {
        (void) common::unscale_I(labels, sf.get_labels_factor(), sf.get_dc_offset_labels());
    }

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::dq_scaling_factor_container_t &new_sf, bool overwrite = false);

    static void add(datamodel::dq_scaling_factor_container_t &sf, const datamodel::DQScalingFactor_ptr &p_new_sf, bool overwrite = false);
};


}
}

#include "DQScalingFactorService.tpp"