#pragma once


#include <memory>
#include <common/types.hpp>
#include "model/DataRow.hpp"
#include <model/DataRow.hpp>
#include "onlinesvr.hpp"
#include <model/DQScalingFactor.hpp>

//#define DEBUG_LABELS
namespace svr { namespace dao { class ModelDAO; }}
namespace svr { namespace datamodel { class Model; }}

using Model_ptr = std::shared_ptr<svr::datamodel::Model>;
using Model_vector = std::vector<Model_ptr>;

// #define REVERSE_FEATURES_ROW
// #define PRINTOUT_PER_LEVEL_VALUES
#define CACHED_FEATURE_ITER

namespace svr {
namespace business {

typedef std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>> cycle_returns_t;
typedef std::tuple<arma::rowvec, arma::rowvec, arma::rowvec> tune_returns_t;

//#define FAUX_FINAL_ONLINE

class ModelService
{
private:
    dao::ModelDAO &model_dao;
/*
    tbb::concurrent_map<std::pair<bpt::ptime, size_t>, arma::rowvec> cached_rows;
    tbb::concurrent_map<std::tuple<size_t, bpt::ptime, size_t>, double> cached_quantized_vals;
*/

    static double get_quantized_feature(const size_t pos, const data_row_container::const_iterator &end_iter,
                                        const size_t level, const double quantization_mul, const size_t lag);

    /* [start_iter, end_iter) */
    static bool
    prepare_features(
            const std::set<size_t> &adjacent_levels,
            const size_t lag,
            const data_row_container::const_iterator &end_iter,
            const bpt::time_duration &max_gap,
            const double main_to_aux_period_ratio,
#ifndef NEW_SCALING
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
#endif
            arma::rowvec &row);

    static bool
    prepare_labels(const size_t level_ix, const data_row_container::const_iterator &label_aux_start_iter,
                   const data_row_container::const_iterator &label_aux_end_iter, const boost::posix_time::ptime &start_time,
                   const boost::posix_time::ptime &end_time, arma::rowvec &labels_row, const bpt::time_duration &aux_resolution);

    static bool
    prepare_time_features(const bpt::ptime &value_time, arma::rowvec &row);

public:
    ModelService(
            dao::ModelDAO &model_dao,
            const bool update_r_matrix,
            const size_t max_smo_iterations,
            const double smo_epsilon_divisor,
            const size_t max_segment_length,
            const size_t multistep_len);

    Model_ptr get_model_by_id(const bigint model_id);

    Model_ptr get_model(const bigint ensemble_id, const size_t decon_level);

    int save(const Model_ptr &p_model);

    bool exists(const Model_ptr &p_model);

    int remove(const Model_ptr &p_model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    std::vector<Model_ptr> get_all_models_by_ensemble_id(const bigint ensemble_id);

    bool get_training_data(
            arma::mat &harvest_range,
            arma::mat &all_labels,
            arma::mat &all_last_knowns,
            std::vector<bpt::ptime> &all_times,
#ifndef NEW_SCALING
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const datamodel::dq_scaling_factor_container_t &aux_scaling_factors,
#endif
            const datamodel::datarow_range &main_data_range,
            const datamodel::datarow_range &aux_data_range,
            const size_t lag,
            const std::set<size_t> &adjacent_levels,
            const bpt::time_duration &max_gap,
            const size_t level,
            const double main_to_aux_period_ratio,
            const bpt::ptime &last_modeled_value_time = bpt::min_date_time,
            const bpt::time_duration &main_queue_resolution = bpt::hours(1));

    void get_features_row(
            const bpt::ptime &pred_time,
            const datamodel::datarow_range &aux_range,
            const datamodel::dq_scaling_factor_container_t &aux_scaling_factors,
            const std::set<size_t> &adjacent_levels,
            const size_t current_level,
            const bpt::time_duration &max_gap,
            const double main_to_aux_period_ratio,
            const size_t lag,
            const bpt::time_duration &main_queue_resolution,
            arma::rowvec &features_row);

    static arma::colvec
    predict(
            const Model_ptr &p_model,
            const data_row_container &p_main_decon_data,
            const std::vector<data_row_container_ptr> &aux_data_rows_containers,
            const boost::posix_time::ptime &prediction_time,
            const boost::posix_time::time_duration &resolution,
            const bpt::time_duration &max_gap);

    // Use only when adjacent level values are known for the predicted time range (eg. validation in paramtune)
    static data_row_container
    predict(const Model_ptr &p_model,
            const datamodel::datarow_range &p_main_decon_data,
            const std::vector<datamodel::datarow_range> &aux_data_rows_container,
            const ptimes_set_t &prediction_times,
            const bpt::time_duration &max_gap,
            const bpt::time_duration &resolution);

    static arma::mat
    predict(
            const std::vector<Model_ptr> &models,
            const boost::posix_time::ptime &prediction_time,
            const boost::posix_time::time_duration &resolution,
            const boost::posix_time::time_duration &max_gap,
            const datamodel::DataRow::container &main_decon_data,
            const std::vector<data_row_container_ptr> &aux_decon_data);

    // A bit more expensive but checks for lag count values before found time
    static void
    check_feature_data(
            const datamodel::DataRow::container &data,
            const datamodel::DataRow::container::const_iterator &iter,
            const bpt::time_duration &max_gap,
            const bpt::ptime &feat_time,
            const ssize_t lag_count);

    static void
    check_feature_data(
            const datamodel::DataRow::container &data,
            const datamodel::DataRow::container::const_iterator &iter,
            const bpt::time_duration &max_gap,
            const bpt::ptime &feat_time);

    static const datamodel::datarow_range
    prepare_feat_range(
            const datamodel::DataRow::container &data,
            const boost::posix_time::time_duration &max_gap,
            const boost::posix_time::ptime &predict_time,
            const ssize_t lag_count);

    static void
    train(datamodel::SVRParameters_ptr &svr_parameters, Model_ptr &p_model, const matrix_ptr &p_features,
          const matrix_ptr &p_labels, const bpt::ptime &new_last_modeled_value_time);

    static cycle_returns_t
    final_cycle(
            const datamodel::SVRParameters_ptr &p_best_parameters,
            const int validate_start_pos,
            const matrix_ptr &p_all_labels_mx,
            const matrix_ptr &p_all_features_mx);

    static cycle_returns_t
    train_predict_cycle_online(
            const int validate_start_pos,
            const matrix_ptr &p_all_labels_mx,
            const matrix_ptr &p_all_features_mx,
            OnlineMIMOSVR &svr_model);

    static void
    train_batch(datamodel::SVRParameters_ptr &p_svr_parameters, Model_ptr &p_model, const matrix_ptr &p_features_data, const matrix_ptr &p_labels_data);

    static void
    train_online(const arma::mat &features_data, const arma::mat &labels_data, OnlineMIMOSVR_ptr &p_svr_model);

#ifdef CACHED_FEATURE_ITER
    data_row_container::iterator aux_decon_hint;
#endif
};

} /* namespace business */
} /* namespace svr */


