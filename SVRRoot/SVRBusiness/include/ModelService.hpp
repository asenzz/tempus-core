#pragma once


#include <memory>
#include <limits>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_map.h>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/DataRow.hpp"
#include "model/Ensemble.hpp"
#include "model/DQScalingFactor.hpp"
#include "model/Model.hpp"
#include "model/DeconQueue.hpp"
#include "onlinesvr.hpp"
#include "online_emd.hpp"
#include "model/SVRParameters.hpp"
#include "model/User.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"

// #define PRINTOUT_PER_LEVEL_VALUES


namespace svr {
namespace dao { class ModelDAO; }
namespace datamodel {
class Dataset;
using Dataset_ptr = std::shared_ptr<Dataset>;
class Model;
using Model_ptr = std::shared_ptr<Model>;
class Ensemble;
using Ensemble_ptr = std::shared_ptr<Ensemble>;
}
namespace business {

typedef std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>> cycle_returns_t;


class ModelService
{
    dao::ModelDAO &model_dao;

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
            arma::rowvec &row);

    static bool
    prepare_labels(const size_t level_ix, const data_row_container::const_iterator &label_aux_start_iter,
                   const data_row_container::const_iterator &label_aux_end_iter, const boost::posix_time::ptime &start_time,
                   const boost::posix_time::ptime &end_time, arma::rowvec &labels_row, const bpt::time_duration &aux_resolution);

    static bool
    prepare_time_features(const bpt::ptime &value_time, arma::rowvec &row);

    static arma::vec
    get_last_knowns(const datamodel::Ensemble_ptr &p_ensemble, const size_t level, const std::set<bpt::ptime> &times, const bpt::time_duration &resolution);

public:
    ModelService(dao::ModelDAO &model_dao);

    datamodel::Model_ptr get_model_by_id(const bigint model_id);

    datamodel::Model_ptr get_model(const bigint ensemble_id, const size_t decon_level);

    void configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr &p_ensemble, datamodel::Model_ptr &p_model);

    int save(const datamodel::Model_ptr &p_model);

    bool exists(const datamodel::Model_ptr &p_model);

    int remove(const datamodel::Model_ptr &p_model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    std::deque<datamodel::Model_ptr> get_all_models_by_ensemble_id(const bigint ensemble_id);

    static bool
    get_training_data(
            arma::mat &all_features,
            arma::mat &all_labels,
            arma::mat &all_last_knowns,
            std::deque<bpt::ptime> &all_times,
            const datamodel::datarow_range &label_main,
            const datamodel::datarow_range &label_aux,
            const std::deque<datamodel::datarow_range> &feats_aux,
            const size_t lag,
            const std::set<size_t> &adjacent_levels,
            const bpt::time_duration &max_gap,
            const size_t level,
            const bpt::time_duration &aux_queue_resolution,
            const bpt::ptime &last_modeled_value_time = bpt::min_date_time,
            const bpt::time_duration &main_queue_resolution = bpt::hours(1),
            const size_t multiout = 1);

    static void get_features_row(
            const datamodel::Dataset_ptr &p_dataset,
            const datamodel::Ensemble_ptr &p_ensemble,
            const datamodel::Model_ptr &p_model,
            const bpt::ptime &pred_time,
            arma::rowvec &features_row);

    static data_row_container predict(
            const datamodel::Ensemble_ptr &p_ensemble,
            const datamodel::Model_ptr &p_model,
            const datamodel::dq_scaling_factor_container_t &aux_dq_scaling_factors,
            const std::pair<std::set<bpt::ptime>, arma::mat> &predict_features,
            const bpt::time_duration &resolution);

    // A bit more expensive but checks for lag count values before found time
    static void
    check_feature_data(
            const datamodel::DataRow::container &data,
            const datamodel::DataRow::container::const_iterator &iter,
            const bpt::time_duration &max_gap,
            const bpt::ptime &feat_time,
            const ssize_t lag_count);

    static void check_feature_data(
            const datamodel::DataRow::container &data,
            const datamodel::DataRow::container::const_iterator &iter,
            const bpt::time_duration &max_gap,
            const bpt::ptime &feat_time);

    static const datamodel::datarow_range prepare_feat_range(
            const datamodel::DataRow::container &data,
            const boost::posix_time::time_duration &max_gap,
            const boost::posix_time::ptime &predict_time,
            const ssize_t lag_count);

    static void
    train(datamodel::Model_ptr p_model, const matrix_ptr &p_features, const matrix_ptr &p_labels, const bpt::ptime &new_last_modeled_value_time = bpt::not_a_date_time);

    static bool needs_tuning(const datamodel::t_param_set_ptr &p_param_set);

    static void tune(
            datamodel::Model_ptr p_model,
            std::deque<t_gradient_tuned_parameters> &tune_parameters,
            const matrix_ptr &p_features,
            const matrix_ptr &p_labels,
            const matrix_ptr &p_last_knowns);

    static void
    train_batch(datamodel::t_param_set_ptr &p_param_set, datamodel::Model_ptr &p_model, const matrix_ptr &p_features, const matrix_ptr &p_labels);

    static void train_online(const arma::mat &features_data, const arma::mat &labels_data, std::deque<OnlineMIMOSVR_ptr> &svr_models, const bpt::ptime &last_modeled_time);

    static datamodel::Model_ptr find(const std::deque<datamodel::Model_ptr> &models, const size_t levix);

    static void init_default_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble);

    static size_t to_level_ix(const size_t model_ix);

    static std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
    future_validate(
            const size_t start_ix,
            OnlineMIMOSVR &online_svr,
            const arma::mat &features,
            const arma::mat &labels,
            const arma::mat &last_knowns,
            const std::deque<bpt::ptime> &times,
            const datamodel::dq_scaling_factor_container_t &scaling_factors,
            const std::string &column,
            const bool online_learn = false);
};

} /* namespace business */
} /* namespace svr */


