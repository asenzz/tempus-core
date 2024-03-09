#pragma once


#include <memory>
#include <limits>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>

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
#include "model/InputQueue.hpp"
#include "DatasetService.hpp"

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

    /* [start_iter, end_iter] */
    static double get_quantized_feature(
            const size_t pos,
            const data_row_container::const_iterator &end_iter,
            const size_t level,
            const double quantization_mul,
            const size_t lag);

    /* [start_iter, end_iter) */
    static bool prepare_labels(
            const size_t level_ix,
            const data_row_container::const_iterator &label_aux_start_iter,
            const data_row_container::const_iterator &label_aux_end_iter,
            const boost::posix_time::ptime &start_time,
            const boost::posix_time::ptime &end_time,
            arma::rowvec &labels_row,
            const bpt::time_duration &aux_resolution);

    static bool
    prepare_time_features(const bpt::ptime &value_time, arma::rowvec &row);

    static arma::vec
    get_last_knowns(const datamodel::Ensemble_ptr &p_ensemble, const size_t level, const std::set<bpt::ptime> &times, const bpt::time_duration &resolution);

    static datamodel::DataRow::container::const_iterator
    get_start(const datamodel::DataRow::container &cont, const size_t decremental_offset, const boost::posix_time::ptime &model_last_time,
              const boost::posix_time::time_duration &resolution);
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

    static void
    get_label_data(
            arma::mat &all_labels,
            arma::vec &all_last_knowns,
            std::deque<bpt::ptime> &all_times,
            const datamodel::datarow_crange &main_data,
            const datamodel::datarow_crange &labels_aux,
            const bpt::time_duration &max_gap,
            const size_t level,
            const bpt::time_duration &aux_queue_res,
            const bpt::ptime &last_modeled_value_time,
            const bpt::time_duration &main_queue_resolution,
            const size_t multiout);

    static void
    get_features_data(
            arma::mat &all_features,
            const std::deque<bpt::ptime> &label_times,
            const std::deque<datamodel::DeconQueue_ptr> &features_aux,
            const size_t lag,
            const std::set<size_t> &adjacent_levels,
            const bpt::time_duration &max_gap,
            const bpt::time_duration &aux_queue_res,
            const bpt::time_duration &main_queue_resolution);

    static std::tuple<matrix_ptr, matrix_ptr, vec_ptr>
    get_training_data(
            std::deque<bpt::ptime> &all_times,
            datamodel::Dataset &dataset,
            const datamodel::Ensemble &ensemble,
            const datamodel::Model &model,
            size_t dataset_rows = 0);

    static void get_features_row(
            const datamodel::Dataset_ptr &p_dataset,
            const datamodel::Ensemble_ptr &p_ensemble,
            const datamodel::Model_ptr &p_model,
            const bpt::ptime &pred_time,
            const std::set<size_t> &feat_levels,
            arma::rowvec &features_row);

    static void predict(
            const datamodel::Ensemble_ptr &p_ensemble,
            datamodel::Model_ptr &p_model,
            const datamodel::dq_scaling_factor_container_t &aux_dq_scaling_factors,
            const std::pair<std::set<bpt::ptime>, arma::mat> &predict_features,
            const bpt::time_duration &resolution,
            data_row_container &output_data);

    // A bit more expensive but checks for lag count values before found time
    static void check_feature_data(
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

    static void train(datamodel::Model &model, const matrix_ptr &p_features, const matrix_ptr &p_labels, const vec_ptr &p_lastknowns,
                      const bpt::ptime &new_last_modeled_value_time);

    static void train_batch(datamodel::Model &model, const matrix_ptr &p_features, const matrix_ptr &p_labels, const vec_ptr &p_lastknowns, const bpt::ptime &last_value_time);

    static void train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::vec &last_knowns, const bpt::ptime &last_value_time);

    static bool needs_tuning(const datamodel::t_param_set_ptr &p_param_set);

    static datamodel::Model_ptr find(const std::deque<datamodel::Model_ptr> &models, const size_t levix);

    void init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble);

    static bool check(const std::deque<datamodel::Model_ptr> &models, const size_t model_ct);

    static bool check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const size_t grad_ct);

    static size_t to_level_ix(const size_t model_ix, const size_t level_ct);

    static size_t to_model_ix(const size_t level_ix, const size_t level_ct);

    static size_t to_level_ct(const size_t model_ct);

    static size_t to_model_ct(const size_t level_ct);

    static std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
    future_validate(const size_t start_ix, datamodel::OnlineMIMOSVR &online_svr, const arma::mat &features, const arma::mat &labels,
                    const arma::mat &last_knowns, const std::deque<bpt::ptime> &times, const datamodel::dq_scaling_factor_container_t &scaling_factors,
                    const bool online_learn, const bool verbose);

};

} /* namespace business */
} /* namespace svr */


