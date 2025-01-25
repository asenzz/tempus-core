#pragma once


#include <memory>
#include <limits>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>

#include <oneapi/tbb/concurrent_set.h>
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
#include "model_features.hpp"

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

class ModelService {
    dao::ModelDAO &model_dao;

    static arma::rowvec prepare_special_features(const data_row_container::const_iterator &last_known_it, const bpt::time_duration &resolution, const uint32_t len);

    static arma::vec get_last_knowns(const datamodel::Ensemble &ensemble, const uint16_t level, const data_row_container &times, const bpt::time_duration &resolution);

    static datamodel::DataRow::container::const_iterator
    get_start(const datamodel::DataRow::container &cont, const uint32_t decremental_offset, const boost::posix_time::ptime &model_last_time,
              const boost::posix_time::time_duration &resolution);

public:
    static const uint32_t C_max_quantisation;

    static constexpr uint16_t C_num_quantisations = 1; // 100;

    static const std::deque<uint32_t> C_quantisations;

    static uint32_t get_max_row_len();

    explicit ModelService(dao::ModelDAO &model_dao);

    datamodel::Model_ptr get_model_by_id(const bigint model_id);

    datamodel::Model_ptr get_model(const bigint ensemble_id, const uint16_t decon_level);

    void configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model);

    int save(const datamodel::Model_ptr &p_model);

    bool exists(const datamodel::Model &model);

    int remove(const datamodel::Model_ptr &p_model);

    int remove_by_ensemble_id(const bigint ensemble_id);

    std::deque<datamodel::Model_ptr> get_all_models_by_ensemble_id(const bigint ensemble_id);

    static void prepare_labels(arma::mat &all_labels, arma::vec &all_last_knowns, data_row_container &all_times, const datamodel::datarow_crange &main_data,
                               const datamodel::datarow_crange &aux_data, const bpt::time_duration &max_gap, const uint16_t level, const bpt::time_duration &resolution_aux,
                               const bpt::ptime &last_modeled_value_time, const bpt::time_duration &resolution_main, const uint16_t multistep, const uint32_t lag);

    static void tune_features(arma::mat &out_features, const arma::mat &labels, datamodel::SVRParameters &params, const data_row_container &label_times,
                              const std::deque<datamodel::DeconQueue_ptr> &feat_queues, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res,
                              const bpt::time_duration &main_queue_resolution);

    static void prepare_features(arma::mat &features, const data_row_container &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux,
                                 const datamodel::SVRParameters &params, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res,
                                 const bpt::time_duration &main_queue_resolution);

    static void prepare_weights(arma::mat &weights, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_decon_queues, const uint16_t steps,
                                const bpt::time_duration &resolution_main);

    static std::tuple<mat_ptr, mat_ptr, vec_ptr, mat_ptr, data_row_container_ptr>
    get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, uint32_t dataset_rows = 0);

    static void predict(
            const datamodel::Ensemble &ensemble,
            datamodel::Model &model,
            const datamodel::t_level_predict_features &predict_features,
            const bpt::time_duration &resolution,
            tbb::mutex &insmx,
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

    static void train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model);

    static void train_batch(datamodel::Model &model, const mat_ptr &p_features, const mat_ptr &p_labels, const vec_ptr &p_lastknowns, const mat_ptr &p_weights,
                            const bpt::ptime &last_value_time);

    static void train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::vec &last_knowns, const arma::mat &weights,
                             const bpt::ptime &last_value_time);

    static datamodel::Model_ptr find(const std::deque<datamodel::Model_ptr> &models, const uint16_t levix, const uint16_t stepix);

    void init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble &ensemble);

    static bool check(const std::deque<datamodel::Model_ptr> &models, const uint16_t model_ct);

    static bool check(const std::deque<datamodel::OnlineMIMOSVR_ptr> &models, const uint16_t grad_ct);

    static uint16_t to_level_ix(const uint16_t model_ix, const uint16_t level_ct) noexcept;

    static uint16_t to_model_ix(const uint16_t level_ix, const uint16_t level_ct);

    static uint16_t to_level_ct(const uint16_t model_ct) noexcept;

    static uint16_t to_model_ct(const uint16_t level_ct) noexcept;

    static std::tuple<double, double, arma::vec, arma::vec, double, arma::vec>
    validate(const uint32_t start_ix, const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model, const arma::mat &features,
             const arma::mat &labels, const arma::vec &last_knowns, const arma::mat &weights, const data_row_container &times, const bool online, const bool verbose);

    static void
    do_features(
            arma::mat &out_features, const uint32_t n_rows, const uint32_t lag, const uint32_t coef_lag, const uint32_t coef_lag_, const uint16_t levels, const uint16_t n_queues,
            const datamodel::t_feature_mechanics &fm, const boost::posix_time::time_duration &stripe_period, const std::deque<uint32_t> &chunk_len_quantise,
            const std::deque<uint32_t> &in_rows, const std::deque<arma::mat> &decon, const std::deque<arma::u32_vec> &times_F,
            const std::deque<std::vector<t_feat_params>> &feat_params);
};

} /* namespace business */
} /* namespace svr */


