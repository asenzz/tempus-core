#pragma once

#include <armadillo>
#include <deque>
#include <memory>
#include <set>
#include <tuple>
#include "model_features.hpp"
#include "common/compatibility.hpp"
#include "common/types.hpp"
#include "model/DataRow.hpp"

// #define PRINTOUT_PER_LEVEL_VALUES

namespace svr {
namespace dao {
class ModelDAO;
}

namespace datamodel {
struct t_level_predict_features;

class SVRParameters;
using SVRParameters_ptr = std::shared_ptr<SVRParameters>;

class t_feature_mechanics;

class Dataset;
using Dataset_ptr = std::shared_ptr<Dataset>;

class Model;
using Model_ptr = std::shared_ptr<Model>;

class Ensemble;
using Ensemble_ptr = std::shared_ptr<Ensemble>;

class DeconQueue;
using DeconQueue_ptr = std::shared_ptr<DeconQueue>;

class InputQueue;
using InputQueue_ptr = std::shared_ptr<InputQueue>;

class OnlineSVR;
using OnlineSVR_ptr = std::shared_ptr<OnlineSVR>;
}

namespace business {
class ModelService
{
    dao::ModelDAO &model_dao;

    static arma::rowvec prepare_special_features(const datamodel::data_row_container::const_iterator &last_known_it, const bpt::time_duration &resolution, uint32_t len);

    static arma::vec get_last_knowns(const datamodel::Ensemble &ensemble, uint16_t level, const datamodel::data_row_container &times, const bpt::time_duration &resolution);

public:
    static uint32_t get_max_quantisation();

    static const std::deque<uint32_t> &get_quantisations();

    static uint32_t get_max_row_len();

    explicit ModelService(dao::ModelDAO &model_dao);

    datamodel::Model_ptr get_model_by_id(const bigint model_id) const;

    datamodel::Model_ptr get_model(const bigint ensemble_id, uint16_t decon_level) const;

    void configure(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model) const;

    int save(const datamodel::Model_ptr &p_model) const;

    bool exists(const datamodel::Model &model) const;

    int remove(const datamodel::Model_ptr &p_model) const;

    int remove_by_ensemble_id(const bigint ensemble_id) const;

    std::deque<datamodel::Model_ptr> get_all_models_by_ensemble_id(const bigint ensemble_id) const;

    static void prepare_labels(arma::mat &all_labels, arma::vec &all_last_knowns, datamodel::data_row_container &all_times, const datamodel::datarow_crange &main_data,
                               const datamodel::datarow_crange &aux_data, const bpt::time_duration &max_gap, uint16_t level, const bpt::time_duration &resolution_aux,
                               const bpt::ptime &last_modeled_value_time, const bpt::time_duration &resolution_main, uint16_t steps, uint32_t lag);

    static void tune_features(
        arma::mat &out_features, const arma::mat &labels, datamodel::SVRParameters &params, const datamodel::data_row_container &label_times,
        const std::deque<datamodel::DeconQueue_ptr> &feat_queues,
        const bpt::time_duration &aux_queue_res, const bpt::time_duration &main_queue_resolution);

    static void do_features(
        arma::mat &out_features, uint32_t n_rows, uint32_t lag, uint32_t coef_lag, uint32_t coef_lag_,
        const datamodel::t_feature_mechanics &fm, const boost::posix_time::time_duration &stripe_period, const std::deque<uint32_t> &chunk_len_quantise,
        const std::deque<uint32_t> &in_rows, const std::deque<arma::mat> &decon, const std::deque<std::vector<t_feat_params> > &feat_params, const std::set<uint16_t> &adjacent_levels);

    static void prepare_features(
        arma::mat &features, const datamodel::data_row_container &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux, const datamodel::SVRParameters &param,
        const bpt::time_duration &resolution_aux, const bpt::time_duration &main_queue_resolution);

    static void prepare_weights(arma::mat &weights, const datamodel::data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_decon_queues, uint16_t steps,
                                const bpt::time_duration &resolution_main);

    static std::tuple<mat_ptr, mat_ptr, vec_ptr, mat_ptr, datamodel::data_row_container_ptr>
    get_training_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model, uint32_t dataset_rows = 0);

    static void predict(const datamodel::Ensemble &ensemble, datamodel::Model &model, const datamodel::t_level_predict_features &predict_features, const bpt::time_duration &resolution,
                        tbb::mutex &insmx, datamodel::data_row_container &output_data);

#ifdef INTEGRATION_TEST
    static void predict(
        const datamodel::Ensemble &ensemble, datamodel::Model &model, const datamodel::t_level_predict_features &predict_features, const bpt::time_duration &resolution, tbb::mutex &insmx,
        const arma::mat &labels, datamodel::data_row_container &output_data);

    static std::tuple<double, double, arma::vec, arma::vec, arma::vec, double, arma::vec> validate(
        uint32_t start_ix, const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model, const arma::mat &features, const arma::mat &labels,
        const arma::vec &last_knowns, const arma::mat &weights, const datamodel::data_row_container &times, bool online, bool verbose);
#endif

    // A bit more expensive but checks for lag count values before found time
    static void check_feature_data(
        const datamodel::DataRow::container &data,
        const datamodel::DataRow::container::const_iterator &iter,
        const bpt::time_duration &max_gap,
        const bpt::ptime &feat_time,
        ssize_t lag_count);

    static void check_feature_data(
        const datamodel::DataRow::container &data,
        const datamodel::DataRow::container::const_iterator &iter,
        const bpt::time_duration &max_gap,
        const bpt::ptime &feat_time);

    static void train(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, datamodel::Model &model);

    static void train_batch(datamodel::Model &model, const mat_ptr &p_features, const mat_ptr &p_labels, const mat_ptr &p_weights, const bpt::ptime &last_value_time);

    static void train_online(datamodel::Model &model, const arma::mat &features, const arma::mat &labels, const arma::mat &weights, const bpt::ptime &last_value_time);

    static datamodel::Model_ptr find(const std::deque<datamodel::Model_ptr> &models, uint16_t levix, uint16_t stepix);

    static datamodel::SVRParameters_ptr produce_parameters(const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::Model &model,
                                                           const std::deque<datamodel::SVRParameters_ptr> &paramset, uint16_t chunk_ix, uint16_t grad_ix);

    void init_models(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble &ensemble) const;

    static bool check(const std::deque<datamodel::Model_ptr> &models, uint16_t model_ct);

    static bool check(const std::deque<datamodel::OnlineSVR_ptr> &models, uint16_t grad_ct);

    static uint16_t to_level_ct(uint16_t model_ct) noexcept;

    static uint16_t to_model_ct(uint16_t level_ct) noexcept;
};
} /* namespace business */
} /* namespace svr */
