#pragma once

// Bagged MIOC-SVR on chunks
// TODO test gradient boosting, find warm-start online solver, test manifold

#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_map.h>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include "common/compatibility.hpp"
#include "common/constants.hpp"
#include "model/SVRParameters.hpp"

namespace svr {

constexpr unsigned C_interlace_manifold_factor = 1e9; // Every Nth row is used from a manifold dataset to train the produced model
constexpr double C_itersolve_delta = 1e-4;
constexpr double C_itersolve_range = 1e2;
constexpr double C_mean_weight_threshold = 2;
constexpr size_t C_grid_depth = 4; // Tune
constexpr double C_grid_range_div = 5;

#define USE_MAGMA
#define FORGET_MIN_WEIGHT

/*
 * Kernel matrix is set like this:
    auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(),svr_parameters);
    for (ssize_t i = 0; i < samples_trained_number; ++i) {
        for (ssize_t j = 0; j <= i; ++j) {
            const auto value = (*kernel)(X->get_row_ref(i), X->get_row_ref(j));
            p_kernel_matrix->set_value(i, j, value);
            p_kernel_matrix->set_value(j, i, value);
        }
    }
*/


struct t_param_preds {
    const double score = std::numeric_limits<double>::max();
    svr::datamodel::SVRParameters_ptr p_params;
    std::shared_ptr<std::deque<arma::mat>> p_predictions;
    std::shared_ptr<std::deque<arma::mat>> p_labels;
    std::shared_ptr<std::deque<arma::mat>> p_last_knowns;

    t_param_preds(const double score,
                  const datamodel::SVRParameters_ptr &params,
                  const std::shared_ptr<std::deque<arma::mat>> &predictions,
                  const std::shared_ptr<std::deque<arma::mat>> &labels,
                  const std::shared_ptr<std::deque<arma::mat>> &last_knowns) :
            score(score), p_params(params), p_predictions(predictions), p_labels(labels), p_last_knowns(last_knowns) {}
};
typedef std::shared_ptr<t_param_preds> t_param_preds_ptr;

struct param_preds_cmp
{
    bool operator()(const t_param_preds_ptr &lhs, const t_param_preds_ptr &rhs) const
    {
        return lhs->score < rhs->score;
    }
};

typedef std::deque /* chunk */ <std::set<t_param_preds_ptr, param_preds_cmp>> t_gradient_tuned_parameters;
typedef tbb::concurrent_map<size_t /* level */, std::deque /* grad */ <t_gradient_tuned_parameters>> t_tuned_parameters;

struct MimoBase
{
    std::deque<arma::mat> chunk_weights;
    std::deque<arma::mat> weights_mask;
    std::deque<double> chunks_weight;
    arma::mat total_weights;
    double epsilon;
    std::deque<double> mae_chunk_values;
    template<class A> void serialize(A &ar, const unsigned int version);
};
typedef std::shared_ptr<MimoBase> MimoBase_ptr;

class OnlineMIMOSVR;
using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

class  OnlineMIMOSVR
{
    std::mutex learn_mx;

    tbb::concurrent_set<size_t> tmp_ixs;
    size_t samples_trained;
    matrix_ptr p_features;
    matrix_ptr p_labels;
    datamodel::t_param_set_ptr p_param_set;
    OnlineMIMOSVR_ptr p_manifold;
    double labels_scaling_factor = 1;
    matrices_ptr p_kernel_matrices;
    mimo_type_e model_type;
    std::deque<MimoBase_ptr> main_components;
    std::deque<arma::uvec> ixs;
    size_t multistep_len = svr::common::__multistep_len;
    size_t max_chunk_size = C_kernel_default_max_chunk_size;
    size_t gradient = std::numeric_limits<size_t>::max();
    size_t decon_level = std::numeric_limits<size_t>::max();

    void init_model_base(const double epsilon, const mimo_type_e model_type);

public:
    OnlineMIMOSVR(const datamodel::t_param_set_ptr &p_param_set_,
                  const size_t multistep_len = svr::common::__multistep_len,
                  const size_t chunk_size = C_kernel_default_max_chunk_size);

    OnlineMIMOSVR(const datamodel::t_param_set_ptr &p_param_set_,
                  const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain,
                  const matrices_ptr &p_kernel_matrices = nullptr,
                  const size_t multistep_len = svr::common::__multistep_len,
                  const size_t chunk_size = C_kernel_default_max_chunk_size);

    explicit OnlineMIMOSVR(std::stringstream &input_stream);

    OnlineMIMOSVR();
    ~OnlineMIMOSVR() = default;

    void reset_model();

    bool operator==(OnlineMIMOSVR const &) const;

    template<typename S> static bool save(const OnlineMIMOSVR &osvr, S &output_stream);
    template<typename S> static OnlineMIMOSVR_ptr load(S &input_stream);
    template<class A> void serialize(A &ar, const unsigned int version);


    // Move to solver module
    static void solve_irwls(const arma::mat &epsilon_eye_K, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters, const bool psd);
    static arma::mat do_ocl_solve(const double *host_a, double *host_b, const int m, const int nrhs);
    static void solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, arma::mat &solved, const size_t iters, const bool psd);
    static arma::mat solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, const size_t iters, const bool psd);
    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    ssize_t get_samples_trained_number() const
    {
        return samples_trained;
    }

    void clear_kernel_matrix()
    {
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_kernel_matrices->size()))
        for (auto &k: *p_kernel_matrices) k.clear();
    }

    void init_manifold(const datamodel::SVRParameters_ptr &p);

    datamodel::t_param_set_ptr get_param_set_ptr() const;
    datamodel::t_param_set_ptr &get_param_set_ptr();
    svr::datamodel::t_param_set get_param_set() const;
    svr::datamodel::t_param_set &get_param_set();
    void set_param_set(const datamodel::t_param_set_ptr &p_svr_param_set);
    void set_param_set(const datamodel::t_param_set &svr_param_set_);
    void set_params(const datamodel::SVRParameters_ptr &p_svr_parameters_, const size_t chunk_ix = 0);
    void set_params(const datamodel::SVRParameters &svr_parameters_, const size_t chunk_ix = 0);
    datamodel::SVRParameters_ptr get_params_ptr(const size_t chunk_ix = 0) const;
    datamodel::SVRParameters &get_params(const size_t chunk_ix = 0);
    datamodel::SVRParameters get_params(const size_t chunk_ix = 0) const;

    void do_over_train_zero_epsilon(const arma::mat &xx_train, const arma::mat &yy_train, const arma::mat &eye_K, const size_t chunk_idx);

    bool is_manifold(datamodel::SVRParameters_ptr &p_out) const;
    bool is_manifold() const;
    bool is_gradient() const;
    bool needs_tuning() const;

    static void tune(t_gradient_tuned_parameters &predictions,
                     const datamodel::t_param_set &template_parameters,
                     const arma::mat &features,
                     const arma::mat &labels,
                     const arma::mat &last_knowns,
                     const size_t chunk_size);

#if defined(TUNE_HYBRID)
    double produce_kernel_inverse_order(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    size_t get_num_chunks() const;
    static size_t get_num_chunks(const size_t n_rows, const size_t chunk_size_);
    static std::deque<arma::uvec> get_indexes(const size_t n_rows, const datamodel::SVRParameters &svr_parameters, const size_t max_chunk_size);
    std::deque<arma::uvec> get_indexes(const size_t n_rows, const datamodel::SVRParameters &svr_parameters) const;
    arma::uvec get_other_ixs(const size_t i) const;
    std::deque<arma::uvec> get_indexes() const;

    void batch_train(const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain, const matrices_ptr &kernel_matrices = nullptr);

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time, const bool masked = false) const;

    arma::mat single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const;

    std::pair<matrix_ptr, matrix_ptr> produce_residuals() const;

    size_t learn(const arma::mat &new_x_train, const arma::mat &new_y_train, const bool temp_learn = false, const std::deque<size_t> forget_ixs = {}, const bpt::ptime &label_time = bpt::special_values::not_a_date_time);

    arma::mat &get_features();

    arma::mat &get_labels();

    size_t get_multistep_len() const
    {
        return multistep_len;
    }

    mimo_type_e get_model_type() const
    {
        return model_type;
    }

    arma::uvec get_active_ixs()
    {
        arma::uvec active_ixs;
        for (const auto &ix: ixs) active_ixs.insert_rows(active_ixs.n_rows, ix);
        active_ixs = arma::unique(active_ixs);
        return active_ixs;
    }

    static arma::mat
    init_predict_kernel_matrix(
            const datamodel::SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &x_predict,
            const bpt::ptime &predicted_time);

    arma::mat manifold_predict(const arma::mat &x_predict) const;

    static arma::mat init_kernel_matrix(const datamodel::SVRParameters &params, const arma::mat &x);

    static std::deque<arma::mat> *prepare_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t);
    static arma::mat *prepare_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, size_t len = 0);
    static arma::mat *prepare_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);

    static double get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const size_t train_len, const double meanabs_labels);
    static std::deque<arma::mat> &get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);
    static arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const size_t short_size_X = 0);
    static arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &pred_time);

    static void clear_gradient_caches(const datamodel::SVRParameters &p);

    static double calc_gamma(const arma::mat &Z, const double train_len, const double mean_L);
    static double calc_epsco(const arma::mat &K, const size_t train_len);
    void calc_weights(const size_t chunk_ix, const arma::mat &y_train, const arma::mat &epsco_eye_K, MimoBase &component, const size_t iters);
    void update_total_weights();
    arma::mat &get_weights(uint8_t type);
    arma::mat get_weights(uint8_t type) const;
};

} // svr

using OnlineMIMOSVR_ptr = std::shared_ptr<svr::OnlineMIMOSVR>;

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OY_));               \
    (_MY_) = (_NY_) - (_OY_); }
