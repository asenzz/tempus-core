#pragma once

// Bagged MIOC-SVR on chunks
// TODO finish gradient boosting, warm-start online solver, test manifold

#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>
#include <oneapi/tbb/concurrent_map.h>
#include <oneapi/tbb/concurrent_set.h>

#include "common/constants.hpp"
#include "common/defines.h"
#include "common/compatibility.hpp"
#include "model/SVRParameters.hpp"

namespace svr {

#define INTERLACE_MANIFOLD_FACTOR 10 // Every Nth row is used from a manifold dataset to train the produced model

constexpr unsigned IRWLS_ITER = 2e2;
constexpr unsigned IRWLS_ITER_ONLINE = 4;
constexpr unsigned IRWLS_ITER_TUNE = 1;

#define USE_MAGMA
constexpr double OUTLIER_ALPHA = 1e-6;

#ifdef FINER_GAMMA_TUNE
constexpr double DE_FINE_DIVISOR = 1e1; // > 1
constexpr double FINE_GAMMA_MULTIPLE = 1. + 1. / DE_FINE_DIVISOR;
constexpr double C_fine_gamma_div = 1e1;
constexpr double C_fine_gamma_mult = 1e1;
#endif


const auto C_gamma_multis = [](){
    std::deque<double> r = {/*1e-1, */1./*, 1e1*/};
    //for (double it = 1e-1; it < 1e1; it *= FINE_GAMMA_MULTIPLE) r.emplace_back(it);
    return r;
} ();


constexpr double BEST_PREDICT_CHUNKS_DIVISOR = 1; // above and 1
#define NO_ONLINE_LEARN

// #define PSEUDO_ONLINE // Really does batch train instead of online learn
// #define PSEUDO_PREDICT_LEVEL 14 // Pseudo-predict copies the last known label instead of predicting
constexpr unsigned C_logged_level = 999; // from 0 to level count, 999 for no logged level
// #define PSD_MANIFOLD
#define FAST_MANIFOLD_TRAIN
constexpr double TOL_ROWS = 1e-14;
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

typedef tbb::concurrent_map<size_t /* chunk */, tbb::concurrent_map<uint64_t /* epsco */, tbb::concurrent_set<t_param_preds_ptr, param_preds_cmp>>> t_grad_preds;
typedef tbb::concurrent_map<size_t, /* level */ t_grad_preds> predictions_t;


class OnlineMIMOSVR;

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

class OnlineMIMOSVR
{
    std::mutex learn_mx;

    class MimoBase;
    typedef std::shared_ptr<MimoBase> MimoBase_ptr;

    tbb::concurrent_set<size_t> tmp_ixs;
    size_t samples_trained = 0;
    matrix_ptr p_features = std::make_shared<arma::mat>();
    matrix_ptr p_labels = std::make_shared<arma::mat>();
    datamodel::t_param_set_ptr p_param_set = nullptr; // TODO Convert to a new class
    OnlineMIMOSVR_ptr p_manifold = nullptr;
    double labels_scaling_factor = 1;
    matrices_ptr p_kernel_matrices = nullptr;
    mimo_type_e model_type;
    std::deque<MimoBase_ptr> main_components;
    std::deque<arma::uvec> ixs;
    const size_t multistep_len;
    size_t chunk_size = CHUNK_DECREMENT;
    size_t gradient = std::numeric_limits<size_t>::max();
    size_t decon_level = std::numeric_limits<size_t>::max();

    void init_model_base(const double epsilon, const mimo_type_e model_type);

    bool save_kernel_matrix = false;

    class MimoBase
    {
    public:
        std::deque<arma::mat> chunk_weights;
        std::deque<double> chunks_weight;
        arma::mat total_weights;
        double epsilon;
        std::deque<double> mae_chunk_values;
    };
public:
    OnlineMIMOSVR(const datamodel::t_param_set_ptr &p_param_set_,
                  const size_t multistep_len = svr::common::__multistep_len,
                  const size_t chunk_size = CHUNK_DECREMENT);

    OnlineMIMOSVR(const datamodel::t_param_set_ptr &p_param_set_,
                  const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain,
                  const matrices_ptr &p_kernel_matrices = nullptr,
                  const size_t multistep_len = svr::common::__multistep_len,
                  const size_t chunk_size = CHUNK_DECREMENT,
                  const bool pseudo_online = false);

    OnlineMIMOSVR();

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
    datamodel::SVRParameters_ptr &get_params_ptr(const size_t chunk_ix = 0);
    datamodel::SVRParameters_ptr get_params_ptr(const size_t chunk_ix = 0) const;
    datamodel::SVRParameters &get_params(const size_t chunk_ix = 0);
    datamodel::SVRParameters get_params(const size_t chunk_ix = 0) const;

    void do_over_train_zero_epsilon(const arma::mat &xx_train, const arma::mat &yy_train, const arma::mat &eye_K, const size_t chunk_idx);

    datamodel::SVRParameters_ptr is_manifold() const;

    static void tune_kernel_params(tbb::concurrent_map<size_t, tbb::concurrent_map<uint64_t, tbb::concurrent_set<t_param_preds_ptr, param_preds_cmp>>> &predictions,
                                   datamodel::t_param_set &p_svr_parameters,
                                   const arma::mat &features,
                                   const arma::mat &labels,
                                   const arma::mat &last_knowns,
                                   const size_t chunk_size);

#if defined(TUNE_HYBRID)
    double produce_kernel_inverse_order(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    static std::deque<arma::uvec> get_indexes(const size_t n_rows, const datamodel::SVRParameters &svr_parameters, const size_t chunk_size);
    std::deque<arma::uvec> get_indexes(const size_t n_rows, const datamodel::SVRParameters &svr_parameters) const;

    arma::uvec get_other_ixs(const size_t i) const;

    std::deque<arma::uvec> get_indexes() const;

    void batch_train(const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain, const matrices_ptr &kernel_matrices = nullptr, const bool pseudo_online = false);

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time) const;

    arma::mat single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const;

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

    void reset_model(const bool pseudo_online = false);

    ~OnlineMIMOSVR() {};

    bool operator==(OnlineMIMOSVR const &) const;

    static OnlineMIMOSVR_ptr load_online_mimosvr(const char *filename);

    bool save_online_mimosvr(const char *filename);

    template<typename S> static bool save_online_svr(const OnlineMIMOSVR &osvr, S &output_stream);

    template<typename S> static OnlineMIMOSVR_ptr load_online_svr(S &input_stream);

    explicit OnlineMIMOSVR(std::stringstream &input_stream);

    friend void log_model(const svr::OnlineMIMOSVR &m, const std::deque<arma::mat> &chunk_kernels);

    template<typename S>
    static bool save_onlinemimosvr_no_weights_no_kernel(const OnlineMIMOSVR &osvr, S &output_stream);

    template<typename S>
    static OnlineMIMOSVR_ptr load_onlinemimosvr_no_weights_no_kernel(S &input_stream);

    static OnlineMIMOSVR_ptr load_onlinemimosvr_no_weights_no_kernel(const char *filename);

    bool save_onlinemimosvr_no_weights_no_kernel(const char *filename);

    static arma::mat
    init_predict_kernel_matrix(
            const datamodel::SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &x_predict,
            const bpt::ptime &predicted_time);

    arma::mat manifold_predict(const arma::mat &x_predict) const;

    static arma::mat init_kernel_matrix(const datamodel::SVRParameters &params, const arma::mat &x, const arma::mat &y);

    template<typename rT, typename kT, typename fT> static rT
    cached(std::map<kT, rT> &cache_cont, std::mutex &mx, const kT &cache_key, const fT &f);

    static std::deque<arma::mat> *prepare_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t);
    static arma::mat *prepare_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, size_t len = 0);
    static arma::mat *prepare_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);

    static double get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const size_t train_len, const double meanabs_labels);
    static std::deque<arma::mat> &get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);
    static arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const size_t short_size_X = 0);
    static arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &pred_time);

    static void clear_Zy_cache();

    size_t get_num_chunks() const;

    static size_t get_manifold_nrows(const size_t n_rows);

    void calc_weights(const size_t chunk_ix, const arma::mat &y_train, const arma::mat &epsco_eye_K, MimoBase &component, const size_t iters);
    void update_total_weights();
    arma::mat &get_weights(uint8_t type);
    arma::mat get_weights(uint8_t type) const;

    static std::tuple<double, double, std::deque<double>, std::deque<double>, double, std::deque<double>>
    future_validate(
            const size_t start_ix,
            svr::OnlineMIMOSVR &online_svr,
            const arma::mat &features,
            const arma::mat &labels,
            const arma::mat &last_knowns,
            const std::deque<bpt::ptime> &times,
            const bool single_pred = false,
            const double scale_label = 1,
            const double dc_offset = 0);

};

} // svr

using OnlineMIMOSVR_ptr = std::shared_ptr<svr::OnlineMIMOSVR>;

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OY_));               \
    (_MY_) = (_NY_) - (_OY_); }
