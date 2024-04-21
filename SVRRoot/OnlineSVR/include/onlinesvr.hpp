#pragma once

// Bagged MIOC-SVR on chunks
// TODO test gradient boosting, find warm-start online solver, test manifold

#include <boost/date_time/posix_time/ptime.hpp>
#include <memory>
#include <set>
#include <armadillo>
#include <deque>
#include "common/compatibility.hpp"
#include "common/constants.hpp"
#include "common/defines.h"
#include "model/DQScalingFactor.hpp"
#include "model/SVRParameters.hpp"

namespace svr {
namespace business {
class calc_cache;
}

namespace datamodel {

constexpr unsigned C_interlace_manifold_factor = 20; // Every Nth row is used from a manifold dataset to train the produced model
constexpr double C_itersolve_delta = 1e-4;
constexpr double C_itersolve_range = 1e2;
constexpr double C_tune_range_min_lambda = 0;
constexpr double C_tune_range_max_lambda = 1e2;
constexpr double C_chunk_overlap = 0; // Chunk rows overlap ratio, higher generates more chunks
constexpr double C_chunk_offlap = 1. - C_chunk_overlap;
constexpr double C_chunk_tail = .1;
constexpr double C_chunk_header = 1. - C_chunk_tail;
constexpr double C_predict_chunks = .5; // Ratio of best chunks used for predictions

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

struct t_gradient_data
{
    mat_ptr p_features, p_labels;
    vec_ptr p_last_knowns;
    t_gradient_data(const mat_ptr &p_features, const mat_ptr &p_labels, const vec_ptr &p_last_knowns)
            : p_features(p_features), p_labels(p_labels), p_last_knowns(p_last_knowns) {}
};

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;

class OnlineMIMOSVR;

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

class OnlineMIMOSVR final : public Entity
{
    bigint model_id = 0;
    Dataset_ptr p_dataset;
    std::string column_name;
    std::set<size_t> tmp_ixs;
    size_t samples_trained = 0;
    mat_ptr p_features, p_labels;
    vec_ptr p_last_knowns;
    t_param_set param_set;
    OnlineMIMOSVR_ptr p_manifold;
    matrices_ptr p_kernel_matrices;
    datamodel::dq_scaling_factor_container_t scaling_factors;
    std::deque<arma::mat> weight_chunks, train_feature_chunks_t, train_label_chunks;
    std::deque<arma::uvec> ixs;
    std::deque<double> chunks_score;
    arma::mat all_weights;
    size_t multistep_len = svr::common::C_default_multistep_len;
    size_t max_chunk_size = common::C_default_kernel_max_chunk_size;
    size_t gradient = DEFAULT_SVRPARAM_GRAD_LEVEL;
    size_t decon_level = DEFAULT_SVRPARAM_DECON_LEVEL;

    virtual void init_id() override;

    virtual std::string to_string() const override;

    void parse_params();

public:
    OnlineMIMOSVR(const bigint id, const bigint model_id, const t_param_set &param_set, const Dataset_ptr &p_dataset = nullptr);

    OnlineMIMOSVR(
            const bigint id, const bigint model_id, const t_param_set &param_set,
            const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown,
            const bpt::ptime &last_value_time, const matrices_ptr &p_kernel_matrices = nullptr,
            const Dataset_ptr &p_dataset = nullptr);

    explicit OnlineMIMOSVR(const bigint id, const bigint model_id, std::stringstream &input_stream);

    OnlineMIMOSVR();

    ~OnlineMIMOSVR() = default;

    void reset();

    bool operator==(OnlineMIMOSVR const &) const;

    template<typename S> static void save(const OnlineMIMOSVR &osvr, S &output_stream);

    std::string save() const;

    template<typename S> static OnlineMIMOSVR_ptr load(S &input_stream);

    template<class A> void serialize(A &ar, const unsigned int version);

    business::calc_cache &ccache();

    bigint get_model_id() const;

    void set_model_id(const size_t new_model_id);

    Dataset_ptr get_dataset() const;

    Dataset_ptr &get_dataset();

    const dq_scaling_factor_container_t &get_scaling_factors() const;

    void set_scaling_factor(const DQScalingFactor_ptr &p_sf);

    void set_scaling_factors(const dq_scaling_factor_container_t &new_scaling_factors);

    void set_dataset(const Dataset_ptr &p_dataset);

    // Move to solver module
    static void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, arma::mat &w, const size_t iters, const bool psd);

    static void solve_batched_irwls(
            const std::vector<arma::mat> &K_epsco, const std::vector<arma::mat> &K, const std::vector<arma::mat> &rhs, std::vector<arma::mat> &solved, const size_t iters);

    static arma::mat do_ocl_solve(const double *host_a, double *host_b, const int m, const int nrhs);

    static void solve_dispatch(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, arma::mat &w, const size_t iters, const bool psd);

    static arma::mat solve_dispatch(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, const size_t iters, const bool psd);

    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    size_t get_gradient_level() const;

    size_t get_decon_level() const;

    ssize_t get_samples_trained_number() const;

    void clear_kernel_matrix();

    void init_manifold(const SVRParameters_ptr &p, const bpt::ptime &last_manifold_time);

    OnlineMIMOSVR_ptr get_manifold();

    t_param_set get_param_set() const;

    t_param_set &get_param_set();

    void set_param_set(const t_param_set &param_set_);

    void set_params(const SVRParameters_ptr &p_svr_parameters_, const size_t chunk_ix = 0);

    void set_params(const SVRParameters &svr_parameters_, const size_t chunk_ix = 0);

    SVRParameters_ptr get_params_ptr(const size_t chunk_ix = 0) const;

    SVRParameters_ptr is_manifold() const;

    bool is_gradient() const;

    static bool needs_tuning(const t_param_set &param_set);

    bool needs_tuning() const;

    void tune();

    void recombine_params(const size_t chunk_ix);

#if defined(TUNE_HYBRID)
    double produce_kernel_inverse_order(const SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    static double get_gamma_range_variance(const size_t train_len);

    static std::deque<double> get_gamma_base_multipliers(const double gamma_range_variance);

    static size_t get_full_train_len(const size_t n_rows, const size_t decrement);

    size_t get_num_chunks();

    static size_t get_num_chunks(const size_t n_rows, const size_t chunk_size_);

    static std::deque<arma::uvec> generate_indexes(const size_t n_rows, const size_t decrement, const size_t max_chunk_size);

    arma::uvec get_other_ixs(const size_t i) const;

    std::deque<arma::uvec> generate_indexes() const;

    std::deque<size_t> get_predict_chunks() const;

    arma::mat predict(const arma::mat &x_predict);

    arma::mat manifold_predict(const arma::mat &x_predict) const;

    t_gradient_data produce_residuals();

    void learn(const arma::mat &new_x_train, const arma::mat &new_y_train, const arma::vec &new_y_last_knowns, const bpt::ptime &last_value_time,
                 const bool temp_learn = false, const std::deque<size_t> &forget_ixs = {});

    void batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const bpt::ptime &last_value_time, const matrices_ptr &precalc_kernel_matrices = nullptr);

    arma::mat &get_features();

    arma::mat &get_labels();

    size_t get_multistep_len() const;

    arma::uvec get_active_ixs() const;

    arma::mat prepare_Ky(const SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &x_predict, const bpt::ptime &time);

    arma::mat prepare_Ky(const SVRParameters &svr_parameters, const arma::mat &x_train_t, const arma::mat &x_predict_t);

    arma::mat prepare_K(const SVRParameters &params, const arma::mat &x_t, const bpt::ptime &time);

    static arma::mat prepare_K(const SVRParameters &params, const arma::mat &x_t);

    static mat_ptr prepare_Z(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    static mat_ptr prepare_Z(const SVRParameters &params, const arma::mat &features_t);

    static mat_ptr prepare_Zy(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &time);

    static mat_ptr prepare_Zy(const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t);

    static std::shared_ptr<std::deque<arma::mat>> prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t);

    static double calc_gamma(const arma::mat &Z, const double mean_L);

    static double calc_epsco(const arma::mat &K);

    void calc_weights(const size_t chunk_ix, const size_t iters);

    void update_all_weights();
};

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

} // datamodel
} // svr

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OY_));               \
    (_MY_) = (_NY_) - (_OY_); }
