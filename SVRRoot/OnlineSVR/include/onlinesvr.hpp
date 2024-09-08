#pragma once

// Bagged MIOC-SVR on chunks
// TODO test gradient boosting, find warm-start online solver, test manifold

#include <boost/date_time/posix_time/ptime.hpp>
#include <memory>
#include <set>
#include <deque>
#include <armadillo>
#include <magma_types.h>
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
constexpr double C_tune_range_min_lambda = 0;
constexpr double C_tune_range_max_lambda = 1e2;
constexpr double C_chunk_overlap = 1. - 1. / 4.; // Chunk rows overlap ratio [0..1], higher generates more chunks
constexpr double C_chunk_offlap = 1. - C_chunk_overlap;
constexpr double C_chunk_tail = .1;
constexpr double C_chunk_header = 1. - C_chunk_tail;
constexpr unsigned C_predict_chunks = 1; // TODO Review. Best chunks used for predictions
constexpr unsigned C_end_chunks = 1; // [1..1/offlap]
constexpr double C_gamma_variance = 6e4;
constexpr magma_int_t C_rbt_iter = 40; // default 30
constexpr double C_rbt_threshold = 0; // [0..1] default 1
constexpr double C_features_superset_coef = 1e2; // [1..+inf)
#ifdef EMO_DIFF
constexpr double C_diff_coef = 1;
#else
constexpr double C_diff_coef = 1;
#endif

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

struct mmm_t { double mean = 0, max = 0, min = 0; };

class OnlineMIMOSVR final : public Entity
{
    bigint model_id = 0;
    bpt::ptime last_trained_time;
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
    std::deque<std::pair<double, double>> chunks_score;
    arma::mat all_weights;
    unsigned multiout = common::C_default_multiout;
    unsigned max_chunk_size = common::C_default_kernel_max_chunk_len;
    unsigned gradient = C_default_svrparam_grad_level;
    unsigned level = C_default_svrparam_decon_level;
    unsigned step = C_default_svrparam_step;
    unsigned projection = 0;

    virtual void init_id() override;

    virtual std::string to_string() const override;

    void parse_params();

    static std::deque<size_t> get_predict_chunks(const std::deque<std::pair<double, double>> &chunks_score);

    void clean_chunks();

    arma::mat feature_chunk_t(const arma::uvec &ixs_i);

    arma::mat predict_chunk_t(const arma::mat &x_predict);

public:
    static std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
    cuvalidate(const double lambda, const double gamma_param, const unsigned lag, const arma::mat &tune_cuml, const arma::mat &train_cuml,
               const arma::mat &tune_label_chunk, const mmm_t &train_L_m, const double labels_sf, magma_queue_t ma_queue);

    std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
    cuvalidate_batched(const double lambda, const double gamma_param, const unsigned lag, const arma::mat &tune_cuml, const arma::mat &train_cuml,
                              const arma::mat &tune_label_chunk, const mmm_t &train_L_m, const double labels_sf, magma_queue_t ma_queue);

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
    static void solve_opt(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const unsigned iters);

    static void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, arma::mat &w, const unsigned iters);

    static std::deque<arma::mat> solve_batched_irwls(
            const std::deque<arma::mat> &K_epsco, const std::deque<arma::mat> &K, const std::deque<arma::mat> &rhs, const size_t iters,
            const magma_queue_t &magma_queue, const size_t gpu_phy_id);

    static arma::mat do_ocl_solve(const double *host_a, double *host_b, const int m, const unsigned nrhs);

    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    dtype(OnlineMIMOSVR::gradient) get_gradient_level() const noexcept;

    dtype(OnlineMIMOSVR::level) get_decon_level() const noexcept;

    dtype(OnlineMIMOSVR::step) get_step() const noexcept;

    dtype(OnlineMIMOSVR::multiout) get_multiout() const noexcept;

    dtype(OnlineMIMOSVR::samples_trained) get_samples_trained_number() const noexcept;

    void clear_kernel_matrix();

    void init_manifold(const SVRParameters_ptr &p, const bpt::ptime &last_manifold_time);

    OnlineMIMOSVR_ptr get_manifold();

    dtype(OnlineMIMOSVR::param_set) get_param_set() const noexcept;

    dtype(OnlineMIMOSVR::param_set) &get_param_set() noexcept;

    void set_param_set(const dtype(OnlineMIMOSVR::param_set) &param_set_);

    void set_params(const SVRParameters_ptr &p_svr_parameters_, const unsigned chunk_ix = 0);

    void set_params(const SVRParameters &param, const unsigned chunk_ix = 0);

    SVRParameters &get_params(const unsigned chunk_ix = 0) const;

    SVRParameters_ptr get_params_ptr(const unsigned chunk_ix = 0) const;

    SVRParameters_ptr is_manifold() const;

    bool is_gradient() const;

    static bool needs_tuning(const t_param_set &param_set);

    bool needs_tuning() const;

    void tune();

    void tune_fast();

    void recombine_params(const unsigned chunkix, const unsigned stepix);

#if defined(TUNE_HYBRID)
    double produce_kernel_inverse_order(const SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    static double get_gamma_range_variance(const unsigned train_len);

    static unsigned get_full_train_len(const unsigned n_rows, const unsigned decrement);

    size_t get_num_chunks();

    static size_t get_num_chunks(const unsigned n_rows, const unsigned chunk_size_);

    static std::deque<arma::uvec> generate_indexes(const unsigned n_rows_dataset, const unsigned decrement, const unsigned max_chunk_size);

    arma::uvec get_other_ixs(const unsigned i) const;

    std::deque<arma::uvec> generate_indexes() const;

    arma::mat predict(const arma::mat &x_predict);

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &time);

    arma::mat manifold_predict(const arma::mat &x_predict) const;

    t_gradient_data produce_residuals();

    void learn(const arma::mat &new_x, const arma::mat &new_y, const arma::vec &new_ylk, const bpt::ptime &last_value_time,
               const bool temp_learn = false, const std::deque<unsigned int> &forget_ixs = {});

    void batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const bpt::ptime &last_value_time, const matrices_ptr &precalc_kernel_matrices = nullptr);

    arma::mat &get_features();

    arma::mat &get_labels();

    arma::uvec get_active_ixs() const;

    static mat_ptr prepare_Ky(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t,
                              const bpt::ptime &predict_time, const bpt::ptime &trained_time);

    static mat_ptr prepare_Ky(const SVRParameters &svr_parameters, const arma::mat &x_train_t, const arma::mat &x_predict_t);

    static mat_ptr prepare_K(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &x_t, const bpt::ptime &time);

    static mat_ptr prepare_K(const SVRParameters &params, const arma::mat &x_t);

    static mat_ptr prepare_Z(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    static mat_ptr prepare_Z(const SVRParameters &params, const arma::mat &features_t);

    static mat_ptr prepare_Zy(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t,
                              const bpt::ptime &time, const bpt::ptime &trained_time);

    static mat_ptr prepare_Zy(const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t);

    static std::shared_ptr<std::deque<arma::mat>> prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t);

    static mat_ptr all_cumulatives(const SVRParameters &p, const arma::mat &features_t);

    static double calc_gamma(const arma::mat &Z, const double train_L_m);

    static double calc_epsco(const arma::mat &K);

    void calc_weights(const unsigned int chunk_ix, const unsigned int iters);

    void update_all_weights();

    static double calc_qgamma(const double Z_mean, const double Z_minmax, const double L_mean, const double L_minmax, const double train_len, const double q);
};

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

__global__ void G_div_inplace(double *__restrict__ const x, const double a, const unsigned n);


} // datamodel
} // svr

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OY_));               \
    (_MY_) = (_NY_) - (_OY_); }
