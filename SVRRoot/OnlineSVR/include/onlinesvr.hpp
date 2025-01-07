#pragma once

// Bagged MIOC-SVR on chunks
// TODO test gradient boosting, find warm-start online solver, test manifold

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/math/ccmath/ccmath.hpp>
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
#include "cuqrsolve.cuh"

namespace svr {
namespace business {
class calc_cache;
}

namespace datamodel {

constexpr uint32_t C_interlace_manifold_factor = 20; // Every Nth row is used from a manifold dataset to train the produced model
constexpr float C_tune_range_min_lambda = 0;
constexpr float C_tune_range_max_lambda = 10;
constexpr float C_chunk_overlap = 1. - 1. / 4.; // Chunk rows overlap ratio [0..1], higher generates more chunks
constexpr float C_chunk_offlap = 1. - C_chunk_overlap;
constexpr float C_chunk_tail = .1;
constexpr float C_chunk_header = 1. - C_chunk_tail;
constexpr uint16_t C_predict_chunks = 1; // TODO Review. Best chunks used for predictions
constexpr uint16_t C_end_chunks = 1; // [1..1/offlap]
constexpr magma_int_t C_rbt_iter = 40; // default 30
constexpr float C_rbt_threshold = 0; // [0..1] default 1
constexpr uint32_t C_features_superset_coef = 100; // [1..+inf)
constexpr float C_solve_opt_coef = 2; // Rows count multiplier to calculate iterations for NL solver
constexpr uint16_t C_solve_opt_particles = 80;
constexpr uint16_t C_weight_cols = 1;
#ifdef EMO_DIFF
constexpr double C_diff_coef = 1;
#else
constexpr double C_diff_coef = 1;
#endif

#define SINGLE_CHUNK_LEVEL
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
    constexpr static unsigned C_epscos_len = 1;

    bigint model_id = 0;
    bpt::ptime last_trained_time;
    Dataset_ptr p_dataset;
    std::string column_name;
    std::set<size_t> tmp_ixs;
    size_t samples_trained = 0;
    mat_ptr p_features, p_labels, p_input_weights;
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

    template<class A> void serialize(A &ar, const unsigned version);

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
    static void solve_opt(const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const uint16_t iters_irwls);

    static void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, arma::mat &w, const uint16_t iters);

    static std::deque<arma::mat> solve_batched_irwls(
            const std::deque<arma::mat> &K_epsco, const std::deque<arma::mat> &K, const std::deque<arma::mat> &rhs, const size_t iters,
            const magma_queue_t &magma_queue, const size_t gpu_phy_id);

    static arma::mat do_ocl_solve(CPTRd host_a, double *host_b, const int m, const unsigned nrhs);

    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    DTYPE(OnlineMIMOSVR::gradient) get_gradient_level() const noexcept;

    DTYPE(OnlineMIMOSVR::level) get_decon_level() const noexcept;

    DTYPE(OnlineMIMOSVR::step) get_step() const noexcept;

    DTYPE(OnlineMIMOSVR::multiout) get_multiout() const noexcept;

    DTYPE(OnlineMIMOSVR::samples_trained) get_samples_trained_number() const noexcept;

    void clear_kernel_matrix();

    void init_manifold(const SVRParameters_ptr &p, const bpt::ptime &last_manifold_time);

    OnlineMIMOSVR_ptr get_manifold();

    DTYPE(OnlineMIMOSVR::param_set) get_param_set() const noexcept;

    DTYPE(OnlineMIMOSVR::param_set) &get_param_set() noexcept;

    void set_param_set(const DTYPE(OnlineMIMOSVR::param_set) &param_set_);

    void set_params(const SVRParameters_ptr &p_svr_parameters_, const uint16_t chunk_ix = 0);

    void set_params(const SVRParameters &param, const uint16_t chunk_ix = 0);

    SVRParameters &get_params(const uint16_t chunk_ix = 0) const;

    SVRParameters_ptr get_params_ptr(const uint16_t chunk_ix = 0) const;

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

    uint16_t get_num_chunks() const;

    static uint16_t get_num_chunks(const uint32_t n_rows, const uint32_t chunk_size_);

    static std::deque<arma::uvec> generate_indexes(const unsigned n_rows_dataset, const unsigned decrement, const unsigned max_chunk_size);

    arma::uvec get_other_ixs(const unsigned i) const;

    std::deque<arma::uvec> generate_indexes() const;

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &time = bpt::not_a_date_time);

    arma::mat manifold_predict(const arma::mat &x_predict) const;

    t_gradient_data produce_residuals();

    void learn(const arma::mat &new_x, const arma::mat &new_y, const arma::vec &new_ylk, const arma::mat &new_w, const bpt::ptime &last_value_time,
               const bool temp_learn = false, const std::deque<uint32_t> &forget_ixs = {});

    void batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const mat_ptr &p_input_weights_, const bpt::ptime &last_value_time,
                     const matrices_ptr &precalc_kernel_matrices = {});

    arma::mat &get_features();

    arma::mat &get_labels();

    arma::uvec get_active_ixs() const;

    static mat_ptr prepare_Ky(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t,
                              const bpt::ptime &predict_time, const bpt::ptime &trained_time);

    static mat_ptr prepare_Ky(const SVRParameters &svr_parameters, const arma::mat &x_train_t, const arma::mat &x_predict_t);

    static mat_ptr prepare_Ky(business::calc_cache &ccache, const datamodel::SVRParameters &params, const arma::mat &x_train_t, const arma::mat &x_predict_t,
                                      const bpt::ptime &predict_time, const bpt::ptime &trained_time, const uint8_t devices);

    static mat_ptr prepare_K(business::calc_cache &ccache, SVRParameters &params, const arma::mat &x_t, const bpt::ptime &time);

    static mat_ptr prepare_K(SVRParameters &params, const arma::mat &x_t);

    static mat_ptr prepare_Z(business::calc_cache &ccache, SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    static mat_ptr prepare_Z(SVRParameters &params, const arma::mat &features_t);

    static mat_ptr prepare_Zy(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t,
                              const bpt::ptime &predict_time, const bpt::ptime &trained_time);

    static mat_ptr prepare_Zy(const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t);

    static std::shared_ptr<std::deque<arma::mat>> prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t);

    static mat_ptr all_cumulatives(const SVRParameters &p, const arma::mat &features_t);

    static double calc_gamma(const arma::mat &Z, const arma::mat &L);

    static double calc_gamma(const arma::mat &Z, const arma::mat &L, const double bias);

    static arma::vec calc_gammas(const arma::mat &Z, const arma::mat &L, const double bias);

    static double calc_epsco(const arma::mat &K, const arma::mat &labels);

    static std::array<double, C_epscos_len> get_epscos(const double K_mean);

    static arma::mat calc_weights(arma::mat K, const arma::mat &labels, const arma::mat &instance_weights_matrix, const double epsco, const uint16_t iters_irwls,
                                  const uint16_t iters_opt);

    static arma::mat calc_weights(const arma::mat &K, const arma::mat &labels, const double epsco, const uint16_t iters_irwls, const uint16_t iters_opt);

    static arma::mat weight_matrix(const arma::uvec &ixs, const arma::mat &weights);

    void calc_weights(const uint16_t chunk_ix, const uint16_t iters_irwls, const uint16_t iters_opt);

    void update_all_weights();

    static void self_predict(const unsigned m, const unsigned n, CRPTRd K, CRPTRd w, CRPTRd rhs, RPTR(double) diff);

    static arma::mat self_predict(const arma::mat &K, const arma::mat &w, const arma::mat &rhs);

};

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

// ICPX bug forced to move this out of cuvalidate
constexpr uint8_t get_streams_per_gpu(const uint32_t n_rows)
{
    const uint32_t C_max_alloc_gpu = 4232085504; // NVidia V100 16GB has malloc limit of VRAM/4
    return boost::math::ccmath::fmax(1, boost::math::ccmath::round(.04 * C_max_alloc_gpu / (n_rows * n_rows)));
}

class cuvalidate {

    static constexpr auto streams_per_gpu = get_streams_per_gpu(common::C_default_kernel_max_chunk_len);
    static const uint16_t n_gpus;

    static constexpr uint8_t irwls_iters = 4, magma_iters = 4;
    static constexpr double one = 1, oneneg = -1, zero = 0;
    static constexpr uint32_t train_clamp = 0; // C_test_len
    const solvers::mmm_t &train_L_m;
    const double labels_sf, iters_mul;
    const bool weighted;
    const uint16_t dim, lag_tile_width, lag;
    const uint32_t m, n, train_len, tune_len, calc_start, calc_len, train_F_rows, train_F_cols, train_cuml_rows, tune_F_rows, tune_F_cols, tune_cuml_rows;
    const uint64_t mm, mm_size, K_train_len, K_train_size, train_len_n, train_n_size, tune_len_n, tune_n_size, K_tune_len;

    struct dev_ctx {
        struct stream_ctx {
            cudaStream_t custream;
            cublasHandle_t cublas_H;
            magma_queue_t ma_queue;
            double *d_K_train, *K_train_off, *d_K_epsco, *j_solved, *j_work, *j_K_work, *d_best_weights, *d_Kz_tune;
        };
        double *d_train_label_chunk, *d_tune_label_chunk, *train_label_off, *d_train_features_t, *d_train_cuml, *d_train_W, *d_tune_W, *d_tune_features_t, *d_tune_cuml;
        std::deque<stream_ctx> sx;
    };
    std::deque<dev_ctx> dx;

public:
    cuvalidate(
            const uint16_t lag,
            const arma::mat &tune_cuml, const arma::mat &train_cuml,
            const arma::mat &tune_features_t, const arma::mat &train_features_t,
            const arma::mat &tune_label_chunk, const arma::mat &train_label_chunk,
            const arma::mat &tune_W, const arma::mat &train_W, const solvers::mmm_t &train_L_m, const double labels_sf);
    ~cuvalidate();

    std::tuple<double, double, double, t_param_preds::t_predictions_ptr> operator()(const double lambda, const double gamma_bias, const double tau);
};

} // datamodel
} // svr

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OY_));               \
    (_MY_) = (_NY_) - (_OY_); }

#include "onlinesvr_persist.tpp"
