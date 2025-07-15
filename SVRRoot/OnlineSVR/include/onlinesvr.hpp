#pragma once

// Bagged MIOC-SVR on chunks
// TODO test gradient boosting, find warm-start online solver, test manifold

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/math/ccmath/ccmath.hpp>
#include <oneapi/tbb/mutex.h>
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

#define SINGLE_CHUNK_LEVEL
#define USE_MAGMA
#define FORGET_MIN_WEIGHT

/*
 * Kernel matrix is set like this:
    auto kernel = IKernel<double>::get_kernel(template_parameters.get_kernel_type(),template_parameters);
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
    friend class boost::serialization::access;

    constexpr static unsigned C_epscos_len = 1;
    static constexpr uint16_t C_solve_opt_particles = 100;
    static constexpr double C_diff_coef = 1;

    t_weight_scaling_factors weight_scaling_factors;

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
    uint16_t multiout = common::C_default_multiout;
    uint32_t max_chunk_size;
    uint16_t gradient = C_default_svrparam_grad_level;
    uint16_t level = C_default_svrparam_decon_level;
    uint16_t step = C_default_svrparam_step;
    uint16_t active_chunks, start_predict_chunk, projection = 0;

    virtual void init_id() override;

    virtual std::string to_string() const override;

    void parse_params();

    static std::deque<size_t> get_predict_chunks(const std::deque<std::pair<double, double>> &chunks_score);

    void clean_chunks();

    arma::mat feature_chunk_t(const arma::uvec &ixs_i);

    arma::mat predict_chunk_t(const arma::mat &x_predict);

public:
    static constexpr magma_int_t C_rbt_iter = 40;
    static constexpr float C_rbt_threshold = 0; // [0..1] default 1
    static constexpr float C_chunk_overlap = 0; // TODO Consider removing overlap logic, Do not use with manifolds 1. - 1. / 4.; // Chunk rows overlap ratio [0..1], higher generates more chunks
    static constexpr float C_chunk_offlap = 1. - C_chunk_overlap;
    static constexpr float C_chunk_tail = .1;
    static constexpr float C_chunk_header = 1. - C_chunk_tail;
    static constexpr uint16_t C_end_chunks = 1; // [1..1/offlap]

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

    std::stringstream save() const;

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
    static void solve_opt(const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const uint32_t iters, t_weight_scaling_factors &weight_scaling_factors, const uint16_t chunk);

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

    void tune_sys();

    void recombine_params(const unsigned chunkix, const unsigned stepix);

#if defined(TUNE_HYBRID)
    double produce_kernel_inverse_order(const SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    static double get_gamma_range_variance(const unsigned train_len);

    static uint32_t get_full_train_len(const uint32_t n_rows, const uint32_t decrement);

    uint16_t get_num_chunks() const;

    static uint16_t get_num_chunks(const uint32_t n_rows, const uint32_t chunk_size_);

    static std::deque<arma::uvec> generate_indexes(const bool projection, const uint32_t n_rows_dataset, const uint32_t decrement, const uint32_t max_chunk_size);

    arma::uvec get_other_ixs(const uint16_t i) const; // Get row indexes not belonging to the chunk

    std::deque<arma::uvec> generate_indexes() const;

    arma::uvec get_active_ixs() const;

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &time = bpt::not_a_date_time);

    arma::mat manifold_predict(const arma::mat &x_predict, const boost::posix_time::ptime &time) const;

    t_gradient_data produce_residuals();

    void learn(const arma::mat &new_x, const arma::mat &new_y, const arma::vec &new_ylk, const arma::mat &new_w, const bpt::ptime &last_value_time,
               const bool temp_learn = false, const std::deque<uint32_t> &forget_ixs = {});

    void batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const mat_ptr &p_input_weights_, const bpt::ptime &last_value_time,
                     const matrices_ptr &precalc_kernel_matrices = {});

    arma::mat &get_features();

    arma::mat &get_labels();

    static std::tuple<double, double> calc_gamma(const arma::mat &Z, const arma::mat &L);

    static arma::vec calc_gammas(const arma::mat &Z, const arma::mat &L);

    static double calc_epsco(const arma::mat &K, const arma::mat &labels);

    static std::array<double, C_epscos_len> get_epscos(const double K_mean);

#ifdef INSTANCE_WEIGHTS

    static arma::mat calc_weights(
            arma::mat K, const arma::mat &labels, const arma::mat &instance_weights_matrix,
                                  const double epsco, const uint16_t iters_irwls, const uint16_t iters_opt,
                                  t_weight_scaling_factors &weight_scaling_factors, const uint16_t chunk);

#else

    static arma::mat calc_weights(const arma::mat &K, const arma::mat &labels, const double epsco,
                                  const uint16_t iters_irwls, const uint32_t iters_opt,
                                  t_weight_scaling_factors &weight_scaling_factors, const uint16_t chunk);

#endif

    void calc_weights(const uint16_t chunk_ix, const uint16_t iters_irwls, const uint32_t iters_opt);

    static arma::mat instance_weight_matrix(const arma::uvec &ixs, const arma::mat &weights);

    void update_all_weights();

    static void self_predict(const unsigned m, const unsigned n, CRPTRd K, CRPTRd w, CRPTRd rhs, RPTR(double) diff);

    static arma::mat self_predict(const arma::mat &K, const arma::mat &w, const arma::mat &rhs);

    template<typename T> static inline arma::Mat<T> prepare_labels(const arma::Mat<T> &labels)
    {
        return labels * labels.n_elem - arma::sum(arma::vectorise(labels));
    }

};

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

// ICPX bug forced to move this out of cuvalidate
constexpr uint8_t get_streams_per_gpu(const uint32_t n_rows)
{
    const uint32_t C_max_alloc_gpu = 4232085504; // NVidia V100 16GB has malloc limit of VRAM/4
    return boost::math::ccmath::fmax(1, boost::math::ccmath::round(.04 * C_max_alloc_gpu / (n_rows * n_rows)));
}

class cusys {
    static constexpr auto streams_per_gpu = 6;
    static const uint16_t n_gpus;

    static constexpr uint32_t train_clamp = 0;
    SVRParameters template_parameters;
    const solvers::mmm_t &L3m;
    const bool weighted;
    const uint16_t dim, lag_tile_width, lag;
    const uint32_t n, train_len, tune_len, tune_len_n, calc_start, calc_len, train_F_rows, train_F_cols; // calc len of 3000 seems to work best
    const uint64_t K_train_len, K_train_size, train_len_n, train_n_size, tune_n_size, K_tune_len;
    const arma::mat ref_K;
    const double ref_K_mean, ref_K_meanabs;

    struct dev_ctx {
        struct stream_ctx {
            cudaStream_t custream;
            cublasHandle_t cublas_H;
            magma_queue_t ma_queue;
            double *d_K_train, *K_train_off, *j_work, *j_K_work, *d_best_weights;
        };
        double *d_train_label_chunk, *train_label_off, *d_train_cuml, *d_train_W, *d_K_train_l, *d_ref_K;
        std::deque<stream_ctx> sx;
    };
    std::deque<dev_ctx> dx;

public:
    static SVRParameters make_tuning_template(const SVRParameters &sample_parameters);

    cusys(const uint16_t lag, const arma::mat &train_cuml, const arma::mat &train_label_chunk, const arma::mat &train_W, const solvers::mmm_t &L3m, const SVRParameters &parameters);
    ~cusys();
    std::tuple<double, double, double> operator()(const double lambda, const double tau) const;
};

} // datamodel
} // svr

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OX_));               \
    (_MY_) = (_NY_) - (_OY_); }

#include "onlinesvr_persist.tpp"
