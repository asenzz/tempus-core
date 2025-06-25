#pragma once

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/math/ccmath/ccmath.hpp>
#include <oneapi/tbb/mutex.h>
#include <memory>
#include <set>
#include <deque>
#include <armadillo>
#include <magma_types.h>
#include <tuple>
#include <tuple>
#include <tuple>

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
#define FORGET_MIN_WEIGHT
// #define SOLVE_PRUNE // Matrix solver is PPrune instead of PETSc

struct t_gradient_data
{
    mat_ptr p_features, p_labels;

    t_gradient_data(const mat_ptr &p_features, const mat_ptr &p_labels) : p_features(p_features), p_labels(p_labels)
    {
    }
};

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;

class OnlineSVR;

using OnlineSVR_ptr = std::shared_ptr<OnlineSVR>;

class OnlineSVR final : public Entity
{
    friend class boost::serialization::access;

    bigint model_id = 0;
    bpt::ptime last_trained_time;
    Dataset_ptr p_dataset;
    std::string column_name;
    std::set<size_t> tmp_ixs;
    size_t samples_trained = 0;
    mat_ptr p_features, p_labels, p_input_weights;
    t_param_set param_set;
    matrices_ptr p_kernel_matrices;
    datamodel::dq_scaling_factor_container_t scaling_factors;
    std::deque<arma::mat> weight_chunks, train_feature_chunks_t, train_label_chunks;
    std::deque<arma::uvec> ixs;
    arma::vec chunks_score;
    arma::mat all_weights;
    uint16_t multiout = common::C_default_multiout;
    uint32_t max_chunk_size;
    uint16_t gradient = C_default_svrparam_grad_level;
    uint16_t level = C_default_svrparam_decon_level;
    uint16_t step = C_default_svrparam_step;
    PROPERTY(uint16_t, projection, 0);

    virtual void init_id() override;

    virtual std::string to_string() const override;

    void parse_params();

    arma::u32_vec get_predict_chunks() const;

    void clean_chunks();

    arma::mat feature_chunk_t(const arma::uvec &ixs_i) const;

    arma::mat predict_chunk_t(const arma::mat &x_predict) const;

public:
    const float chunk_offlap;

    static constexpr uint16_t C_end_chunks = 1; // [1..1/offlap]

    OnlineSVR(const bigint id, const bigint model_id, const t_param_set &param_set, const Dataset_ptr &p_dataset = nullptr);

    OnlineSVR(const bigint id, const bigint model_id, const t_param_set &param_set, const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown,
              const bpt::ptime &last_value_time, const matrices_ptr &p_kernel_matrices = nullptr, const Dataset_ptr &p_dataset = nullptr);

    explicit OnlineSVR(const bigint id, const bigint model_id, std::stringstream &input_stream);

    OnlineSVR();

    ~OnlineSVR() = default;

    void reset();

    bool operator==(OnlineSVR const &) const;

    template<typename S> static void save(const OnlineSVR &osvr, S &output_stream);

    std::stringstream save() const;

    template<typename S> static OnlineSVR_ptr load(S &input_stream);

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

    static std::deque<arma::mat> solve_batched_irwls(
        const std::deque<arma::mat> &K_epsco, const std::deque<arma::mat> &K, const std::deque<arma::mat> &rhs, const size_t iters,
        const magma_queue_t &magma_queue, const size_t gpu_phy_id);

    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    DTYPE(OnlineSVR::gradient) get_gradient_level() const noexcept;

    DTYPE(OnlineSVR::level) get_decon_level() const noexcept;

    DTYPE(OnlineSVR::step) get_step() const noexcept;

    DTYPE(OnlineSVR::multiout) get_multiout() const noexcept;

    DTYPE(OnlineSVR::samples_trained) get_samples_trained_number() const noexcept;

    void clear_kernel_matrix();

    DTYPE(OnlineSVR::param_set) get_param_set() const noexcept;

    DTYPE(OnlineSVR::param_set) &get_param_set() noexcept;

    void set_param_set(const DTYPE(OnlineSVR::param_set) &param_set_);

    void set_params(const SVRParameters_ptr &p_svr_parameters_, const uint16_t chunk_ix = 0);

    void set_params(const SVRParameters &param, const uint16_t chunk_ix = 0);

    SVRParameters &get_params(const uint16_t chunk_ix = 0) const;

    SVRParameters_ptr get_params_ptr(const uint16_t chunk_ix = 0) const;

    SVRParameters_ptr is_manifold() const;

    SVRParameters_ptr is_tft() const;

    bool is_gradient() const;

    static bool needs_tuning(const t_param_set &param_set);

    bool needs_tuning() const;

    void tune();

    static uint32_t get_full_train_len(const uint32_t n_rows, const uint32_t decrement);

    uint32_t get_num_chunks() const;

    static uint32_t get_num_chunks(const uint32_t n_rows, const uint32_t chunk_size_);

    arma::uvec get_other_ixs(const uint16_t i) const; // Get row indexes not belonging to the chunk

    std::deque<arma::uvec> generate_indexes() const;

    arma::uvec get_active_ixs() const;

    arma::mat predict(const arma::mat &x_predict, const bpt::ptime &time = bpt::not_a_date_time);

    t_gradient_data produce_residuals();

    void learn(const arma::mat &new_x, const arma::mat &new_y, const arma::mat &new_w, const bpt::ptime &last_value_time,
               const bool temp_learn = false, const std::deque<uint32_t> &forget_ixs = {});

    void batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const mat_ptr &p_input_weights_, const bpt::ptime &time,
                     const matrices_ptr &precalc_kernel_matrices = {});

    arma::mat &get_features();

    arma::mat &get_labels();

    static std::tuple<double, double> calc_gamma(const arma::mat &Z, const arma::mat &L);

    static arma::vec calc_gammas(const arma::mat &Z, const arma::mat &L);

    static double calc_epsco(const arma::mat &K, const arma::mat &labels);

    static double calc_weights(const arma::mat &K, const arma::mat &labels, const uint32_t iters_opt, const uint16_t iters_irwls, arma::mat &weights);

    static double d_calc_weights(const arma::mat &K_, const arma::mat &labels_, const uint32_t iter_opt, const uint16_t iter_irwls, arma::mat &weights);

    void calc_weights(const uint16_t chunk_ix, const uint32_t iter_opt, const uint16_t iter_irwls);

    static arma::mat instance_weight_matrix(const arma::uvec &ixs, const arma::mat &weights);

    void update_all_weights();

    static void self_predict(const unsigned m, const unsigned n, CRPTRd K, CRPTRd w, CRPTRd rhs, RPTR(double) diff);

    static arma::mat self_predict(const arma::mat &K, const arma::mat &w, const arma::mat &rhs);

    static double score_weights(const uint32_t m, const uint32_t n, CRPTRd K, CRPTRd w, CRPTRd rhs);

    template<typename T> static inline arma::Mat<T> prepare_labels(const arma::Mat<T> &labels)
    {
        return labels * labels.n_elem - arma::sum(arma::vectorise(labels));
    }

    void prepare_chunk(uint32_t i);

    void prepare_chunk(const SVRParameters_ptr &p);

    static SVRParameters make_tuning_template(const SVRParameters &example);
};

using OnlineSVR_ptr = std::shared_ptr<OnlineSVR>;

class cutuner
{
    static constexpr uint16_t streams_per_gpu = 1;
    const uint16_t n_gpus;
    SVRParameters template_parameters;
    const bool weighted;
    const uint32_t n, train_len, calc_start, calc_len, train_F_rows; // calc len of 3000 seems to work best
    const uint64_t K_train_len, K_train_size, K_calc_len, K_off, train_len_n, train_n_size;
    const arma::mat ref_K, train_F;
    const double ref_K_mean, ref_K_meanabs;

public:
    struct dev_ctx
    {
        struct stream_ctx
        {
            cudaStream_t custream;
            cublasHandle_t cublas_H;
            magma_queue_t ma_queue;
            double *d_K_train, *K_train_off;
        };

        double *d_train_F, *d_train_W, *d_ref_K, *d_D_paths;
        std::deque<stream_ctx> sx;
    };

    std::deque<dev_ctx> dx;

    cutuner(const arma::mat &train_F, const arma::mat &train_label_chunk, const arma::mat &train_W, const SVRParameters &parameters);

    ~cutuner();

    std::tuple<double, double, double> normalize_result(const dev_ctx &dx_, const dev_ctx::stream_ctx &dxsx, const SVRParameters &parameters) const;

    void prepare_second_phase(const SVRParameters &first_phase_parameters);

    std::tuple<double, double, double> phase1(const double tau, const double H, const double D, const double V) const;

    std::tuple<double, double, double> phase2(const double lambda) const;
};
} // datamodel
} // svr

#define MANIFOLD_SET(_MX_, _MY_, _NX_, _NY_, _OX_, _OY_) {  \
    (_MX_) = arma::join_rows((_NX_), (_OX_));               \
    (_MY_) = (_NY_) - (_OY_); }

#include "onlinesvr_persist.tpp"
