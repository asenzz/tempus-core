#pragma once

// Bagged MIOC-SVR on chunks
// TODO Multiple parameters per chunk, finish gradient boosting, warm-start online solver, test manifold


#include <cstdarg>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <armadillo>
#include <vector>

#include "common/constants.hpp"
#include "common/defines.h"
#include "model/SVRParameters.hpp"
#include "common/compatibility.hpp"

namespace svr {

#define CHUNK_SIZE CHUNK_DECREMENT // Higher for higher precision and higher GPU use
constexpr unsigned IRWLS_ITER_BATCH = 99;
constexpr unsigned IRWLS_ITER_ONLINE = 4;
#define OUTLIER_TEST

constexpr double OUTLIER_ALPHA = 1e-6;
#ifdef OUTLIER_TEST
constexpr double SANE_PREDICT = 1e6;
#endif


/* Auto cost and gamma */
constexpr double DE_FINE_DIVISOR = 10; // > 1
// constexpr double MULTIPLE_EPSCO = 2; // TODO Try C_input_obseg * .5, .25  // 2. for price, C_input_obseg for direction, epsilon-cost multiple, above 1, smaller is better and slower
const std::deque<double> C_gamma_multis {1., 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7}; // Full gammas multis span
#define FINER_GAMMA_TUNE

// const double TUNE_EPSCOST_MAX = svr::common::C_input_obseg_labels;
// const double TUNE_EPSCOST_MIN = 1 / svr::common::C_input_obseg_labels; // std::pow<double>(svr::common::C_input_obseg_labels, -1);
#ifdef FINER_GAMMA_TUNE
// constexpr double MULTIPLE_EPSCO_FINE = MULTIPLE_EPSCO;
constexpr double FINE_GAMMA_MULTIPLE = 1. + 1. / DE_FINE_DIVISOR;
constexpr double C_fine_gamma_mult = 10;
constexpr double C_fine_gamma_div = 10;
#else
// #define MULTIPLE_EPSCO_FINE     (1.33)
#define FINE_GAMMA_MULTIPLE     (1.66)
#endif

constexpr double BEST_PREDICT_CHUNKS_DIVISOR = 1; // above and 1
constexpr double VALIDATION_SIZE = double(EMO_TUNE_VALIDATION_WINDOW) / double(CHUNK_SIZE); // Grid search validation row count
constexpr unsigned AUTO_GU_CT = 1e9; // High count disables auto gu during online learning, yields better precision, faster train
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
    double score = std::numeric_limits<double>::max();
    datamodel::SVRParameters_ptr p_params = nullptr;
    std::vector<arma::mat> *p_predictions = nullptr;
    std::vector<arma::mat> *p_labels = nullptr;
    std::vector<arma::mat> *p_last_knowns = nullptr;

    t_param_preds(const t_param_preds &param_preds) :
            score(param_preds.score), p_params(param_preds.p_params), p_predictions(param_preds.p_predictions), p_labels(param_preds.p_labels), p_last_knowns(param_preds.p_last_knowns)
    {}

    void clear() // Release memory
    {
        p_params.reset();
        p_predictions->clear();
        delete p_predictions;
        p_labels->clear();
        delete p_labels;
        p_last_knowns->clear();
        delete p_last_knowns;
    }

    t_param_preds(const double score, std::shared_ptr<datamodel::SVRParameters> &params, std::vector<arma::mat> *predictions, std::vector<arma::mat> *labels, std::vector<arma::mat> *last_knowns) :
            score(score), p_params(params), p_predictions(predictions), p_labels(labels), p_last_knowns(last_knowns) {}
};
typedef std::shared_ptr<t_param_preds> t_param_preds_ptr;

// typedef std::tuple<double /* score */, double /* cost */, double /* gamma */, double /* lambda */, std::vector<arma::mat>> t_param_preds;
struct param_preds_cmp
{
    bool operator()(const t_param_preds_ptr &lhs, const t_param_preds_ptr &rhs) const
    {
        return lhs->score < rhs->score;
    }
};

typedef std::map<uint64_t, std::set<t_param_preds_ptr, param_preds_cmp>> param_preds_set_t;
typedef std::vector<param_preds_set_t> predictions_t;

enum class MimoType // : uint8_t // "enum class" defines this as a scoped enumeration instead of a standard enumeration
{
    single = 1,
    twin = 2,
};

class OnlineMIMOSVR;

using OnlineMIMOSVR_ptr = std::shared_ptr<OnlineMIMOSVR>;

class OnlineMIMOSVR
{
    static const std::vector<size_t> LINEAR_LEVELS;
    static double constexpr ETA = 0.0001;

    std::mutex learn_mx;

    class MimoBase;

    size_t temp_idx = std::numeric_limits<size_t>::max();
    size_t samples_trained = 0;

    matrix_ptr p_features = std::make_shared<arma::mat>();
    matrix_ptr p_labels = std::make_shared<arma::mat>();
    datamodel::SVRParameters_ptr p_svr_parameters = nullptr; // TODO Make it a vector for every chunk
    OnlineMIMOSVR_ptr p_manifold = nullptr;
    double labels_scaling_factor = 1;
    matrices_ptr p_kernel_matrices = nullptr;
    std::vector<arma::mat> r_matrix;
    std::mutex r_mx;
    MimoType mimo_type;
    std::map<uint8_t, MimoBase> main_components;
    std::vector<arma::uvec> ixs;

    const size_t multistep_len;

    void init_mimo_base(const double &epsilon, const MimoType &mimo_type);

    bool save_kernel_matrix = false;

    class MimoBase
    {
    public:
        std::vector<arma::mat> chunk_weights;
        arma::mat total_weights;
        double epsilon;
        std::vector<double> mae_chunk_values;
    };

    void add_to_r_matrix(const size_t inner_ix, const size_t outer_ix, const double correction);

    void remove_from_r_matrix(const size_t inner_ix, const size_t outer_ix);

    std::vector<double> auto_gu;

    static double go_for_variable_gu(const arma::mat &kernel, const arma::mat &rhs, double &gu, const bool fine_search = true);

public:
    // Move to solver module
    static arma::mat call_gpu_oversolve(const arma::mat &Left, const arma::mat &Right);
    static arma::mat call_gpu_dynsolve(const arma::mat &left, const arma::mat &right);
    static void call_gpu_dynsolve(const arma::mat &left, const arma::mat &right, arma::mat &output);
    static void solve_irwls(const arma::mat &epsilon_eye_K, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters);
    static arma::mat arma_multisolve(const arma::mat &epsilon_eye_K, const arma::mat &Z, const arma::mat &rhs);
    static arma::mat do_ocl_solve(const double *host_a, double *host_b, const int m, const int nrhs);
    static void solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, arma::mat &solved, const size_t iters);
    static arma::mat solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, const size_t iters);
    static arma::mat direct_solve(const arma::mat &a, const arma::mat &b);

    ssize_t get_samples_trained_number() const
    {
        return samples_trained;
    }

    void init_r_matrix();

    arma::mat get_r_matrix()
    {
        return r_matrix[0];
    }

    void clear_kernel_matrix()
    {
        for (auto &k: *p_kernel_matrices) k.clear();
    }

    OnlineMIMOSVR(const datamodel::SVRParameters_ptr &p_svr_parameters_, const MimoType type = MimoType::single, const size_t multistep_len = svr::common::__multistep_len);

    OnlineMIMOSVR(const datamodel::SVRParameters_ptr &p_boot_svr_parameters, const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain, const bool update_r_matrix = svr::common::__dont_update_r_matrix, const matrices_ptr &kernel_matrices = nullptr, const bool pseudo_online = false, const MimoType type = MimoType::single, const size_t multistep_len = svr::common::__multistep_len);

    OnlineMIMOSVR() : mimo_type(MimoType::single), multistep_len(common::__multistep_len)
    {
        init_mimo_base(std::numeric_limits<double>::min(), MimoType::single);
        LOG4_WARN("Created OnlineMIMOSVR object with default constructor and default multistep_len " << multistep_len);
    }

    static OnlineMIMOSVR_ptr init_manifold(const arma::mat &features, const arma::mat &labels, const datamodel::SVRParameters &svr_parameters);

    void init_manifold(const arma::mat &features, const arma::mat &labels);

    arma::mat &get_weights(uint8_t type);

    arma::mat get_weights(uint8_t type) const;

    datamodel::SVRParameters_ptr get_svr_parameters_ptr() const
    {
        return p_svr_parameters;
    }

    datamodel::SVRParameters_ptr &get_svr_parameters_ptr()
    {
        return p_svr_parameters;
    }

    svr::datamodel::SVRParameters get_svr_parameters() const
    {
        return *p_svr_parameters;
    }

    svr::datamodel::SVRParameters &get_svr_parameters()
    {
        return *p_svr_parameters;
    }

    void set_svr_parameters(const datamodel::SVRParameters_ptr &p_svr_parameters_)
    {
        p_svr_parameters = p_svr_parameters_;
    }

    void set_svr_parameters(const datamodel::SVRParameters &svr_parameters_)
    {
        *p_svr_parameters = svr_parameters_;
    }

    void do_over_train_zero_epsilon(const arma::mat &xx_train, const arma::mat &yy_train, const size_t chunk_idx);

    static arma::mat get_reference_distance_matrix(const arma::mat &L);

    static void tune_kernel_params(param_preds_set_t &predictions, datamodel::SVRParameters_ptr &_p_svr_parameters, const arma::mat &features, const arma::mat &labels, const arma::mat &last_knowns, size_t chunk_ix = std::numeric_limits<size_t>::max());

    static std::tuple<std::vector<arma::mat>, std::vector<double>> get_reference_matrices(const arma::mat &L, const datamodel::SVRParameters &svr_parameters);

#if defined(TUNE_HYBRID)
    static double produce_kernel_inverse_order(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train);
#endif

    static matrices_ptr produce_kernel_matrices(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train, const arma::mat &y_train, const OnlineMIMOSVR_ptr &p_manifold);

    static std::vector<arma::mat> produce_labels(const arma::mat &y_train, const datamodel::SVRParameters &svr_parameters);

    static std::vector<arma::uvec> get_indexes(const size_t n_rows, const datamodel::SVRParameters &svr_parameters);

    arma::uvec get_other_ixs(const size_t i) const;

    std::vector<arma::uvec> get_indexes() const;

    void batch_train(const matrix_ptr &p_xtrain, const matrix_ptr &p_ytrain, const bool update_r_matrix, const matrices_ptr &kernel_matrices = nullptr, const bool pseudo_online = false);

    arma::mat chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time) const;

    arma::mat single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const;

    inline arma::mat get_kernel_matrix()
    {
        return p_kernel_matrices->at(0);
    }

    size_t learn(const arma::mat &new_x_train, const arma::mat &new_y_train, const bool temp_learn = false, const bool update_r_matrix = false, const size_t forget_index = std::numeric_limits<size_t>::max(), const bpt::ptime &label_time = bpt::special_values::not_a_date_time);

#if 0
    void best_forget(const bool dont_update_r_matrix);

    void forget(const size_t inner_ix, const size_t outer_ix, const bool dont_update_r_matrix);
#endif

    arma::mat &get_features();

    arma::mat &get_labels();

    size_t get_multistep_len() const
    {
        return multistep_len;
    }

    MimoType get_mimo_type() const
    {
        return mimo_type;
    }

    void update_total_weights();

    void update_ixs(const size_t ix);

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

    static OnlineMIMOSVR_ptr
    load_online_mimosvr(const char *filename);

    bool
    save_online_mimosvr(const char *filename);

    template<typename S> static bool
    save_online_svr(const OnlineMIMOSVR &osvr, S &output_stream);

    template<typename S> static OnlineMIMOSVR_ptr
    load_online_svr(S &input_stream);

    explicit OnlineMIMOSVR(std::stringstream &input_stream);

    friend void log_model(const svr::OnlineMIMOSVR &m, const std::vector<arma::mat> &chunk_kernels);

    template<typename S>
    static bool save_onlinemimosvr_no_weights_no_kernel(const OnlineMIMOSVR &osvr, S &output_stream);

    template<typename S>
    static OnlineMIMOSVR_ptr load_onlinemimosvr_no_weights_no_kernel(S &input_stream);

    static OnlineMIMOSVR_ptr load_onlinemimosvr_no_weights_no_kernel(const char *filename);

    bool save_onlinemimosvr_no_weights_no_kernel(const char *filename);

    static std::vector<size_t> sort_indexes(const std::vector<double> &v);

    static arma::mat
    init_predict_kernel_matrix(
            const datamodel::SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &x_predict,
            const OnlineMIMOSVR_ptr &p_manifold,
            const bpt::ptime &predicted_time);

    static arma::mat init_predict_manifold_matrix(
        const arma::mat &x_train,
        const arma::mat &x_predict,
        const OnlineMIMOSVR_ptr &p_manifold);

    static arma::mat
    init_dist_predict_kernel_matrix(
            const datamodel::SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &x_predict,
            const OnlineMIMOSVR_ptr &p_manifold);

    static void init_kernel_matrix(
            const datamodel::SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &y_train,
            arma::mat &kernel_matrix,
            const OnlineMIMOSVR_ptr &p_manifold);

    static void
    prepare_manifold_kernel(
            const arma::mat &x_train,
            const arma::mat &y_train,
            arma::mat &kernel_matrix,
            const OnlineMIMOSVR_ptr &p_manifold);

    static std::vector<arma::mat> prepare_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features);

    static arma::mat *prepare_K(const datamodel::SVRParameters &params, const arma::mat &features_t, const double meanabs_labels, const size_t train_len, const arma::mat &labels);
    static arma::mat *prepare_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, size_t len = 0);
    static arma::mat *prepare_Zy(const datamodel::SVRParameters &params, const arma::mat &features /* transposed*/, const arma::mat &predict_features /* transposed*/, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);
    static std::vector<arma::mat> get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features /* transposed */, const bpt::ptime &pred_time = bpt::special_values::not_a_date_time);
    static arma::mat &
    get_cached_K(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const double meanabs_labels, const size_t train_len);
    static arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const size_t short_size_X = 0);
    static arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features /* transposed */, const arma::mat &predict_features /* transposed */, const bpt::ptime &pred_time);
    static void clear_Zy_cache();

    static size_t get_chunk_num(const size_t decrement);

    static size_t get_manifold_nrows(const size_t n_rows);

    void calc_weights(const size_t chunk_ix, const arma::mat &y_train, const double C, MimoBase &component, const size_t iters);

    static std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
    future_validate(
            const size_t start_ix,
            svr::OnlineMIMOSVR &online_svr,
            const arma::mat &features,
            const arma::mat &labels,
            const arma::mat &last_knowns,
            const std::vector<bpt::ptime> &times,
            const bool single_pred = false,
            const double scale_label = 1,
            const double dc_offset = 0);
};

} // svr

using OnlineMIMOSVR_ptr = std::shared_ptr<svr::OnlineMIMOSVR>;
