#pragma once

#include "common/constants.hpp"
#include "_deprecated/svm.h"
#include "_deprecated/smo.h"

#include "model/_deprecated/vektor.tcc"
#include "model/_deprecated/vmatrix.tcc"
#include "model/SVRParameters.hpp"
#include "util/math_utils.hpp"

#include <cstdarg>
#include <tuple>

#include <mutex>

// #define ONLINE_SVR_DEBUG

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/prod.hpp"

// Must be included after the viennacl definitions
#include "kernel_global_alignment.hpp"
#include "kernel_factory.hpp"


using svr::datamodel::vektor;
using svr::datamodel::vmatrix;
using svr::datamodel::SVRParameters;
using svr::datamodel::kernel_type_e;

namespace svr {

typedef enum class set_type_
{
    NO_SET,
    SUPPORT_SET,
    ERROR_SET,
    REMAINING_SET
} set_type_e, *set_type_e_ptr;


//Online SVR class

class OnlineSVR
{
    const size_t online_iters_limit_mult = common::__online_iters_limit_mult;
    const size_t online_learn_iter_limit = common::__online_learn_iter_limit;
    const size_t max_variations = common::__max_variations;
    const double smo_epsilon_divisor = common::__smo_epsilon_divisor;
    const double smo_cost_divisor = common::__smo_cost_divisor;
    const size_t stabilize_iterations_count = common::__stabilize_iterations_count;
    const ssize_t number_variations = common::__default_number_variations;
public:
    double get_smo_epsilon_divisor() const { return smo_epsilon_divisor; }
    double get_smo_cost_divisor() const { return smo_cost_divisor; }

public:
    // Initialization
    OnlineSVR();

    explicit OnlineSVR(const SVRParameters &svr_parameters_);

    explicit OnlineSVR(std::stringstream &input_stream);

    // Move assignment operator - invalidates the other object.
    OnlineSVR &operator=(OnlineSVR &&other) noexcept;

    ~OnlineSVR();

    void clear();

    OnlineSVR *clone() const;

    // Attributes Operations
    SVRParameters &get_svr_parameters();

    void set_svr_parameters(const SVRParameters &);

    bool get_stabilized_learning() const;

    void set_stabilized_learning(const bool stabilized_learning_arg);

    bool get_save_kernel_matrix() const;

    void set_save_kernel_matrix(const bool save_kernel_matrix_arg);

    ssize_t get_samples_trained_number() const;

    ssize_t get_support_set_elements_number() const;

    ssize_t get_error_set_elements_number() const;

    ssize_t get_remaining_set_elements_number() const;

    vektor<ssize_t> *get_support_set_indexes() const;

    vektor<ssize_t> *get_error_set_indexes() const;

    vektor<ssize_t> *get_remaining_set_indexes() const;

    void set_samples_trained_number(const size_t samples_num);

    void clear_indices();

    void add_to_support_set(const ssize_t index);

    void add_to_remaining_set(const ssize_t index);

    void add_to_error_set(const ssize_t index);

    void set_bias(const double bias);

    void add_weight(const double weight);

    void set_kernel_matrix(vmatrix<double> *matrix);

    void set_weights(vektor<double> *weights);

    void set_x(vmatrix<double> *x);

    void set_y(vektor<double> *y);

    void clear_R_matrix();

    void set_R_matrix(vmatrix<double> *r_matrix);

    // Learning Operations
    void train_batch(
            const arma::mat &X,
            const arma::colvec &Y);

    size_t train(
            const vmatrix<double> &X,
            const vektor<double> &Y);

    size_t train(
            const vmatrix<double> &features,
            const vektor<double> &labels,
            const vmatrix<double> &test_set_features,
            const vektor<double> &test_set_labels);

    size_t train(
            const vmatrix<double> &X,
            const vektor<double> &Y,
            const size_t training_size,
            const size_t test_size);

    size_t train(const vektor<double> &features, const double label);

    size_t forget(const vektor<ssize_t> &indexes);

    size_t forget(const ssize_t sample_index);

    size_t forget(const vektor<double> &features);

    void stabilize_by_warm_batch_start();

/*
 * TODO Implement!
 *
    void SelfLearning(vmatrix<double> *TrainingSetX,
            vektor<double> *TrainingSetY, vmatrix<double> *ValidationSetX,
            vektor<double> *ValidationSetY, double ErrorTollerance);

    static void CrossValidation(vmatrix<double> *TrainingSetX,
            vektor<double> *TrainingSetY, vektor<double> *EpsilonList,
            vektor<double> *CList, vektor<double> *KernelParamList,
            int SetNumber, char *ResultsFileName);

    static double CrossValidation(vektor<vmatrix<double> *> *SetX,
            vektor<vektor<double> *> *SetY, double Epsilon, double C,
            double KernelParam);

    static void LeaveOneOut(vmatrix<double> *TrainingSetX,
            vektor<double> *TrainingSetY, vektor<double> *EpsilonList,
            vektor<double> *CList, vektor<double> *KernelParamList,
            char *ResultsFileName);

    static double LeaveOneOut(vmatrix<double> *SetX, vektor<double> *SetY,
            double Epsilon, double C, double KernelParam);
*/

    // Predict/Margin Operations
    double predict(const vektor<double> &features) const;

    double predict(const ssize_t index) const;

    vektor<double> predict(const vmatrix<double> &features) const;

    vektor<double> predict_inplace_all() const;

    long double predict_stable(const vektor<double> &features);

    vektor<double> margin_inplace_all() const; // TODO Parallelize

    long double margin(const vektor<double> &X, const double Y) const;

    double margin(const size_t index) const;

    vektor<double> margin(const vmatrix<double> &X, const vektor<double> &Y);

    static const double calc_kernel_matrix_error(const arma::mat &y_train, const svr::smo::kernel_matrix &kernel_matrix);

    static double produce_kernel_matrices_variance(
            const SVRParameters &svr_parameters,
            const arma::mat &x_train,
            const arma::mat &y_train,
            const bool save_matrices);

    // Control Operations
    bool verify_kkt_conditions() const;

    bool
    verify_kkt_condition(
            const ssize_t sample_index,
            double &out_violation_weight,
            double &out_violation_in_sample_predict) const;

    double get_in_sample_training_error() const;

    void report_in_sample_training_error() const;

    bool load_online_svr(const char *filename);

    void load_online_svr(const svm_model &libsvm_model, const vmatrix<double> &learning_data,
                         const vektor<double> &reference_data);

    void load_online_svr(std::stringstream &input_stream);

    bool save_onlinesvr(const char *filename) const;

    void save_online_svr(std::stringstream &output_stream) const;

    // static
    template<typename S>
    static bool load_online_svr(OnlineSVR &osvr, S &input_stream);

    template<typename S>
    static bool load_online_svr_old(OnlineSVR &osvr, S &input_stream);

    template<typename S>
    static bool save_online_svr(const OnlineSVR &osvr, S &output_stream);

    vektor<double> *get_weights() const;

    vmatrix<double> *get_x() const;

    vmatrix<double> *get_r() const;

    vektor<double> *get_y() const;

    vektor<double> &get_in_sample_margins();

    vmatrix<double> *get_kernel_matrix() const;

    void build_kernel_matrix();

    void sanity_ensured();

    bool operator==(OnlineSVR const &) const;

    size_t size_of() const;

private:
    // Parameter Attributes
    SVRParameters svr_parameters;
    ssize_t samples_trained_number = 0; // how much observations were used in train
    size_t learn_calls = 0;
    size_t unlearn_calls = 0;
    bool stabilized_learning = false;
    bool save_kernel_matrix = false;

    // Training Set Attributes
    vmatrix<double> *X = nullptr;
    vektor<double> *Y = nullptr;
    vektor<double> *p_weights = nullptr;
    double bias = 0.; // negative rho

    // Work Set Attributes
    vektor<ssize_t> *p_support_set_indexes = nullptr;
    vektor<ssize_t> *p_error_set_indexes = nullptr;
    vektor<ssize_t> *p_remaining_set_indexes = nullptr;
    vmatrix<double> *p_r_matrix = nullptr;
    vmatrix<double> *p_kernel_matrix = nullptr;
    vektor<double> in_sample_margins;

    static std::mutex gpu_handler_mutex;
    static svr::common::gpu_context static_paramtune_context;

    // TODO: hook to config files, db etc.
    /*
    bool auto_error_tolerance = true;
    double error_tolerance = ERROR_TOLERANCE;
    */

    // Private Learning Operations
    void load_temporal_margin(const double sample_margin, vektor<double> &h);

    void violate_margin(const vektor<double> &h) const;

    void violate_weights() const;

    size_t learn(const vektor<double> &features, const double label);

    size_t unlearn(const ssize_t Index);

    vmatrix<double> q(const vektor<ssize_t> &indexes1, const vektor<ssize_t> &indexes2) const;

    vmatrix<double> q(const vektor<ssize_t> &v) const;

    vektor<double> q(const vektor<ssize_t> &v, const ssize_t index) const;

    vektor<double> q(const ssize_t index) const;

    long double q(const ssize_t index1, const ssize_t index2) const;

    // Matrix R Operations
    void add_to_r_matrix(const ssize_t sample_index, const set_type_e sample_old_set,
                         const vektor<double> &beta, vektor<double> &gamma);

    void remove_from_r_matrix(const ssize_t sample_index);

    vektor<double> find_beta(const ssize_t sample_index) const;

    vektor<double> find_gamma(const vektor<double> &beta, const ssize_t sample_index) const;

    long double find_gamma_sample(vektor<double> &beta, const ssize_t sample_index);

    // KernelMatrix Operations
    void add_to_kernel_matrix(const vektor<double> &features);

    void remove_from_kernel_matrix(const ssize_t sample_index);

    // Variation Operations
    double find_variation_lc1(const vektor<double> &h, const vektor<double> &gamma,
                              const ssize_t sample_index, const int direction) const;

    double find_variation_lc2(const ssize_t sample_index, const int direction) const;

    double find_variation_lc(const ssize_t sample_index) const;

    void find_variation_ls(
            const vektor<double> &h, const vektor<double> &beta, const int direction, vektor<double> &ls) const;

    void find_variation_le(
            const vektor<double> &h, const vektor<double> &gamma, const int direction, vektor<double> &le) const;

    void find_variation_lr(
            const vektor<double> &h, const vektor<double> &gamma, const int direction, vektor<double> &lr) const;

    void find_learning_min_variations(
            const vektor<double> &h,
            const vektor<double> &beta,
            const vektor<double> &gamma,
            const ssize_t sample_index,
            const ssize_t number_variations,
            vektor<double> &min_variations,
            vektor<ssize_t> &min_indexes,
            vektor<ssize_t> &flags) const;

    void find_learning_min_variation(
            const vektor<double> &h, const vektor<double> &beta, const vektor<double> &gamma,
            const ssize_t sample_index,
            double &min_variation,
            ssize_t &min_index, ssize_t &flag) const;

    void find_unlearning_min_variations(
            const vektor<double> &h,
            const vektor<double> &beta,
            const vektor<double> &gamma,
            const ssize_t sample_index,
            const ssize_t number_variations,
            vektor<double> &min_variations,
            vektor<ssize_t> &min_indexes,
            vektor<ssize_t> &flags) const;

    void find_unlearning_min_variation(const vektor<double> &h, const vektor<double> &beta,
                                       const vektor<double> &gamma, const ssize_t sample_index,
                                       double &out_min_variation,
                                       ssize_t &out_min_index, ssize_t &out_flag);

    // Set Operations
    void update_weights_bias(vektor<double> &h, vektor<double> &beta,
                             vektor<double> &gamma, const ssize_t sample_index, const double min_variation);

    void add_sample_to_remaining_set(const ssize_t sample_index);

    void add_to_support_set(vektor<double> &h, vektor<double> &beta,
                            vektor<double> &gamma, const ssize_t sample_index, const double min_variation);

    void add_to_error_set(const int sample_index, double MinVariation);

    void move_from_support_to_error_remaining_set(vektor<double> &h,
                                                  vektor<double> &beta,
                                                  vektor<double> &gamma,
                                                  const ssize_t min_index,
                                                  const double min_variation);

    void move_from_error_to_support_set(vektor<double> &h,
                                        vektor<double> &beta, vektor<double> &gamma, const ssize_t min_index,
                                        const double min_variation);

    void move_from_remaining_to_support_set(vektor<double> &h,
                                            vektor<double> &beta, vektor<double> &gamma, const ssize_t min_index,
                                            const double min_variation);

    void remove_from_support_set(const ssize_t sample_set_index);

    void remove_from_error_set(const ssize_t sample_set_index);

    void remove_from_remaining_set(const ssize_t sample_set_index);

    void remove_sample(const ssize_t sample_index);

    void reorder_set_indexes_decreasing(const ssize_t sample_index);

    /*
    // Other Control Operations
    bool VerifyKKTConditions(vektor<double> *H);

    bool VerifyKKTConditions(vektor<double> *H, int *SampleIndex,
                             int *SampleSetIndex, bool print_weights = false);
    */

    // Note that these methods cover similar, but not the same conditions as
    // VerifyKKTConditions and VerifyKKTCondition,
    // So calls to these two sets of functions should be applied carefully.
    bool violate_weights_support() const;

    bool violate_weights_remaining() const;

    bool violate_weights_error() const;

    bool violate_margin_support(const vektor<double> &h) const;

    bool violate_margin_remaining(const vektor<double> &h) const;

    bool violate_margin_error(const vektor<double> &h) const;

    void integrity_assured(const bool force = false);
};

} // svr

using OnlineSVR_ptr = std::shared_ptr<svr::OnlineSVR>;
