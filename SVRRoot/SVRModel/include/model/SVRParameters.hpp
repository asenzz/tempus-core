#pragma once

#include <ostream>
#include <string>
#include "common/defines.h"
#include "common/constants.hpp"
#include "common/logging.hpp"
#include "model/Entity.hpp"

//#define SMO_EPSILON 1e-3

namespace svr {
namespace datamodel {

typedef enum class kernel_type : int
{
    LINEAR = 0,
    POLYNOMIAL = 1,
    RBF = 2,
    RBF_GAUSSIAN = 3,
    RBF_EXPONENTIAL = 4,
    MLP = 5,
    GA = 6, // global alignment
    PATH = 7, // path kernel
    DEEP_PATH = 8,
    number_of_kernel_types = 9 // end of enum = invalid type
} e_kernel_type;

template<typename ST>
ST tostring(const datamodel::e_kernel_type kt)
{
    switch (kt) {
        case e_kernel_type::LINEAR:
            return "LINEAR";
        case e_kernel_type::POLYNOMIAL:
            return "POLYNOMIAL";
        case e_kernel_type::RBF:
            return "RBF";
        case e_kernel_type::RBF_GAUSSIAN:
            return "RBF_GAUSSIAN";
        case e_kernel_type::RBF_EXPONENTIAL:
            return "RBF_EXPONENTIAL";
        case e_kernel_type::GA:
            return "GA";
        case e_kernel_type::PATH:
            return "PATH";
        case e_kernel_type::DEEP_PATH:
            return "DEEP_PATH";
        default:
            return "UNKNOWN";
    }
}

e_kernel_type operator++(e_kernel_type &k_type);

e_kernel_type operator++(e_kernel_type &k_type, int);

class SVRParameters;

using SVRParameters_ptr = std::shared_ptr<datamodel::SVRParameters>;

struct less_SVRParameters_ptr
{
    bool operator()(const datamodel::SVRParameters_ptr &lhs, const datamodel::SVRParameters_ptr &rhs) const;
};

typedef std::set<datamodel::SVRParameters_ptr, less_SVRParameters_ptr> t_param_set;
typedef std::shared_ptr<t_param_set> t_param_set_ptr; // TODO Convert to a new class

// Default SVR parameters
constexpr uint16_t C_default_svrparam_decon_level = 0;
constexpr uint16_t C_default_svrparam_step = 0;
constexpr uint16_t C_default_svrparam_chunk_ix = 0;
constexpr uint16_t C_default_svrparam_grad_level = 0;
constexpr double C_default_svrparam_svr_cost = 0;
constexpr double C_default_svrparam_svr_epsilon = 0;
constexpr double C_default_svrparam_kernel_param1 = 0;
constexpr double C_default_svrparam_kernel_param2 = 0;
constexpr double C_default_svrparam_kernel_param_tau = .75;
constexpr uint32_t C_default_svrparam_decrement_distance = common::C_best_decrement;
constexpr double C_default_svrparam_adjacent_levels_ratio = 1;
constexpr svr::datamodel::e_kernel_type C_default_svrparam_kernel_type = svr::datamodel::e_kernel_type::PATH;
constexpr auto C_default_svrparam_kernel_type_uint = uint16_t(svr::datamodel::e_kernel_type::PATH);
constexpr uint32_t C_default_svrparam_lag_count = 100; // All parameters should have the same lag count because of kernel function limitations
const uint16_t C_default_svrparam_feature_quantization = std::stoul(common::C_default_feature_quantization_str);

struct t_feature_mechanics
{
    arma::u32_vec quantization; // Quantisation is per level - until computational resources allow for different quantisation per feature column
    arma::fvec stretches;
    std::deque<arma::uvec> trims;
    arma::u32_vec shifts;
    arma::fvec skips;

    bool needs_tuning() const noexcept;
};


std::ostream &operator <<(std::ostream &s, const t_feature_mechanics &fm);

class SVRParameters : public Entity
{
    bigint dataset_id = 0; /* TODO Replace with pointer to dataset id */

    std::string input_queue_table_name; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name; // TODO Replace with pointer to Input Queue
    uint16_t levels_ct = C_default_svrparam_decon_level + 1;
    uint16_t decon_level_ = C_default_svrparam_decon_level;
    uint16_t step_ = C_default_svrparam_step;
    uint16_t chunk_ix_ = C_default_svrparam_chunk_ix;
    uint16_t grad_level_ = C_default_svrparam_grad_level;
    // TODO Implement manifold projection index

    double svr_C = C_default_svrparam_svr_cost; // TODO Remove
    arma::vec epsco; // TODO Save to DB and init properly
    double svr_epsilon = C_default_svrparam_svr_epsilon;
    double svr_kernel_param = C_default_svrparam_kernel_param1;
    arma::vec gamma; // TODO Save to DB and init properly
    double svr_kernel_param2 = C_default_svrparam_kernel_param2;
    double svr_adjacent_levels_ratio = C_default_svrparam_adjacent_levels_ratio;
    std::set<uint16_t> adjacent_levels;
    e_kernel_type kernel_type = C_default_svrparam_kernel_type;
    uint32_t lag_count = C_default_svrparam_lag_count;

    t_feature_mechanics feature_mechanics; // TODO Save to DB and init properly

public:
    explicit SVRParameters() : Entity(0) {}

    SVRParameters(
            const bigint id,
            const bigint dataset_id,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const uint16_t level_ct,
            const uint16_t decon_level,
            const uint16_t step,
            const uint16_t chunk_ix = C_default_svrparam_chunk_ix,
            const uint16_t grad_level = C_default_svrparam_grad_level,
            const double svr_C = C_default_svrparam_svr_cost,
            const double svr_epsilon = C_default_svrparam_svr_epsilon,
            const double svr_kernel_param = C_default_svrparam_kernel_param1,
            const double svr_kernel_param2 = C_default_svrparam_kernel_param2,
            const uint32_t svr_decremental_distance = C_default_svrparam_decrement_distance,
            const double svr_adjacent_levels_ratio = C_default_svrparam_adjacent_levels_ratio,
            const e_kernel_type kernel_type = C_default_svrparam_kernel_type,
            const uint32_t lag_count = C_default_svrparam_lag_count,
            const std::set<uint16_t> &adjacent_levels = {});

    SVRParameters(const SVRParameters &o);

    SVRParameters &operator=(const SVRParameters &v);

    virtual void init_id() override;

    bool operator==(const SVRParameters &o) const;

    bool operator!=(const SVRParameters &o) const;

    bool operator<(const SVRParameters &o) const;

    bigint get_dataset_id() const;

    void set_dataset_id(const bigint &value);

    std::string get_input_queue_column_name() const;

    void set_input_queue_column_name(const std::string &value);

    std::string get_input_queue_table_name() const;

    void set_input_queue_table_name(const std::string &value);

    uint16_t get_level_count() const noexcept;

    void set_level_count(const uint16_t levels) noexcept;

    uint16_t get_decon_level() const noexcept;

    void set_decon_level(const uint16_t _decon_level) noexcept;

    uint16_t get_step() const noexcept;

    void set_step(const uint16_t _step) noexcept;

    uint16_t get_chunk_index() const noexcept;

    void set_chunk_index(const uint16_t _chunk_ix) noexcept;

    uint16_t get_grad_level() const noexcept;

    void set_grad_level(const uint16_t _grad_level) noexcept;

    void decrement_gradient() noexcept;

    double get_svr_C() const noexcept;

    void set_svr_C(const double _svr_C) noexcept;

    void set_epsco(const arma::vec &epsco) noexcept;

    arma::vec get_epsco() const noexcept;

    double get_svr_epsilon() const noexcept;

    void set_svr_epsilon(const double _svr_epsilon) noexcept;

    arma::vec get_gamma() const noexcept;

    arma::vec &get_gamma() noexcept;

    void set_gamma(const arma::vec &gamma) noexcept;

    double get_svr_kernel_param() const noexcept;

    void set_svr_kernel_param(const double _svr_kernel_param) noexcept;

    double get_svr_kernel_param2() const noexcept;

    void set_svr_kernel_param2(const double _svr_kernel_param2) noexcept;

    PROPERTY(double, kernel_param3, C_default_svrparam_kernel_param_tau);

    PROPERTY(uint32_t, svr_decremental_distance, C_default_svrparam_decrement_distance) // TODO Refactor all class properties to use the PROPERTY macro

    PROPERTY(double, min_Z, 0);

    PROPERTY(double, max_Z, 1);

    // Only head param (chunk 0, grad 0, manifold 0) takes effect
    double get_svr_adjacent_levels_ratio() const noexcept;

    void set_svr_adjacent_levels_ratio(const double _svr_adjacent_levels_ratio) noexcept;

    std::set<uint16_t> &get_adjacent_levels();

    const std::set<uint16_t> &get_adjacent_levels() const;

    e_kernel_type get_kernel_type() const noexcept;

    bool is_manifold() const;

    void set_kernel_type(const e_kernel_type _kernel_type) noexcept;

    // Lag count across all models should be the same with the current infrastructure inplace // Only head param (chunk 0, grad 0, manifold 0) takes effect
    uint32_t get_lag_count() const noexcept;

    void set_lag_count(const uint32_t _lag_count) noexcept;

    t_feature_mechanics &get_feature_mechanics();

    t_feature_mechanics get_feature_mechanics() const;

    void set_feature_mechanics(const t_feature_mechanics &f);

    std::string to_string() const override;

    std::string to_sql_string() const;

    bool from_sql_string(const std::string &sql_string);
};


template<typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &os, const SVRParameters &e)
{
    os << e.to_string();
    return os;
}

struct t_param_preds
{
    typedef std::array<arma::mat *, C_max_j> t_predictions, *t_predictions_ptr;

    double score = std::numeric_limits<double>::infinity();
    svr::datamodel::SVRParameters params{};
    t_predictions_ptr p_predictions = nullptr;

    static void free_predictions(const t_predictions_ptr &p_predictions_);

    void free();
};

typedef std::shared_ptr<t_param_preds> t_param_preds_ptr;


struct t_parameter_predictions_set {
    std::array<arma::mat, C_max_j> labels;
    std::array<arma::mat, C_max_j> last_knowns;
    std::array<t_param_preds, common::C_tune_keep_preds> param_pred;
};

using t_parameter_predictions_set_ptr = std::shared_ptr<t_parameter_predictions_set>;
typedef std::unordered_map<uint16_t /* level */, t_parameter_predictions_set> t_level_tuned_parameters;
using t_level_tuned_parameters_ptr = std::shared_ptr<t_level_tuned_parameters>;

}
}

