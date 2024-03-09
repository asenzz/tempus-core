#pragma once

#include <ostream>
#include <string>
#include "common/defines.h"
#include "common/constants.hpp"
#include "common/Logging.hpp"
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
    DEEP_PATH2 = 9,
    number_of_kernel_types = 10 // end of enum = invalid type
} kernel_type_e, *kernel_type_e_ptr;

template<typename ST>
ST tostring(const datamodel::kernel_type_e kt)
{
    switch (kt) {
        case kernel_type_e::LINEAR:
            return "LINEAR";
        case kernel_type_e::POLYNOMIAL:
            return "POLYNOMIAL";
        case kernel_type_e::RBF:
            return "RBF";
        case kernel_type_e::RBF_GAUSSIAN:
            return "RBF_GAUSSIAN";
        case kernel_type_e::RBF_EXPONENTIAL:
            return "RBF_EXPONENTIAL";
        case kernel_type_e::GA:
            return "GA";
        case kernel_type_e::PATH:
            return "PATH";
        case kernel_type_e::DEEP_PATH:
            return "DEEP_PATH";
        case kernel_type_e::DEEP_PATH2:
            return "DEEP_PATH2";
        default:
            return "UNKNOWN";
    }
}

kernel_type_e operator++(kernel_type_e &k_type);

kernel_type_e operator++(kernel_type_e &k_type, int);

class SVRParameters;

using SVRParameters_ptr = std::shared_ptr<datamodel::SVRParameters>;

struct less_SVRParameters_ptr
{
    bool operator()(const datamodel::SVRParameters_ptr &lhs, const datamodel::SVRParameters_ptr &rhs) const;
};

typedef std::set<datamodel::SVRParameters_ptr, less_SVRParameters_ptr> t_param_set;
typedef std::shared_ptr<t_param_set> t_param_set_ptr; // TODO Convert to a new class

enum class bound_type : int
{
    min = 0,
    max = 1
};

typedef std::tuple<
        std::string, /* input_queue_table_name; */
        std::string, /* input_queue_column_name; */
        size_t /* decon_level */,
        size_t /* chunk ix */,
        size_t /* grad level */> svr_parameters_index_t;

class SVRParameters : public Entity
{
private:
    bigint dataset_id = 0; /* TODO Replace with pointer to dataset id */

    std::string input_queue_table_name; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name; // TODO Replace with pointer to Input Queue
    size_t decon_level_ = DEFAULT_SVRPARAM_DECON_LEVEL;
    size_t chunk_ix_ = DEFAULT_SVRPARAM_CHUNK_IX;
    size_t grad_level_ = DEFAULT_SVRPARAM_GRAD_LEVEL;

    double svr_C = DEFAULT_SVRPARAM_SVR_COST;
    double svr_epsilon = DEFAULT_SVRPARAM_SVR_EPSILON;
    double svr_kernel_param = DEFAULT_SVRPARAM_KERNEL_PARAM_1;
    double svr_kernel_param2 = DEFAULT_SVRPARAM_KERNEL_PARAM_2;
    u_int64_t svr_decremental_distance = DEFAULT_SVRPARAM_DECREMENT_DISTANCE;
    double svr_adjacent_levels_ratio = DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO;
    kernel_type_e kernel_type = DEFAULT_SVRPARAM_KERNEL_TYPE;
    size_t lag_count = DEFAULT_SVRPARAM_LAG_COUNT;

public:
    explicit SVRParameters() : Entity(0) {}

    SVRParameters(
            const bigint id,
            const bigint dataset_id,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name,
            const size_t decon_level,
            const size_t chunk_ix = DEFAULT_SVRPARAM_CHUNK_IX,
            const size_t grad_level = DEFAULT_SVRPARAM_GRAD_LEVEL,
            const double svr_C = DEFAULT_SVRPARAM_SVR_COST,
            const double svr_epsilon = DEFAULT_SVRPARAM_SVR_EPSILON,
            const double svr_kernel_param = DEFAULT_SVRPARAM_KERNEL_PARAM_1,
            const double svr_kernel_param2 = DEFAULT_SVRPARAM_KERNEL_PARAM_2,
            const u_int64_t svr_decremental_distance = DEFAULT_SVRPARAM_DECREMENT_DISTANCE,
            const double svr_adjacent_levels_ratio = DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO,
            const kernel_type_e kernel_type = DEFAULT_SVRPARAM_KERNEL_TYPE,
            const size_t lag_count = DEFAULT_SVRPARAM_LAG_COUNT);

    SVRParameters(const SVRParameters &params);

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

    size_t get_decon_level() const;

    void set_decon_level(const size_t _decon_level);

    size_t get_chunk_ix() const;

    void set_chunk_ix(const size_t _chunk_ix);

    size_t get_grad_level() const;

    void set_grad_level(const size_t _grad_level);

    void decrement_gradient();

    double get_svr_C() const;

    void set_svr_C(const double _svr_C);

    double get_svr_epsilon() const;

    void set_svr_epsilon(const double _svr_epsilon);

    double get_svr_kernel_param() const;

    void set_svr_kernel_param(const double _svr_kernel_param);

    double get_svr_kernel_param2() const;

    void set_svr_kernel_param2(const double _svr_kernel_param2);

    // Only head param (chunk 0, grad 0, manifold 0) takes effect
    u_int64_t get_svr_decremental_distance() const;

    void set_svr_decremental_distance(const uint64_t _svr_decremental_distance);

    // Only head param (chunk 0, grad 0, manifold 0) takes effect
    double get_svr_adjacent_levels_ratio() const;

    void set_svr_adjacent_levels_ratio(const double _svr_adjacent_levels_ratio);

    kernel_type_e get_kernel_type() const;

    bool is_manifold() const;

    void set_kernel_type(const kernel_type_e _kernel_type);

    // Lag count across all models should be the same with the current infrastructure inplace // Only head param (chunk 0, grad 0, manifold 0) takes effect
    size_t get_lag_count() const;

    void set_lag_count(const size_t _lag_count);

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
            score(score), p_params(params), p_predictions(predictions), p_labels(labels), p_last_knowns(last_knowns)
    {}
};

typedef std::shared_ptr<t_param_preds> t_param_preds_ptr;

struct param_preds_cmp
{
    bool operator()(const t_param_preds_ptr &lhs, const t_param_preds_ptr &rhs) const
    {
        return lhs->score < rhs->score;
    }
};
typedef std::set<t_param_preds_ptr, param_preds_cmp> t_parameter_predictions_set;
using t_parameter_predictions_set_ptr = std::shared_ptr<t_parameter_predictions_set>;
typedef std::unordered_map<std::tuple<size_t /* level */, size_t /* grad */, size_t /* chunk */>, t_parameter_predictions_set_ptr> t_tuned_parameters;
using t_tuned_parameters_ptr = std::shared_ptr<t_tuned_parameters>;

}
}

