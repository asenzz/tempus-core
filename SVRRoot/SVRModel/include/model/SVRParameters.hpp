#pragma once

#include "common/types.hpp"
#include "model/Entity.hpp"
#include "common/Logging.hpp"
#include "util/string_utils.hpp"
#include "common/constants.hpp"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <utility>

//#define SMO_EPSILON 1e-3

namespace svr {
namespace datamodel {

#define DEFAULT_SVR_HYPERPARAMS \
        bigint(0),\
        bigint(0),\
        "",\
        "",\
        DEFAULT_SVRPARAM_DECON_LEVEL,\
        DEFAULT_SVRPARAM_CHUNK_IX,\
        DEFAULT_SVRPARAM_GRAD_LEVEL,\
        DEFAULT_SVRPARAM_SVR_COST,\
        DEFAULT_SVRPARAM_SVR_EPSILON,\
        DEFAULT_SVRPARAM_KERNEL_PARAM_1,\
        DEFAULT_SVRPARAM_KERNEL_PARAM_2,\
        DEFAULT_SVRPARAM_DECREMENT_DISTANCE,\
        DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO,\
        DEFAULT_SVRPARAM_KERNEL_TYPE,\
        DEFAULT_SVRPARAM_LAG_COUNT

#define DEFAULT_APP_HYPERPARAMS(AP) \
        (AP).get_online_iters_limit_mult(),\
        (AP).get_online_learn_iter_limit(),\
        (AP).get_max_variations(),\
        (AP).get_smo_epsilon_divisor(),\
        (AP).get_smo_cost_divisor(),\
        (AP).get_stabilize_iterations_count(),\
        (AP).get_default_number_variations(),\
        (AP).get_dont_update_r_matrix(),\
        (AP).get_max_smo_iterations()

#define DEFAULT_ALL_HYPERPARAMS(ap) DEFAULT_SVR_HYPERPARAMS, DEFAULT_APP_HYPERPARAMS(ap)

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
        case kernel_type_e::LINEAR: return "LINEAR";
        case kernel_type_e::POLYNOMIAL: return "POLYNOMIAL";
        case kernel_type_e::RBF: return "RBF";
        case kernel_type_e::RBF_GAUSSIAN: return "RBF_GAUSSIAN";
        case kernel_type_e::RBF_EXPONENTIAL: return "RBF_EXPONENTIAL";
        case kernel_type_e::GA: return "GA";
        case kernel_type_e::PATH: return "PATH";
        case kernel_type_e::DEEP_PATH: return "DEEP_PATH";
        case kernel_type_e::DEEP_PATH2: return "DEEP_PATH2";
        default: return "UNKNOWN";
    }
}

inline kernel_type_e operator++(kernel_type_e &k_type)
{
    int tmp = static_cast<int>(k_type);
    tmp++;
    return k_type = static_cast<kernel_type_e >( tmp);
}

inline kernel_type_e operator++(kernel_type_e &k_type, int)
{
    kernel_type_e tmp(k_type);
    ++k_type;
    return tmp;
}

enum class bound_type : int
{
    min = 0,
    max = 1
};

typedef std::tuple<
        std::string, /* input_queue_table_name; */
        std::string, /* input_queue_column_name; */
        size_t /* decon_level */> svr_parameters_index_t;

class SVRParameters : public Entity
{
private:
    bigint dataset_id; /* TODO Replace with pointer to dataset id */

    std::string input_queue_table_name; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name; // TODO Replace with pointer to Input Queue
    size_t decon_level_;
    size_t chunk_ix_;
    size_t grad_level_;

    double svr_C;
    double svr_epsilon;
    double svr_kernel_param;
    double svr_kernel_param2;
    u_int64_t svr_decremental_distance;
    double svr_adjacent_levels_ratio;
    kernel_type_e kernel_type;
    size_t lag_count;
    std::vector<size_t> sub_vector_indices;

public:
    size_t online_iters_limit_mult;
    size_t online_learn_iter_limit;
    size_t max_variations;
    double smo_epsilon_divisor;
    double smo_cost_divisor;
    size_t stabilize_iterations_count;
    ssize_t number_variations;
    bool dont_update_r_matrix;
    size_t max_iter;
    bool skip = false;

public:
    SVRParameters() : SVRParameters(
            DEFAULT_SVR_HYPERPARAMS,
            common::__online_iters_limit_mult, common::__online_learn_iter_limit,
            common::__max_variations, common::__smo_epsilon_divisor, common::__smo_cost_divisor,
            common::__stabilize_iterations_count, common::__default_number_variations,
            common::__dont_update_r_matrix, common::__max_iter)
    {}

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
            const size_t lag_count = DEFAULT_SVRPARAM_LAG_COUNT,

            // Global parameters needed by SVR algorithms
            const size_t online_iters_limit_mult = common::__online_iters_limit_mult,
            const size_t online_learn_iter_limit = common::__online_learn_iter_limit,
            const size_t max_variations = common::__max_variations,
            const double smo_epsilon_divisor = common::__smo_epsilon_divisor,
            const double smo_cost_divisor = common::__smo_cost_divisor,
            const size_t stabilize_iterations_count = common::__stabilize_iterations_count,
            const ssize_t number_variations = common::__default_number_variations,
            const bool dont_update_r_matrix = common::__dont_update_r_matrix,
            const size_t max_iter = common::__max_iter)
            : Entity(id),
              dataset_id(dataset_id),
              input_queue_table_name(input_queue_table_name),
              input_queue_column_name(input_queue_column_name),
              decon_level_(decon_level),
              chunk_ix_(chunk_ix),
              grad_level_(grad_level),
              svr_C(svr_C),
              svr_epsilon(svr_epsilon),
              svr_kernel_param(svr_kernel_param),
              svr_kernel_param2(svr_kernel_param2),
              svr_decremental_distance(svr_decremental_distance),
              svr_adjacent_levels_ratio(svr_adjacent_levels_ratio),
              kernel_type(kernel_type),
              lag_count(lag_count),

            // Globals
              online_iters_limit_mult(online_iters_limit_mult),
              online_learn_iter_limit(online_learn_iter_limit),
              max_variations(max_variations),
              smo_epsilon_divisor(smo_epsilon_divisor),
              smo_cost_divisor(smo_cost_divisor),
              stabilize_iterations_count(stabilize_iterations_count),
              number_variations(number_variations),
              dont_update_r_matrix(dont_update_r_matrix),
              max_iter(max_iter)
    {
        set_kernel_type(kernel_type);
    }

    SVRParameters(const SVRParameters &params) :
            SVRParameters(
                    bigint(0),
                    params.get_dataset_id(),
                    params.get_input_queue_table_name(),
                    params.get_input_queue_column_name(),
                    params.get_decon_level(),
                    params.get_chunk_ix(),
                    params.get_grad_level(),
                    params.get_svr_C(),
                    params.get_svr_epsilon(),
                    params.get_svr_kernel_param(),
                    params.get_svr_kernel_param2(),
                    params.get_svr_decremental_distance(),
                    params.get_svr_adjacent_levels_ratio(),
                    params.get_kernel_type(),
                    params.get_lag_count(),
                    params.online_iters_limit_mult, params.online_learn_iter_limit, params.max_variations,
                    params.smo_epsilon_divisor, params.smo_cost_divisor, params.stabilize_iterations_count,
                    params.number_variations, params.dont_update_r_matrix, params.max_iter)
    {
        set_kernel_type(kernel_type);
    }

    SVRParameters &operator =(const SVRParameters &v)
    {
        set_id(v.get_id());
        dataset_id = v.dataset_id;
        input_queue_table_name = v.input_queue_table_name;
        input_queue_column_name = v.input_queue_column_name;
        decon_level_ = v.decon_level_;
        chunk_ix_ = v.chunk_ix_;
        grad_level_ = v.grad_level_;
        svr_C = v.svr_C;
        svr_epsilon = v.svr_epsilon;
        svr_kernel_param = v.svr_kernel_param;
        svr_kernel_param2 = v.svr_kernel_param2;
        svr_decremental_distance = v.svr_decremental_distance;
        svr_adjacent_levels_ratio = v.svr_adjacent_levels_ratio;
        kernel_type = v.kernel_type;
        lag_count = v.lag_count;
        // Globals
        online_iters_limit_mult = v.online_iters_limit_mult;
        online_learn_iter_limit = v.online_learn_iter_limit;
        max_variations = v.max_variations;
        smo_epsilon_divisor = v.smo_epsilon_divisor;
        smo_cost_divisor = v.smo_cost_divisor;
        stabilize_iterations_count = v.stabilize_iterations_count;
        number_variations = v.number_variations;
        dont_update_r_matrix = v.dont_update_r_matrix;
        max_iter = v.max_iter;
        return *this;
    }

    bool operator==(const SVRParameters &other) const
    {
        return get_dataset_id() == other.get_dataset_id()
               && get_input_queue_table_name() == other.get_input_queue_table_name()
               && get_input_queue_column_name() == other.get_input_queue_column_name()
               && get_decon_level() == other.get_decon_level()
               && get_chunk_ix() == other.get_chunk_ix()
               && get_grad_level() == other.get_grad_level()
               && get_svr_C() == other.get_svr_C()
               && get_svr_epsilon() == other.get_svr_epsilon()
               && get_svr_kernel_param() == other.get_svr_kernel_param()
               && get_svr_kernel_param2() == other.get_svr_kernel_param2()
               && get_svr_decremental_distance() == other.get_svr_decremental_distance()
               && get_svr_adjacent_levels_ratio() == other.get_svr_adjacent_levels_ratio()
               && get_kernel_type() == other.get_kernel_type()
               && get_lag_count() == other.get_lag_count();
    }

    bool operator!=(const SVRParameters &other) const
    {
        return !operator==(other);
    }

    bigint get_dataset_id() const
    {
        return dataset_id;
    }

    void set_dataset_id(const bigint &value)
    {
        dataset_id = value;
    }

    std::string get_input_queue_column_name() const
    {
        return input_queue_column_name;
    }

    void set_input_queue_column_name(const std::string &value)
    {
        input_queue_column_name = value;
    }

    std::string get_input_queue_table_name() const
    {
        return input_queue_table_name;
    }

    void set_input_queue_table_name(const std::string &value)
    {
        input_queue_table_name = value;
    }

    size_t get_decon_level() const
    {
        return decon_level_;
    }

    void set_decon_level(const size_t _decon_level)
    {
        this->decon_level_ = _decon_level;
    }

    size_t get_chunk_ix() const
    {
        return chunk_ix_;
    }

    void set_chunk_ix(const size_t _chunk_ix)
    {
        this->chunk_ix_ = _chunk_ix;
    }

    size_t get_grad_level() const
    {
        return grad_level_;
    }

    void set_grad_level(const size_t _grad_level)
    {
        grad_level_ = _grad_level;
    }

    void decrement_gradient()
    {
        if (grad_level_) --grad_level_;
    }

    void set_skip(const bool _skip)
    {
        skip = _skip;
    }

    bool get_skip() const
    {
        return skip;
    }

    double get_svr_C() const
    {
        return svr_C;
    }

    void set_svr_C(const double _svr_C)
    {
        // if (_svr_C < 0) THROW_EX_FS(std::invalid_argument, "Applied cost parameter " << _svr_C << " is less than zero.");
        svr_C = _svr_C;
    }

    double get_svr_epsilon() const
    {
        return svr_epsilon;
    }


    void set_svr_epsilon(const double _svr_epsilon)
    {
        if (_svr_epsilon < 0) THROW_EX_FS(std::invalid_argument, "Epsilon parameter " << _svr_epsilon << " is less than zero.");
        svr_epsilon = _svr_epsilon;
    }


    double get_svr_kernel_param() const
    {
        return svr_kernel_param;
    }

    void set_svr_kernel_param(const double _svr_kernel_param)
    {
        svr_kernel_param = _svr_kernel_param;
    }

    double get_svr_kernel_param2() const
    {
        return svr_kernel_param2;
    }

    void set_svr_kernel_param2(const double _svr_kernel_param2)
    {
        svr_kernel_param2 = _svr_kernel_param2;
    }


    u_int64_t get_svr_decremental_distance() const
    {
        return svr_decremental_distance;
    }


    void set_svr_decremental_distance(const uint64_t _svr_decremental_distance)
    {
        svr_decremental_distance = _svr_decremental_distance;
    }


    double get_svr_adjacent_levels_ratio() const
    {
        return svr_adjacent_levels_ratio;
    }


    void set_svr_adjacent_levels_ratio(const double _svr_adjacent_levels_ratio)
    {
        if (_svr_adjacent_levels_ratio < 0. || _svr_adjacent_levels_ratio > 1.)
            THROW_EX_FS(std::range_error, "Adjacent levels ratio " << _svr_adjacent_levels_ratio << " is out of 0..1 range.");
        svr_adjacent_levels_ratio = _svr_adjacent_levels_ratio;
    }


    kernel_type_e get_kernel_type() const
    {
        return kernel_type;
    }

    bool is_manifold() const
    {
        return kernel_type == kernel_type_e::DEEP_PATH || kernel_type == kernel_type_e::DEEP_PATH2;
    }

    void set_kernel_type(const kernel_type_e _kernel_type)
    {
        if (_kernel_type < kernel_type_e::number_of_kernel_types)
            kernel_type = _kernel_type;
        else
            THROW_EX_FS(std::invalid_argument, "Wrong kernel type " << (ssize_t) _kernel_type);
    }


    size_t get_lag_count() const
    {
        return lag_count;
    }

    void set_lag_count(const size_t _lag_count)
    {
        if (_lag_count == 0) THROW_EX_FS(std::invalid_argument, "Lag count parameter is zero.");
        lag_count = _lag_count;
    }


    std::string to_string() const override
    {
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<double>::max_digits10);
        ss << "id " << id
           << ", dataset id " << dataset_id
           << ", cost " << svr_C
           << ", epsilon " << svr_epsilon
           << ", kernel param " << svr_kernel_param
           << ", kernel param 2 " << svr_kernel_param2
           << ", decrement distance " << svr_decremental_distance
           << ", svr adjacent levels ratio " << svr_adjacent_levels_ratio
           << ", kernel type " << static_cast<int>(kernel_type)
           << ", lag count " << lag_count
           << ", table name " << input_queue_table_name
           << ", column name " << input_queue_column_name
           << ", decon level " << decon_level_
           << ", chunk " << chunk_ix_
           << ", gradient " << grad_level_;

        return ss.str();
    }

    std::string to_sql_string() const
    {
        std::stringstream ss;
        ss.precision(std::numeric_limits<double>::max_digits10);

        ss << "\t" << get_id()
           << "\t" << get_dataset_id()
           << "\t" << input_queue_table_name
           << "\t" << input_queue_column_name
           << "\t" << decon_level_
           << "\t" << chunk_ix_
           << "\t" << grad_level_
           << "\t" << svr_C
           << "\t" << svr_epsilon
           << "\t" << svr_kernel_param
           << "\t" << svr_kernel_param2
           << "\t" << svr_decremental_distance
           << "\t" << svr_adjacent_levels_ratio
           << "\t" << static_cast<int>(kernel_type)
           << "\t" << lag_count;

        return ss.str();
    }

    bool from_sql_string(const std::string &sql_string)
    {
        std::vector<std::string> tokens;
        svr::common::split(sql_string, '\t', tokens);

        if (tokens.size() != 13) {
            LOG4_ERROR("Incorrect number of tokens " << tokens.size());
            return false;
        }
        set_id(atoll(tokens[0].c_str()));
        set_dataset_id(atoll(tokens[1].c_str()));
        set_input_queue_table_name(tokens[2]);
        set_input_queue_column_name(tokens[3]);
        set_decon_level(atoll(tokens[4].c_str()));
        set_chunk_ix(atoll(tokens[5].c_str()));
        set_grad_level(atoll(tokens[6].c_str()));
        set_svr_C(atof(tokens[7].c_str()));
        set_svr_epsilon(atof(tokens[8].c_str()));
        set_svr_kernel_param(atof(tokens[9].c_str()));
        set_svr_kernel_param2(atof(tokens[10].c_str()));
        set_svr_decremental_distance(atoll(tokens[11].c_str()));
        set_svr_adjacent_levels_ratio(atof(tokens[12].c_str()));
        set_kernel_type(kernel_type_e(atoll(tokens[13].c_str())));
        set_lag_count(atoll(tokens[14].c_str()));

        LOG4_DEBUG("Successfully loaded parameters " << to_sql_string());

        return true;
    }

    std::string to_options_string(const size_t model_number) const
    {
        std::stringstream ss;

        ss << "\"svr_c_" << model_number << "\":\"" << get_svr_C() << "\","
           << "\"svr_epsilon_" << model_number << "\":\"" << get_svr_epsilon() << "\","
           << "\"svr_kernel_param_" << model_number << "\":\"" << get_svr_kernel_param() << "\","
           << "\"svr_kernel_param2_" << model_number << "\":\"" << get_svr_kernel_param2() << "\","
           << "\"svr_decremental_distance_" << model_number << "\":\"" << get_svr_decremental_distance() << "\","
           << "\"svr_svr_adjacent_levels_ratio_" << model_number << "\":\"" << get_svr_adjacent_levels_ratio() << "\","
           << "\"svr_kernel_type_" << model_number << "\":\"" << static_cast<int>(get_kernel_type()) << "\","
           << "\"svr_error_tolerance_" << model_number << "\":\"" << "tune" << "\","
           << "\"lag_count_" << model_number << "\":\"" << get_lag_count() << "\"";

        return ss.str();
    }


    std::vector<double> to_vector() const
    {
        // TODO
        return std::vector<double>();
    }

    const std::vector<size_t> &get_sub_vector_indices() const
    {
        return sub_vector_indices;
    }

    void set_sub_vector_indices(const std::vector<size_t> &new_sub_vector_indices)
    {
        sub_vector_indices = new_sub_vector_indices;
    }
};

struct tune_svr_parameters
{
    bool svr_C{false};
    bool svr_epsilon{false};
    bool svr_kernel_param{false};
    bool svr_kernel_param2{false};
    bool svr_decremental_distance{false};
    bool svr_adjacent_levels_ratio{false};
    bool lag_count{false};
};

struct Bounds
{
    datamodel::SVRParameters min_bounds;
    datamodel::SVRParameters max_bounds;
    tune_svr_parameters is_tuned;

    std::vector<double> to_vector(const bound_type &bt) const
    {
        const datamodel::SVRParameters *current_parameters;
        switch (bt) {
            case bound_type::min :
                current_parameters = &min_bounds;
                break;
            case bound_type::max :
                current_parameters = &max_bounds;
                break;
        }

        std::vector<double> result;
        result.push_back(current_parameters->get_svr_C());
        result.push_back(current_parameters->get_svr_epsilon());
        result.push_back(current_parameters->get_svr_kernel_param());
        result.push_back(current_parameters->get_svr_kernel_param2());
        result.push_back(current_parameters->get_svr_decremental_distance());
        result.push_back(current_parameters->get_svr_adjacent_levels_ratio());
        result.push_back(static_cast<double>(current_parameters->get_lag_count()));

        return result;
    }
};

using SVRParameters_ptr = std::shared_ptr<SVRParameters>;
using ensemble_svr_parameters_t = std::map<
/* input queue table name */ std::pair<std::string,
/* input queue column name */ std::string>,
/* svr parameters */ std::vector<SVRParameters_ptr> >;

}
}

