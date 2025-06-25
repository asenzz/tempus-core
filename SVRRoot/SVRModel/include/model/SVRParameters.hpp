#pragma once

#include <boost/throw_exception.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <ostream>
#include <string>
#include "common/defines.h"
#include "common/constants.hpp"
#include "common/logging.hpp"
#include "common/serialization.hpp"
#include "model/Entity.hpp"

//#define SMO_EPSILON 1e-3

namespace svr {
namespace datamodel {

class OnlineSVR;
using OnlineSVR_ptr = std::shared_ptr<OnlineSVR>;

typedef enum class kernel_type : int {
    LINEAR = 0,
    POLYNOMIAL = 1,
    RBF = 2,
    RBF_GAUSSIAN = 3,
    RBF_EXPONENTIAL = 4,
    MLP = 5,
    GA = 6,
    PATH = 7,
    DEEP_PATH = 8,
    DTW = 9,
    TFT = 10,
    GBM = 11,
    end = 12
} e_kernel_type;

e_kernel_type operator++(e_kernel_type &k_type);

e_kernel_type operator++(e_kernel_type &k_type, int);

e_kernel_type fromstring(const std::string &kernel_type_str);

std::string tostring(const e_kernel_type kt);

std::ostream &operator<<(std::ostream &os, const e_kernel_type &kt);

class SVRParameters;

using SVRParameters_ptr = std::shared_ptr<SVRParameters>;

struct less_SVRParameters_ptr
{
    bool operator()(const SVRParameters_ptr &lhs, const SVRParameters_ptr &rhs) const;
};

typedef std::set<SVRParameters_ptr, less_SVRParameters_ptr> t_param_set;

typedef std::shared_ptr<t_param_set> t_param_set_ptr; // TODO Convert to a new class

// Default SVR parameters
constexpr uint16_t C_default_svrparam_decon_level = 0;
constexpr uint16_t C_default_svrparam_step = 0;
constexpr uint16_t C_default_svrparam_chunk_ix = 0;
constexpr uint16_t C_default_svrparam_grad_level = 0;
constexpr double C_default_svrparam_svr_cost = 0;
constexpr double C_default_svrparam_svr_epsilon = 0;
constexpr double C_default_svrparam_kernel_param1 = 0;
constexpr double C_default_svrparam_kernel_param2 = 1;
constexpr double C_default_svrparam_kernel_param_tau = .75;
constexpr uint32_t C_default_svrparam_decrement_distance = common::AppConfig::C_default_kernel_length + common::AppConfig::C_default_shift_limit + common::AppConfig::C_default_outlier_slack;
constexpr double C_default_svrparam_adjacent_levels_ratio = 1;
constexpr e_kernel_type C_default_svrparam_kernel_type = e_kernel_type::GBM;
constexpr auto C_default_svrparam_kernel_type_uint = uint16_t(C_default_svrparam_kernel_type);
#ifdef VALGRIND_BUILD
constexpr uint32_t C_default_svrparam_lag_count = 2;
#else
constexpr uint32_t C_default_svrparam_lag_count = 80; // All parameters should have the same lag count because of kernel function limitations
#endif
const uint16_t C_default_svrparam_feature_quantization = std::stoul(common::C_default_feature_quantization_str);

struct t_feature_mechanics
{
    friend class boost::serialization::access;

    arma::u32_vec quantization; // Quantisation is per level - until computational resources allow for different quantisation per feature column
    arma::fvec stretches;
    std::deque<arma::uvec> trims;
    arma::u32_vec shifts;

    bool needs_tuning() const noexcept;
    std::stringstream save() const;
    static t_feature_mechanics load(const std::string &bin_data);

    template<typename S> void save(const t_feature_mechanics &feature_mechanics, S &output_stream) const
    {
        boost::archive::binary_oarchive oa(output_stream);
        oa << feature_mechanics;
    }

    template<class A> void serialize(A &ar, const unsigned version)
    {
        ar & quantization;
        ar & stretches;
        ar & trims;
        ar & shifts;
    }

    bool operator == (const t_feature_mechanics &o) const;
};
using t_feature_mechanics_ptr = std::shared_ptr<t_feature_mechanics>;
std::ostream &operator <<(std::ostream &s, const t_feature_mechanics &fm);


struct t_delannoy1_path
{
    std::vector<uint16_t> coordinates;
    uint32_t path_count = 0;
    std::vector<double> weights;
};

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

    double svr_epsilon = C_default_svrparam_svr_epsilon;
    double svr_kernel_param = C_default_svrparam_kernel_param1;
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
            const double svr_kernel_param3 = C_default_svrparam_kernel_param_tau,
            const uint32_t svr_decremental_distance = C_default_svrparam_decrement_distance,
            const double svr_adjacent_levels_ratio = C_default_svrparam_adjacent_levels_ratio,
            const e_kernel_type kernel_type = C_default_svrparam_kernel_type,
            const uint32_t lag_count = C_default_svrparam_lag_count,
            const std::set<uint16_t> &adjacent_levels = {},
            const t_feature_mechanics &feature_mechanics = {});

    SVRParameters(const SVRParameters &o);

    SVRParameters &operator=(const SVRParameters &o);

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

    double get_svr_epsilon() const noexcept;

    void set_svr_epsilon(const double _svr_epsilon) noexcept;

    double get_svr_kernel_param() const noexcept;

    void set_svr_kernel_param(const double _svr_kernel_param) noexcept;

    double get_svr_kernel_param2() const noexcept;

    void set_svr_kernel_param2(const double _svr_kernel_param2) noexcept;

    PROPERTY(double, svr_C, C_default_svrparam_svr_cost);

    PROPERTY(double, kernel_param3, C_default_svrparam_kernel_param_tau);

    PROPERTY(uint32_t, svr_decremental_distance, C_default_svrparam_decrement_distance) // TODO Refactor all class properties to use the PROPERTY macro

    PROPERTY(double, min_Z, 0)

    PROPERTY(double, max_Z, 1)

    PROPERTY(float, H_feedback, .3);

    PROPERTY(float, D_feedback, .34);

    PROPERTY(float, V_feedback, .3);

    // Only head param (chunk 0, grad 0, manifold 0) takes effect
    double get_svr_adjacent_levels_ratio() const noexcept;

    void set_svr_adjacent_levels_ratio(const double _svr_adjacent_levels_ratio) noexcept;

    std::set<uint16_t> &get_adjacent_levels();

    const std::set<uint16_t> &get_adjacent_levels() const;

    e_kernel_type get_kernel_type() const noexcept;

    bool is_manifold() const;

    void set_kernel_type(const e_kernel_type _kernel_type) noexcept;

    PROPERTY(OnlineSVR_ptr, manifold);

    // Lag count across all models should be the same with the current infrastructure inplace // Only head param (chunk 0, grad 0, manifold 0) takes effect
    uint32_t get_lag_count() const noexcept;

    void set_lag_count(const uint32_t _lag_count) noexcept;

    t_feature_mechanics &get_feature_mechanics();

    t_feature_mechanics get_feature_mechanics() const;

    void set_feature_mechanics(const t_feature_mechanics &f);

    std::string to_string() const override;

    std::string to_sql_string() const;

    bool from_sql_string(const std::string &sql_string);

    PROPERTY(std::string, tft_model)

    PROPERTY(uint32_t, tft_n_classes, 0)
};

std::ostream &operator<<(std::ostream &os, const SVRParameters &e);

}
}

