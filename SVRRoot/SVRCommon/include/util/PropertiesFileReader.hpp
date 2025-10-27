#pragma once

#include <chrono>
#include <string>
#include <boost/log/trivial.hpp>
#include <oneapi/tbb/mutex.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "common/types.hpp"
#include "string_utils.hpp"

namespace svr {
namespace common {
enum class ConcreteDaoType : uint8_t
{
    PgDao,
    AsyncDao
};

#define CONFPROP(T, X, D)                                               \
private:                                                                \
    static constexpr std::string X##_name = #X;                         \
    const std::string X = common::ctoupper<X##_name.size()>(X##_name.data()); \
    T X##_;                                                             \
    bool X##_set_ = false;                                              \
public:                                                                 \
    static constexpr T C_default_##X = D;                               \
    static constexpr std::string C_default_str_##X = #D;                \
    inline T get_##X() {                                                \
        if (X##_set_ == false) {                                        \
           const tbb::mutex::scoped_lock lk(set_mx);                    \
           if (X##_set_ == false) {                                     \
                X##_ = get_property<T>(config_file, X, #D);             \
                X##_set_ = true;                                        \
           }                                                            \
        }                                                               \
        return X##_;                                                    \
    }

#define CONFPRO_(x, D) CONFPROP(DTYPE(D), x, D)

class PropertiesReader
{
    static constexpr char SQL_PROPERTIES_DIR_KEY[] = "SQL_PROPERTIES_DIR";
    static constexpr char COMMENT_CHARS[] = "#";

    tbb::mutex load_mx;
    MessageProperties property_files;
    const char delimiter;
    std::string property_files_location;

    size_t read_property_file(const std::string &property_file_name);

    static bool is_comment(const std::string &line);

    static bool is_multiline(const std::string &line);

    const std::string &get_property_value(const std::string &property_file, const std::string &key, const std::string &default_value);

protected:
    tbb::mutex set_mx;
    const std::string config_file;

public:
    PropertiesReader(char delimiter, const std::string &config_file);

    const MessageProperties::mapped_type &read_properties(const std::string &property_file);

    template<typename T> inline T get_property(const std::string &property_file, const std::string &key, const std::string &default_value = "")
    {
        try {
            return boost::lexical_cast<T>(get_property_value(property_file, key, default_value));
        } catch (const std::exception &ex) {
            // LOG4_ERROR("Error getting property " << key << " from file " << property_file << ", " << ex.what() << ", default value " << default_value);
            return {};
        };
    }
};

class AppConfig final : public PropertiesReader
{
    CONFPROP(float, instance_inert, 0)

    CONFPROP(float, min_step, .1)

    CONFPROP(float, max_step, .9)

    CONFPROP(uint32_t, min_quant, 5)

    CONFPROP(uint32_t, max_quant, 100)

    CONFPROP(uint32_t, tune_data_iter, 10)

    CONFPROP(uint32_t, tune_data_pop, 16)

    CONFPROP(uint32_t, gpu_chunk, 18000)

    CONFPROP(float, solve_radius, .5)

    CONFPROP(float, predict_focus, .25)

    CONFPROP(float, limes, 1)

    CONFPROP(float, nn_head_coef, .1)

    CONFPROP(float, nn_hide_coef, 2)

    CONFPROP(float, k_learn_rate, 1e-3)

    CONFPROP(uint16_t, k_epochs, 100)

    CONFPROP(uint16_t, opt_depth, 1)

    CONFPROP(uint32_t, hdbs_points, 10)

    CONFPROP(float, tune_max_fback, 1)

    CONFPROP(float, weight_inertia, 0)

    CONFPROP(uint16_t, solve_particles, 80)

    CONFPROP(bool, combine_queues, 0)

    CONFPROP(uint16_t, paral_ensembles, 1)

    CONFPROP(uint16_t, parallel_models, 1)

    CONFPROP(uint16_t, parallel_chunks, 2)

    CONFPROP(bool, xresidual, true)

    CONFPROP(uint32_t, oemd_interleave, 2)

    CONFPROP(float, weights_exp, 1)

    CONFPROP(float, weights_slope, 10)

    CONFPROP(float, stretch_limit, .5)

    CONFPROP(float, stretch_coef, .999)

    CONFPROP(uint32_t, shift_limit, 100)

    CONFPROP(uint32_t, shift_multi, 0) // Shift increment multiplier for align_features()

    CONFPROP(uint32_t, weight_ileave, 10)

    CONFPROP(uint32_t, outlier_slack, 0)

    CONFPROP(uint32_t, kernel_length, 10000)

    CONFPROP(uint32_t, predict_chunks, 100)

    CONFPROP(double, lag_multiplier, 100)

    CONFPROP(uint16_t, interleave, 10)

    CONFPROP(double, tune_max_lambda, 50)

    CONFPROP(double, tune_max_tau, 2)

    CONFPROP(uint16_t, db_num_threads, 8)

    CONFPROP(uint32_t, tune_skip, 0)

    CONFPROP(uint32_t, align_window, 1000) // should be half of average of the decrement distance used across models

    CONFPRO_(max_loop_count, -1) // -1 means no limit

    CONFPROP(uint16_t, weight_layers, 1)

    CONFPROP(uint16_t, db_retries, 10)

    CONFPROP(uint32_t, db_wait, 100)

    CONFPROP(uint16_t, tune_particles1, 10)

    CONFPROP(uint16_t, tune_particles2, 10)

    CONFPROP(uint16_t, tune_iteration1, 10)

    CONFPROP(uint16_t, tune_iteration2, 10)

    CONFPROP(float, chunk_overlap, .0) // 0..1, higher means more chunks

    CONFPROP(float, oemd_skipdiv, 1) // (0..num quantisations], higher means more refined

    CONFPROP(float, oemd_xcor_weig, 1) // OEMD tuning labels to features correlation weight in validation score

    CONFPROP(float, oemd_acor_weig, 1) // OEMD tuning labels autocorrelation weight in validation score

    CONFPROP(float, oemd_rel_pow_w, 1) // OEMD tuning relative power of output signal in validation score weight

    CONFPROP(float, oemd_entweight, 0) // OEMD tuning entropy (complexity) weight in validation score

    CONFPROP(float, oemd_freq_ceil, .25) // OEMD tuning frequency ceiling, [0..0.5]

    CONFPROP(float, label_drift, .5)

    CONFPROP(uint32_t, online_irwls, 8)

private: // TODO port properties below to use the CONFPROP macro
    static constexpr char LOOP_INTERVAL[] = "LOOP_INTERVAL_MS";
    static constexpr char STREAM_LOOP_INTERVAL[] = "STREAM_LOOP_INTERVAL_MS";
    static constexpr char DAEMONIZE[] = "DAEMONIZE";
    static constexpr char FEATURE_QUANTIZATION[] = "FEATURE_QUANTIZATION";
    static constexpr char PREDICTION_HORIZON[] = "PREDICTION_HORIZON";
    static constexpr char TUNE_PARAMETERS[] = "TUNE_PARAMETERS";
    static constexpr char RECOMBINE_PARAMETERS[] = "RECOMBINE_PARAMETERS";
    static constexpr char LOG_LEVEL_KEY[] = "LOG_LEVEL";
    static constexpr char DAO_TYPE_KEY[] = "DAO_TYPE";
    static constexpr char SET_THREAD_AFFINITY[] = "SET_THREAD_AFFINITY";
    static constexpr char STEPS[] = "STEPS";
    static constexpr char OUTPUTS[] = "OUTPUTS";
    static constexpr char ONLINE_LEARN_ITER_LIMIT[] = "ONLINE_LEARN_ITER_LIMIT";
    static constexpr char STABILIZE_ITERATIONS_COUNT[] = "STABILIZE_ITERATIONS_COUNT";
    static constexpr char SCALING_ALPHA[] = "SCALING_ALPHA";
    static constexpr char CONNECTION_STRING[] = "CONNECTION_STRING";
    static constexpr char SLIDE_COUNT[] = "SLIDE_COUNT";
    static constexpr char SLIDE_SKIP[] = "SLIDE_SKIP";
    static constexpr char TUNE_RUN_LIMIT[] = "TUNE_RUN_LIMIT";
    static constexpr char SELF_REQUEST[] = "SELF_REQUEST";
    static constexpr char NUM_QUANTISATIONS[] = "NUM_QUANTISATIONS"; // Higher number of quantisations means more precision (more resource usage)
    static constexpr char QUANTISATION_DIVISOR[] = "QUANTISATION_DIVISOR"; // Lower divisor means fine grained quantisations (more resource usage, 1 quant increment until 2 * divisor)
    static constexpr char OEMD_TUNE_PARTICLES[] = "OEMD_TUNE_PARTICLES"; // Number of particles for tuning, higher means more precision
    static constexpr char OEMD_TUNE_ITERATIONS[] = "OEMD_TUNE_ITERATIONS"; // Number of iterations for tuning, higher means more precision
    static constexpr char SOLVE_ITERATIONS_COEFFICIENT[] = "SOLVE_ITERATIONS_COEFFICIENT"; // Coefficient for iterations in solving, higher means more precision, max recommended 2
    static constexpr char OEMD_MASK_DIR[] = "OEMD_MASK_DIR"; // Directory for OEMD masks

    const ConcreteDaoType dao_type;
    const size_t slide_count_, slide_skip_, tune_run_limit_, feature_quantization_, steps, outputs, online_learn_iter_limit_, stabilize_iterations_count_;
    const float prediction_horizon_, scaling_alpha_, solve_iterations_coefficient_;
    const std::string db_connection_string_, oemd_masks_dir_, db_file_;
#ifdef USE_DUCKDB
    const bool is_duck_;
#endif
    const bool set_thread_affinity_, recombine_parameters_, tune_parameters_, self_request_, daemonize_;
    const boost::log::trivial::severity_level log_level_;
    const std::chrono::milliseconds loop_interval_, stream_loop_interval_;
    const uint16_t num_quantisations_, quantisation_divisor_, oemd_tune_particles_, oemd_tune_iterations_;

public:
    virtual ~AppConfig();

    explicit AppConfig(const std::string &app_config_file, const char delimiter = '=');

    static uint8_t S_log_threshold;
#ifdef USE_MPI
    static int get_mpi_rank();

    static int get_mpi_size();

    static DTYPE(MPI_COMM_WORLD) get_mpi_comm();
#endif
    std::string get_oemd_masks_dir() const noexcept;

    ConcreteDaoType get_dao_type() const noexcept;

    size_t get_default_feature_quantization() const noexcept;

    float get_prediction_horizon() const noexcept;

    const std::string &get_db_connection_string() const noexcept;

    bool get_set_thread_affinity() const noexcept;

    size_t get_steps() const noexcept;

    size_t get_outputs() const noexcept;

    size_t get_online_learn_iter_limit() const noexcept;

    size_t get_stabilize_iterations_count() const noexcept;

    float get_scaling_alpha() const noexcept;

    bool get_tune_parameters() const noexcept;

    bool get_recombine_parameters() const noexcept;

    size_t get_slide_count() const noexcept;

    size_t get_slide_skip() const noexcept;

    size_t get_tune_run_limit() const noexcept;

    boost::log::trivial::severity_level get_log_level() const noexcept;

    bool get_self_request() const noexcept;

    std::chrono::milliseconds get_loop_interval() const noexcept;

    std::chrono::milliseconds get_stream_loop_interval() const noexcept;

    bool get_daemonize() const noexcept;

    uint16_t get_num_quantisations() const noexcept;

    uint16_t get_quantisation_divisor() const noexcept;

    uint16_t get_oemd_tune_particles() const noexcept;

    uint16_t get_oemd_tune_iterations() const noexcept;

    float get_solve_iterations_coefficient() const noexcept;

    static boost::log::trivial::severity_level set_global_log_level(const std::string &log_level_value);

    static boost::log::trivial::severity_level set_global_log_level(boost::log::trivial::severity_level log_threshold);

    static boost::log::trivial::severity_level set_global_log_level(uint8_t log_value);

#ifdef USE_DUCKDB
    const std::string &get_db_file() const noexcept;

    bool is_duck() const noexcept;

    static bool is_a_duck(const std::string &connection_str);
#endif

    std::tuple<bool, std::string> get_connection_arguments() const;
};

using MessageSource_ptr = std::shared_ptr<common::AppConfig>;
} /* namespace common */
} /* namespace svr */
