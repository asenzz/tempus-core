#pragma once

#include <boost/date_time/posix_time/posix_time.hpp> // TODO Port all boost::date_time code to std::chrono
#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <execution>
#include <cmath>
#include <deque>
#include "defines.h"

namespace svr {

constexpr auto C_default_exec_policy = std::execution::par_unseq;
const auto C_n_cpu = std::thread::hardware_concurrency();

#ifdef VALGRIND_BUILD
constexpr unsigned C_slide_skip = 1;
constexpr unsigned C_max_j = 1;
constexpr unsigned C_test_len = 1;
#else
constexpr unsigned C_slide_skip = 10;
constexpr unsigned C_max_j = 10;
constexpr unsigned C_test_len = 1000;
#endif
constexpr unsigned C_tune_min_validation_window = C_test_len - C_slide_skip * (C_max_j - 1);
constexpr double C_slides_len = [](){
    double r = 0;
    for (unsigned j = 0; j < C_max_j; ++j) r += C_test_len - C_slide_skip * (C_max_j - j - 1);
    return r;
} ();


namespace common {

constexpr unsigned C_min_cursor_rows = 1e4;
constexpr unsigned C_cursors_per_query = 8;

const double C_input_obseg_labels = 1;
constexpr double C_input_obseg_features = 1;

constexpr unsigned C_forecast_focus = 115;

constexpr uint8_t C_tune_keep_preds = 2;
const uint64_t C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS eg. 3^22, 4^18

constexpr double C_default_value_tick_volume = 1;

constexpr char C_input_queue_table_name_prefix[] = "q";
constexpr char C_decon_queue_table_name_prefix[] = "z";
constexpr char C_decon_queue_column_level_prefix[] = "level_";
constexpr char C_mt4_date_time_format[] = "%Y.%m.%d %H:%M:%S";

constexpr double C_kernel_path_tau = .25;

const unsigned C_omp_max_active_levels = []() {
    const auto p_env_max_active_levels = getenv("MAX_ACTIVE_LEVELS");
    return p_env_max_active_levels ? strtoul(p_env_max_active_levels, nullptr, 10) : 10 * C_n_cpu;
} ();
const unsigned C_omp_teams_thread_limit = []() {
    const auto p_env_teams_thread_limit = getenv("OMP_THREAD_LIMIT");
    return p_env_teams_thread_limit ? strtoul(p_env_teams_thread_limit, nullptr, 10) : 100 * C_n_cpu;
} ();

#ifdef INTEGRATION_TEST
const auto C_integration_test_validation_window = []() {
    constexpr unsigned test_offset_default = 345;
    const auto p = getenv("SVRWAVE_TEST_WINDOW");
    return p ? (unsigned) strtoul(p, nullptr, 10) : test_offset_default;
}();
#endif

constexpr double C_itersolve_delta = 1e-4;
constexpr double C_itersolve_range = 1e2;

constexpr double C_bad_validation = 1e6;
constexpr unsigned C_cu_tile_width = 32; // For Path kernel must be 32x32 == 1024 == Nx_local or 16 x 16 = 256, careful!
constexpr unsigned C_cu_block_size = C_cu_tile_width * C_cu_tile_width;
constexpr unsigned C_cu_clamp_n = C_cu_block_size * C_cu_block_size;

constexpr unsigned C_default_online_iter_limit = DEFAULT_ONLINE_ITER_LIMIT;
constexpr unsigned C_default_stabilize_iterations_count = DEFAULT_STABILIZE_ITERATIONS_COUNT;

constexpr char C_default_sql_properties_dir[] = "../SVRRoot/SVRPersist/postgres";
constexpr char C_default_log_level[] = "info";
constexpr char C_default_DAO_type[] = "postgres";
constexpr char C_default_online_iter_limit_str[] = TOSTR(DEFAULT_ONLINE_ITER_LIMIT);
constexpr char C_default_stabilize_iterations_count_str[] = TOSTR(DEFAULT_STABILIZE_ITERATIONS_COUNT);
constexpr char C_default_error_tolerance_str[] = "1e-7";
constexpr char C_default_onlinesvr_log_file[] = "learn_log.txt";
constexpr char C_default_tune_parameters_str[] = "0";
constexpr char C_default_recombine_parameters_str[] = "0";
constexpr char C_default_prediction_offset_str[] = "0.1"; // Prediction is ahead of last-known by main queue resolution * OFFSET_PRED_MUL
constexpr char C_default_feature_quantization_str[] = "10";
constexpr char C_default_slide_count_str[] = "13";
constexpr char C_default_tune_run_limit_str[] = "3600";
constexpr char C_default_scaling_alpha_str[] = "0.01";
constexpr char C_default_connection_str[] = "dbname=svrwave user=svrwave password=svrwave host=/var/run/postgresql";
constexpr char C_default_multistep_len_str[] = "1";
constexpr char C_default_multiout_str[] = "1";

constexpr unsigned C_max_csv_token_size = 0xFF;
constexpr unsigned C_default_kernel_max_chunk_len = 4000; // Matrices larger than 65535x65535 will require ILP64 indexing and CUDA kernel updated
const unsigned C_default_multistep_len = std::stoul(C_default_multistep_len_str);
const unsigned C_default_multiout = std::stoul(C_default_multiout_str);
constexpr unsigned C_default_gradient_count = 1;
constexpr unsigned C_default_level_count = 1;

constexpr unsigned C_default_hardware_concurrency = 16;
const boost::posix_time::time_duration C_default_features_max_time_gap = boost::posix_time::hours(60);

constexpr double C_FFA_alpha = .5;   // Starting randomness
constexpr double C_FFA_betamin = .5;   // Attraction
constexpr double C_FFA_gamma = 1.;   // Visibility

constexpr char C_test_primary_column[] = "xauusd_avg_bid"; // Ignore tuning or validating other input queue columns in case of aux columns

}

}
