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
const auto C_n_cpu_str = std::to_string(C_n_cpu);

namespace common {

// For Postgres queries
constexpr uint32_t C_min_cursor_rows = 1e4;

const double C_input_obseg_labels = 1;
constexpr double C_input_obseg_features = 1;

// Save first N forecasts to database for later analysis
constexpr uint16_t C_forecast_focus = 115;

constexpr uint8_t C_tune_keep_preds = 2;
const uint64_t C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS eg. 3^22, 4^18

constexpr double C_default_value_tick_volume = 1;

constexpr char C_input_queue_table_name_prefix[] = "q";
constexpr char C_decon_queue_table_name_prefix[] = "z";
constexpr char C_decon_queue_column_level_prefix[] = "level_";
constexpr char C_mql_date_time_format[] = "%Y.%m.%d %H:%M:%S";

#ifdef INTEGRATION_TEST

#ifdef VALGRIND_BUILD
constexpr uint16_t C_integration_test_validation_window = 2;
#else
const auto C_integration_test_validation_window = [] {
    constexpr uint16_t test_offset_default = 460;
    const auto p = getenv("SVRWAVE_TEST_WINDOW");
    if (!p) {
        std::cout << "Environment variable SVRWAVE_TEST_WINDOW not set, using default " << test_offset_default << std::endl;
        return test_offset_default;
    } else {
        const auto r = (uint16_t) strtoul(p, nullptr, 10);
        return r;
    }
}();
#endif

#else

constexpr uint16_t C_integration_test_validation_window = 0;

#endif

constexpr double C_itersolve_delta = 1e-4;
constexpr double C_itersolve_range = 1e2;

constexpr double C_bad_validation = 1e9;

#ifdef NDEBUG
constexpr uint16_t C_cu_tile_width = 32;
#else
constexpr uint16_t C_cu_tile_width = 16; // For Path kernel must be 32x32 == 1024 == Nx_local or 16 x 16 = 256, careful!
#endif

constexpr uint32_t C_cu_block_size = C_cu_tile_width * C_cu_tile_width;
constexpr uint32_t C_cu_clamp_n = C_cu_block_size * C_cu_block_size;

constexpr uint16_t C_default_online_iter_limit = DEFAULT_ONLINE_ITER_LIMIT;
constexpr uint16_t C_default_stabilize_iterations_count = DEFAULT_STABILIZE_ITERATIONS_COUNT;

constexpr char C_default_sql_properties_dir[] = "../SVRRoot/SVRPersist/postgres";
constexpr char C_default_log_level[] = "info";
constexpr char C_default_DAO_type[] = "postgres";
constexpr char C_default_online_iter_limit_str[] = TOSTR(DEFAULT_ONLINE_ITER_LIMIT);
constexpr char C_default_stabilize_iterations_count_str[] = TOSTR(DEFAULT_STABILIZE_ITERATIONS_COUNT);
constexpr char C_default_tune_parameters_str[] = "0";
constexpr char C_default_recombine_parameters_str[] = "0";
constexpr char C_default_prediction_horizon_str[] = "0.1"; // Prediction is ahead of last-known by main queue resolution * OFFSET_PRED_MUL
constexpr char C_default_feature_quantization_str[] = "10";
constexpr char C_default_slide_count_str[] = "13";
constexpr char C_default_tune_run_limit_str[] = "14400";
constexpr char C_default_scaling_alpha_str[] = "0.01";
constexpr char C_default_connection_str[] = "dbname=svrwave user=svrwave password=svrwave host=/var/run/postgresql";
constexpr char C_default_multistep_len_str[] = "1";
constexpr char C_default_multiout_str[] = "1";
constexpr char C_default_loop_count[] = "-1";
constexpr char C_default_loop_interval_ms[] = "1000";
constexpr char C_default_stream_loop_interval_ms[] = "10";
constexpr char C_default_daemonize[] = "1";
constexpr char C_default_num_quantisations[] = "10";
constexpr char C_default_quantisation_divisor[] = "3";
constexpr char C_default_oemd_quantisation_skipdiv[] = "20";
constexpr char C_default_oemd_tune_particles[] = "16";
constexpr char C_default_oemd_tune_iterations[] = "24";
constexpr char C_default_tune_particles[] = "16";
constexpr char C_default_tune_iterations[] = "24";
constexpr char C_defaut_solve_iterations_coefficient[] = ".1";

constexpr uint16_t C_max_csv_token_size = 0xFF;

const uint16_t C_default_multistep_len = std::stoi(C_default_multistep_len_str);
const uint16_t C_default_multiout = std::stoi(C_default_multiout_str);
constexpr uint16_t C_default_gradient_count = 1;
constexpr uint16_t C_default_level_count = 1;

constexpr uint16_t C_default_hardware_concurrency = 16;
const boost::posix_time::time_duration C_default_features_max_time_gap = boost::posix_time::hours(60);

}

}
