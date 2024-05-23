#pragma once

#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <deque>

#include "defines.h"
#include "util/math_utils.hpp"

namespace svr {

#ifdef VALGRIND_BUILD
constexpr unsigned C_emo_slide_skip = 1;
constexpr unsigned C_emo_max_j = 1;
constexpr unsigned C_emo_test_len = 1;
#else
constexpr unsigned C_emo_slide_skip = 5;
constexpr unsigned C_emo_max_j = 20;
constexpr unsigned C_emo_test_len = 17 * 17; // Must be square-rootable
#endif
constexpr unsigned C_emo_tune_min_validation_window = C_emo_test_len - C_emo_slide_skip * (C_emo_max_j - 1);
constexpr double C_emo_slides_len = [](){
    double r = 0;
    for (unsigned j = 0; j < C_emo_max_j; ++j)
        r += C_emo_test_len - C_emo_slide_skip * (C_emo_max_j - j - 1);
    return r;
} ();


namespace common {

constexpr unsigned C_min_cursor_rows = 1e4;
constexpr unsigned C_cursors_per_query = 8;

const double C_input_obseg_labels = 1;
constexpr double C_input_obseg_features = 1;

constexpr unsigned C_forecast_focus = 115;

constexpr uint8_t C_tune_keep_preds = 2;
const uint64_t C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS // 3^22, 4^18

constexpr double C_default_value_tick_volume = 1;

const std::string C_input_queue_table_name_prefix{"q"};
const std::string C_decon_queue_table_name_prefix{"z"};
const std::string C_decon_queue_column_level_prefix{"level_"};
const std::string C_mt4_date_time_format{"%Y.%m.%d %H:%M:%S"};

const unsigned C_default_online_learn_iter_limit = atol(DEFAULT_ONLINE_ITER_LIMIT);
const unsigned C_default_stabilize_iterations_count = atol(DEFAULT_STABILIZE_ITERATIONS_COUNT);
constexpr unsigned C_max_csv_token_size = 0xFF;
constexpr unsigned C_default_kernel_max_chunk_size = 3000;
const unsigned C_default_multistep_len = atol(DEFAULT_MULTISTEP_LEN);
constexpr unsigned C_default_gradient_count = DEFAULT_SVRPARAM_GRAD_LEVEL + 1;
constexpr unsigned C_default_level_count = DEFAULT_SVRPARAM_DECON_LEVEL + 1;
constexpr double C_kernel_path_tau = .25;

const unsigned C_omp_max_active_levels = []() {
    const auto p_env_max_active_levels = getenv("MAX_ACTIVE_LEVELS");
    return p_env_max_active_levels ? strtoul(p_env_max_active_levels, nullptr, 10) : 10 * std::thread::hardware_concurrency();
} ();
const unsigned C_omp_teams_thread_limit = []() {
    const auto p_env_teams_thread_limit = getenv("OMP_THREAD_LIMIT");
    return p_env_teams_thread_limit ? strtoul(p_env_teams_thread_limit, nullptr, 10) : 100 * std::thread::hardware_concurrency();
} ();

#ifdef INTEGRATION_TEST
const unsigned INTEGRATION_TEST_VALIDATION_WINDOW = []() {
    constexpr unsigned TEST_OFFSET_DEFAULT = 345;
    const auto p = getenv("SVRWAVE_TEST_WINDOW");
    return p ? strtoul(p, nullptr, 10) : TEST_OFFSET_DEFAULT;
}();
#endif

constexpr double C_itersolve_delta = 1e-4;
constexpr double C_itersolve_range = 1e2;

}
}
