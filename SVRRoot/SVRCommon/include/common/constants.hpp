#pragma once


#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <deque>

#include "defines.h"
#include "util/math_utils.hpp"

#ifdef INTEGRATION_TEST
const unsigned INTEGRATION_TEST_VALIDATION_WINDOW = getenv("SVRWAVE_TEST_WINDOW") ? strtoul(getenv("SVRWAVE_TEST_WINDOW"), nullptr, 10) : TEST_OFFSET_DEFAULT;
#else
constexpr unsigned INTEGRATION_TEST_VALIDATION_WINDOW = 0;
#endif

namespace svr {

#ifndef TUNE_HYBRID_IDEAL
constexpr unsigned EMO_SLIDE_SKIP = 6;
constexpr unsigned EMO_MAX_J = 17;
constexpr unsigned DEFAULT_EMO_TEST_LEN = 270;
constexpr unsigned EMO_TEST_LEN = DEFAULT_EMO_TEST_LEN; // = EMO_TUNE_TEST_SIZE
constexpr unsigned EMO_TUNE_MIN_VALIDATION_WINDOW = EMO_TEST_LEN - EMO_SLIDE_SKIP * (EMO_MAX_J - 1); // 200 seems to be the best validation window for predicting the next 120 hours, setting the window to 120 gives worse results for predicting the next 120 hours
constexpr unsigned EMO_SLIDES_LEN = [](){
    unsigned r = 0;
    for (unsigned j = 0; j < EMO_MAX_J; ++j)
        r += EMO_TEST_LEN - EMO_SLIDE_SKIP * (EMO_MAX_J - j - 1);
    return r;
} ();
// constexpr unsigned EMO_TUNE_TEST_SIZE = EMO_SLIDE_SKIP * EMO_MAX_J + EMO_TUNE_VALIDATION_WINDOW; // = EMO_TEST_LEN
#endif

namespace common {

constexpr unsigned C_min_cursor_rows = 1e4;
constexpr unsigned C_cursors_per_query = 8;

constexpr double C_input_obseg_labels = 1;
constexpr double C_input_obseg_features = C_input_obseg_labels;

constexpr unsigned C_forecast_focus = 115;

constexpr uint8_t C_tune_keep_preds = TUNE_KEEP_PREDS;
const uint64_t C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS // 3^22, 4^18

constexpr double C_default_value_tick_volume = 1;

const std::string C_input_queue_table_name_prefix{"q"};
const std::string C_decon_queue_table_name_prefix{"z"};
const std::string C_decon_queue_column_level_prefix{"level_"};
const std::string C_mt4_date_time_format{"%Y.%m.%d %H:%M:%S"};

const unsigned C_default_online_learn_iter_limit = atol(DEFAULT_ONLINE_ITER_LIMIT);
const unsigned C_default_stabilize_iterations_count = atol(DEFAULT_STABILIZE_ITERATIONS_COUNT);
constexpr unsigned C_max_csv_token_size = 0xFF;
constexpr unsigned C_kernel_default_max_chunk_size = 6000;
const unsigned C_default_multistep_len = atol(DEFAULT_MULTISTEP_LEN);
constexpr unsigned C_default_gradient_count = DEFAULT_SVRPARAM_GRAD_LEVEL + 1;
constexpr unsigned C_default_level_count = DEFAULT_SVRPARAM_DECON_LEVEL + 1;
constexpr unsigned C_max_predict_chunk_size = C_kernel_default_max_chunk_size / 10;


const unsigned C_omp_max_active_levels = []() {
    const auto p_env_max_active_levels = getenv("MAX_ACTIVE_LEVELS");
    return p_env_max_active_levels ? strtoul(p_env_max_active_levels, nullptr, 10) : 10 * std::thread::hardware_concurrency();
} ();
const unsigned C_omp_teams_thread_limit = []() {
    const auto p_env_teams_thread_limit = getenv("OMP_THREAD_LIMIT");
    return p_env_teams_thread_limit ? strtoul(p_env_teams_thread_limit, nullptr, 10) : 100 * std::thread::hardware_concurrency();
} ();


}
}
