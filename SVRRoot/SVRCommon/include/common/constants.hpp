#pragma once


#include <cstdlib>
#include <string>
#include <array>
#include <algorithm>
#include <cmath>
#include <deque>

#include "defines.h"
#include "util/math_utils.hpp"

#ifdef MANIFOLD_TEST
const unsigned MANIFOLD_TEST_VALIDATION_WINDOW = getenv("SVRWAVE_TEST_WINDOW") ? strtoul(getenv("SVRWAVE_TEST_WINDOW"), nullptr, 10) : TEST_OFFSET_DEFAULT;
#else
constexpr unsigned MANIFOLD_TEST_VALIDATION_WINDOW = 0;
#endif

namespace svr {

constexpr unsigned MAX_CSV_TOKEN_SIZE = 0xFF;
constexpr unsigned C_kernel_default_max_chunk_size = 6000;
constexpr unsigned C_max_predict_chunk_size = C_kernel_default_max_chunk_size / 10;

#ifndef TUNE_HYBRID_IDEAL
constexpr unsigned EMO_SLIDE_SKIP = 7;
constexpr unsigned EMO_MAX_J = 15;
constexpr unsigned DEFAULT_EMO_SLIDE_LEN = 270; // 65 // TODO test 5 * 16 + 190 / epsco 20
constexpr unsigned EMO_SLIDE_LEN = DEFAULT_EMO_SLIDE_LEN; // = EMO_TUNE_TEST_SIZE
constexpr unsigned EMO_TUNE_VALIDATION_WINDOW = EMO_SLIDE_LEN - EMO_SLIDE_SKIP * EMO_MAX_J; // 200 seems to be the best validation window for predicting the next 120 hours, setting the window to 120 gives worse results for predicting the next 120 hours
constexpr unsigned EMO_TUNE_TEST_SIZE = EMO_SLIDE_SKIP * EMO_MAX_J + EMO_TUNE_VALIDATION_WINDOW; // = EMO_SLIDE_LEN
#endif

namespace common {

constexpr double C_input_obseg_labels = 1;
constexpr double C_input_obseg_features = C_input_obseg_labels;

constexpr uint16_t C_forecast_focus = 115;

constexpr uint8_t   C_tune_keep_preds = TUNE_KEEP_PREDS;
const uint64_t  C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS // 3^22, 4^18

constexpr double    C_default_value_tick_volume = 1;

const std::string C_input_queue_table_name_prefix {"q"};
const std::string C_decon_queue_table_name_prefix {"z"};
const std::string C_decon_queue_column_level_prefix {"level_"};
const std::string C_mt4_date_time_format {"%Y.%m.%d %H:%M:%S"};

const size_t __online_learn_iter_limit = atol(DEFAULT_ONLINE_ITER_LIMIT);
const size_t __stabilize_iterations_count = atol(DEFAULT_STABILIZE_ITERATIONS_COUNT);
const size_t __multistep_len = atol(DEFAULT_MULTISTEP_LEN);


}
}
