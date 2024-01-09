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
#endif

namespace svr {
namespace common {

constexpr unsigned MAX_TUNE_CHUNKS = 1;

constexpr double C_input_obseg_labels = 1; // 83.3553;
constexpr double C_input_obseg_features = C_input_obseg_labels; // / 1660.; // Best for direction .12, .07324079159, .05547593844

constexpr double C_singlesolve_delta = 1e-4;

// const std::deque<double> C_tune_crass_epscost {svr::common::C_input_obseg_labels, 1., 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-4}; // Use for lower obseg eg. 0.5
// const std::deque<double> C_tune_crass_epscost = {1. / (C_input_obseg_labels * C_input_obseg_labels), 1. / C_input_obseg_labels, 1., C_input_obseg_labels, C_input_obseg_labels * C_input_obseg_labels};
const std::deque<double> C_tune_crass_epscost = {0.};
// const std::deque<double> C_tune_crass_epscost {1e-4, 1e-5, 1e-6, 1e-7, 1e-9}; // Higher costs
// const auto C_tune_crass_epscost = [](){ std::deque<double> r; for (double div = common::C_input_obseg_labels; div >= 1; div /= 2) r.emplace_back(svr::common::C_input_obseg_labels / div); return r; } ();
/*
const auto C_tune_crass_epscost = []() {
    std::deque<double> r = {C_input_obseg_features};
    for (double x = 1; x <= 1; x += 1) {
        r.emplace_back(std::pow<double>(C_input_obseg_labels, x));
        // r.emplace_back(std::pow<double>(C_input_obseg_labels, -x));
    }
    return r;
} ();
*/

const std::deque<double> C_tune_lambdas_path {0., 1e-3, .2, .5, .75, 1., 1.5, 2., 3., 15., 75.};
// const std::deque<double> C_tune_lambdas_path {0., 1., 1e1, 1e2};
// const std::deque<double> C_tune_lambdas_path {1.};
// const std::deque<double> C_tune_lambdas_path {0., .2, .75, 1., 3., 75.};
// const double C_tune_lambda_max = *std::max_element(C_tune_lambdas_path.begin(), C_tune_lambdas_path.end()); // Max from above array
// const double C_tune_lambda_min = *std::min_element(C_tune_lambdas_path.begin(), C_tune_lambdas_path.end()); // Min from above array

constexpr uint16_t C_forecast_focus = 115;

constexpr uint8_t   C_tune_keep_preds = TUNE_KEEP_PREDS;
const uint64_t  C_num_combos = std::pow<uint64_t>(3, 22); // should be even power of TUNE_KEEP_PREDS // 3^22, 4^18

constexpr double    C_default_value_tick_volume = 1;

const std::string C_input_queue_table_name_prefix {"q"};
const std::string C_decon_queue_table_name_prefix {"z"};
const std::string C_decon_queue_column_level_prefix {"level_"};
const std::string C_mt4_date_time_format {"%Y.%m.%d %H:%M:%S"};

const size_t __online_iters_limit_mult = atol(DEFAULT_ONLINE_ITERS_LIMIT_MULT);
const size_t __online_learn_iter_limit = atol(DEFAULT_LEARN_ITER_LIMIT);
const size_t __max_variations = atol(DEFAULT_MAX_VARIATIONS);
const double __smo_epsilon_divisor = atof(DEFAULT_SMO_EPSILON_DIVISOR);
const double __smo_cost_divisor = atof(DEFAULT_SMO_COST_DIVISOR);
const size_t __stabilize_iterations_count = atol(DEFAULT_STABILIZE_ITERATIONS_COUNT);
const long __default_number_variations = atol(DEFAULT_DEFAULT_NUMBER_VARIATIONS);
constexpr size_t __max_iter = MAX_ITER_SMO;
const bool __dont_update_r_matrix = atoi(DEFAULT_DONT_UPDATE_R_MATRIX);
const size_t __multistep_len = atol(DEFAULT_MULTISTEP_LEN);


}
}
