//
// Created by zarko on 30.12.20 г..
//
#include <gtest/gtest.h>
#include <fstream>
#include <chrono>

#include "../include/spectral_transform.hpp"
#include "../include/fast_cvmd.hpp"
#include "common/compatibility.hpp"
#include "test_harness.hpp"
#include "util/string_utils.hpp"
#include <DQScalingFactorService.hpp>
#include "DeconQueueService.hpp"
#include "model/DataRow.hpp"
#include "common/parallelism.hpp"


namespace svr {

constexpr size_t INPUT_LIMIT = CVMD_INIT_LEN;
constexpr unsigned DECON_LEVELS = 8;
constexpr double TEST_TOL = 1e-14;
const std::string transform_input_filename = "xauusd_1_2023.csv";
// #define TEST_TIME_INVARIANCE

class cvmd_transform_test : public testing::Test {
public:
    cvmd_transform_test() :
            input("input", "input", "", "", bpt::seconds(1)),
            input_trimmed("input_trimmed", "input_trimmed", "", "", bpt::seconds(1)),
#ifdef VMD_ONLY
            decon("decon", "decon", "decon", 0, DECON_LEVELS),
            decon_trimmed("decon_trimmed", "decon_trimmed", "decon_trimmed", 0, DECON_LEVELS),
#else
            decon("decon", "decon", "decon", 0, 2 * DECON_LEVELS),
            decon_trimmed("decon_trimmed", "decon_trimmed", "decon_trimmed", 0, 2 * DECON_LEVELS),
#endif
            transformer(DECON_LEVELS),
            transformer2(DECON_LEVELS)
    {
        input.set_value_columns(std::deque<std::string>{"xauusd_avg"});
        input_trimmed.set_value_columns(std::deque<std::string>{"xauusd_avg"});
    }

    ~cvmd_transform_test() = default;

    using testing::Test::SetUp;

    virtual void SetUp(
            const std::string &input_filename,
            const size_t levels = DECON_LEVELS)
    {
        auto input_f = open_data_file(input_filename);
        EXPECT_TRUE(input_f.good());
        std::string line;
        while (not input_f.eof()) {
            std::getline(input_f, line);
            if (line.empty()) continue;
            input.get_data().emplace_back(datamodel::DataRow::load(line));
            if (input.size() < 10) LOG4_DEBUG("Read " << *input.back());
            if (input.size() >= INPUT_LIMIT) break;
        }
        LOG4_DEBUG("Read total " << input.size() << " lines.");
    }

    datamodel::InputQueue input, input_trimmed;
    datamodel::DeconQueue decon, decon_trimmed;
    std::ifstream result_f;
    vmd::fast_cvmd transformer, transformer2;
};

TEST_F(cvmd_transform_test, test_transform_correctness)
{
    PROFILE_INFO(SetUp(transform_input_filename), "Test setup");
    auto start = std::chrono::steady_clock::now();
    const auto iq_scaling_factors = business::IQScalingFactorService::calculate(input, 0, std::numeric_limits<size_t>::max());
    LOG4_DEBUG("IQ scaling factor " << iq_scaling_factors.front()->to_string());
    const auto scaler = business::IQScalingFactorService::get_scaler(*iq_scaling_factors.front());
    const auto unscaler = business::IQScalingFactorService::get_unscaler(*iq_scaling_factors.front());
#ifndef ORIG_VMD
    PROFILE_INFO(transformer.initialize(input, 0, decon.get_table_name(), scaler), "VMD initialize");
#endif
    PROFILE_INFO(transformer.transform(input, decon, 0, 0, scaler), "VMD batch transform");
#ifdef TEST_TIME_INVARIANCE
    constexpr size_t C_decon_offset = 5000;

    // PROFILE_INFO(transformer2.initialize(input_trimmed.get_data(), "Test table 2"), "VMD initialize trimmed");
    transformer2.set_vmd_frequencies(transformer.get_vmd_frequencies());
    input_trimmed = input;
    input_trimmed.get_data().erase(input_trimmed.begin(), input_trimmed.begin() + C_decon_offset);
    PROFILE_INFO(transformer2.transform(input_trimmed.get_data(), decon_trimmed, 0, 0, scaler), "VMD batch transform trimmed");
    auto start_t = transformer.get_residuals_length("dont find") + C_decon_offset;
    if (start_t > decon_trimmed.size()) start_t = 0;
    const auto test_len = decon_trimmed.size() - start_t;
    LOG4_INFO("Transformation took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, decon size " << decon.size() << " x "
                << decon[0]->get_values().size() << ", validation start " << start_t << ", decon trimmed size " << decon_trimmed.size());
#else
    const size_t start_t = 0;
    const size_t test_len = decon.size() - start_t;
#endif
    std::vector<std::vector<double>> values(DECON_LEVELS, std::vector<double>(test_len));
#pragma omp parallel for num_threads(adj_threads(test_len))
#ifdef TEST_TIME_INVARIANCE
    for (size_t t = start_t; t < decon_trimmed.size(); ++t) {
        double err = 0;
#else
    for (size_t t = start_t; t < test_len; ++t) {
#endif

#ifdef VMD_ONLY
        for (unsigned l = 0; l < DECON_LEVELS; ++l) {
#ifdef TEST_TIME_INVARIANCE
            err += abs(decon[t + C_decon_offset]->at(l) - decon_trimmed[t]->at(l));
#endif
            values[l][t - start_t] = decon[t]->at(l);
        }
#else
        for (unsigned l = 0; l < DECON_LEVELS; ++l) {
#ifdef TEST_TIME_INVARIANCE
            err += abs(decon[t + C_decon_offset]->at(DECON_LEVELS + l) - decon_trimmed[t]->at(DECON_LEVELS + l));
#endif
            values[l][t - start_t] = decon[t]->at(DECON_LEVELS + l);
        }
#endif
#ifdef TEST_TIME_INVARIANCE
        if (err > TEST_TOL) LOG4_DEBUG("Time " << decon[t]->get_value_time() << ", difference " << err << " at " << t); // CVMD is not time invariant unless omega is preserved
#endif
    }

    OMP_FOR_i_(values.size(), ordered) {
        const auto mean_abs_level = common::meanabs(values[i]);
#pragma omp ordered
        LOG4_DEBUG("Level " << i << ", power " << mean_abs_level);
    }

    data_row_container recon(decon.size());
    double min_v = 0;
#if 0
    OMP_FOR_i_(decon.size(), simd reduction(min:min_v)) {
#else
    for (unsigned i = 0; i < decon.size(); ++i) {
#endif
        const auto &d = *decon[i];
        double v = 0;
        UNROLL(DECON_LEVELS)
        for (size_t l = 0; l < DECON_LEVELS; l += LEVEL_STEP) {
#ifdef VMD_ONLY
            v += d[l];

#else
            v += d[DECON_LEVELS + l];
#endif
        }
        OMPMINAS(min_v, v);
        recon[i] = ptr<datamodel::DataRow>(d.get_value_time(), bpt::second_clock::local_time(), d.get_tick_volume(), std::vector{v});
    }
    if (false /* min_v < 0 */) {
        LOG4_DEBUG("Minimum value " << min_v << " found.");
        OMP_FOR(recon.size())
        for (const auto &r: recon)
            r->at(0) -= min_v;
    }
    OMP_FOR(recon.size())
    for (const auto &r: recon)
        r->at(0) = unscaler(r->at(0));
    double total_diff = 0, highest_diff = 0;
#if 0
    OMP_FOR_i_(recon.size(), simd reduction(+:total_diff) reduction(max:highest_diff)) {
#else
    for (unsigned i = 0; i < recon.size(); ++i) {
#endif
        const auto diff = std::abs(input[i]->at(0) - recon[i]->at(0));
        OMPMAXAS(highest_diff, diff);
        if (diff > TEST_TOL) LOG4_WARN("Input value " << *input[i] << " differs " << diff << " from recon value " << *recon[i] << " at index " << i);
        total_diff += diff;
    }
    LOG4_INFO("Test took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, total reconstruction error " << total_diff <<
                           ", average error " << total_diff / double(input.size()) << ", highest diff " << highest_diff << ", samples count " << decon.size());
}

}
