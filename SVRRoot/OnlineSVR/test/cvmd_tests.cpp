//
// Created by zarko on 30.12.20 г..
//
#include <gtest/gtest.h>
#include <fstream>
#include <chrono>

#include "../include/spectral_transform.hpp"
#include "../include/fast_cvmd.hpp"
#include "test_harness.hpp"
#include "util/string_utils.hpp"
#include <DQScalingFactorService.hpp>
#include "DeconQueueService.hpp"
#include "model/DataRow.hpp"
#include "common/parallelism.hpp"


namespace {

constexpr unsigned INPUT_LIMIT = std::numeric_limits<size_t>::max();
constexpr unsigned DECON_LEVELS = 32;
constexpr double TEST_TOL = 1e-14;
const std::string transform_input_filename = "xauusd_1_2023.csv";


class cvmd_transform_test : public testing::Test
{
public:
    cvmd_transform_test() :
            input("input", "input", "", "", bpt::seconds(1)),
            input_trimmed("input_trimmed", "input_trimmed", "", "", bpt::seconds(1)),
            decon("decon", "decon", "decon", 0, 2 * DECON_LEVELS),
            decon_trimmed("decon_trimmed", "decon_trimmed", "decon_trimmed", 0, 2 * DECON_LEVELS),
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
        while(not input_f.eof()) {
            std::getline(input_f, line);
            if (line.empty()) continue;
            input.get_data().emplace_back(svr::datamodel::DataRow::load(line));
            if (input.size() < 10) LOG4_DEBUG("Read " << *input.back());
            if (input.size() >= INPUT_LIMIT) break;
        }
        LOG4_DEBUG("Read total " << input.size() << " lines.");
    }

    svr::datamodel::InputQueue input, input_trimmed;
    svr::datamodel::DeconQueue decon, decon_trimmed;
    std::ifstream result_f;
    svr::vmd::fast_cvmd transformer, transformer2;
};

constexpr size_t C_decon_offset = 5000;

TEST_F(cvmd_transform_test, test_transform_correctness)
{
    SetUp(transform_input_filename);
    auto start = std::chrono::steady_clock::now();
    const auto iq_scaling_factors = svr::business::IQScalingFactorService::calculate(input, 0, std::numeric_limits<size_t>::max());
    const auto sf = iq_scaling_factors.front()->get_scaling_factor();
    svr::business::t_iqscaler scaler = [sf](const double v) -> double { return v / sf; }, unscaler = [sf](const double v) -> double { return v * sf; };
    PROFILE_EXEC_TIME(transformer.initialize(input, 0, decon.get_table_name(), scaler), "VMD initialize");
    PROFILE_EXEC_TIME(transformer.transform(input, decon, 0, 0, scaler), "VMD batch transform");
    // PROFILE_EXEC_TIME(transformer2.initialize(input_trimmed.get_data(), "Test table 2"), "VMD initialize trimmed");
    transformer2.set_vmd_frequencies(transformer.get_vmd_frequencies());
    input_trimmed = input;
    input_trimmed.get_data().erase(input_trimmed.begin(), input_trimmed.begin() + C_decon_offset);
    PROFILE_EXEC_TIME(transformer2.transform(input_trimmed.get_data(), decon_trimmed, 0, 0, scaler), "VMD batch transform trimmed");
    auto start_t = transformer.get_residuals_length("dont find") + C_decon_offset;
    if (start_t > decon_trimmed.size()) start_t = 0;
    const auto test_len = decon_trimmed.size() - start_t;
    LOG4_INFO("Transformation took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, decon size " << decon.size() << " x " << decon[0]->get_values().size() << ", validation start " << start_t << ", decon trimmed size " << decon_trimmed.size());
    std::vector<std::vector<double>> values(DECON_LEVELS, std::vector<double>(test_len));
#pragma omp parallel for ordered num_threads(svr::adj_threads(test_len))
    for (size_t t = start_t; t < decon_trimmed.size(); ++t) {
        double err = 0;
// #pragma omp parallel for num_threads(svr::adj_threads(DECON_LEVELS)) reduction(+:err)
        for (size_t l = 0; l < DECON_LEVELS; ++l) {
            err += abs(decon[t + C_decon_offset]->at(DECON_LEVELS + l) - decon_trimmed[t]->at(DECON_LEVELS + l));
            values[l][t - start_t] = unscaler(decon[t]->at(DECON_LEVELS + l));
        }

        if (err > TEST_TOL) LOG4_DEBUG("Time " << C_decon_offset << " variance " << err << " at " << t); // CVMD is not time invariant unless a state is preserved
    }

#pragma omp parallel for ordered schedule(static, 1) num_threads(svr::adj_threads(values.size()))
    for (size_t l = 0; l < values.size(); ++l) {
        const auto mean_abs_level = svr::common::meanabs(values[l]);
#pragma omp ordered
        LOG4_DEBUG("Power level " << l << " " << mean_abs_level);
    }

    const auto recon = svr::business::DeconQueueService::reconstruct(decon, svr::business::recon_type_e::ADDITIVE, unscaler);

    double total_diff = 0;
    double highest_diff = 0;
    for (size_t i = 0; i < recon.size(); ++i) {
        const auto diff = std::abs(input[i]->at(0) - recon[i]->at(0));
        if (diff > highest_diff) highest_diff = diff;
        if (diff > TEST_TOL)
            LOG4_WARN("Input value " << *input[i] << " differs " << diff << " from recon value " << *recon[i] << " at index " << i);
        total_diff += diff;
    }
    LOG4_INFO("Test took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, total reconstruction error " << total_diff <<
                ", average error " << total_diff / double(input.size() - C_decon_offset) << ", highest diff " << highest_diff);
}

}
