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

constexpr unsigned INPUT_LIMIT = 100000; // std::numeric_limits<size_t>::max();
constexpr unsigned DECON_LEVELS = 32;
constexpr double TEST_TOL = 1e-15;
const std::string transform_input_filename = "/mnt/faststore/repo/tempus-core/SVRRoot/OnlineSVR/test/test_data/xauusd_1_2023.csv";


class cvmd_transform_test : public testing::Test
{
public:
    cvmd_transform_test() :
            input("input", "input", "", "", bpt::hours(1)),
            input_trimmed("input_trimmed", "input_trimmed", "", "", bpt::hours(1)),
            decon("decon", "decon", 0, DECON_LEVELS),
            decon_trimmed("decon_trimmed", "decon_trimmed", 0, DECON_LEVELS),
            transformer(DECON_LEVELS),
            transformer2(DECON_LEVELS)
    {
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
            std::stringstream str(line);
            std::string word;
            while(getline(str, word, ',')) {
                input.get_data().emplace_back(svr::datamodel::DataRow::load(word));
            }
            if (input.size() < 10) LOG4_DEBUG("Read " << input.back());
            if (input.size() >= INPUT_LIMIT) break;
        }
        LOG4_DEBUG("Read total " << input.size() << " lines.");
    }

    svr::datamodel::InputQueue input, input_trimmed;
    svr::datamodel::DeconQueue decon, decon_trimmed;
    std::ifstream result_f;
    svr::fast_cvmd transformer, transformer2;
};

constexpr size_t C_decon_offset = 5000;

TEST_F(cvmd_transform_test, test_transform_correctness)
{
    SetUp(transform_input_filename);
    auto start = std::chrono::steady_clock::now();
    input_trimmed = input;
    input_trimmed.get_data().erase(input_trimmed.begin() + C_decon_offset, input_trimmed.end());
    PROFILE_EXEC_TIME(transformer.initialize(input, 0, input.get_table_name()), "VMD initialize");
    PROFILE_EXEC_TIME(transformer.transform(input, decon), "VMD batch transform");
    // PROFILE_EXEC_TIME(transformer2.initialize(input_trimmed.get_data(), "Test table 2"), "VMD initialize trimmed");
    transformer2.set_vmd_frequencies(transformer.get_vmd_frequencies());
    PROFILE_EXEC_TIME(transformer2.transform(input_trimmed.get_data(), decon_trimmed), "VMD batch transform trimmed");
    LOG4_INFO("Transformation took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, decon size " << decon.size() << " x " << decon[0]->get_values().size());

    std::vector<std::vector<double>> values(DECON_LEVELS);
    const auto start_t = transformer.get_residuals_length("dont find") + C_decon_offset;
    const auto test_len = decon_trimmed.size() - start_t;
#pragma omp parallel for ordered num_threads(svr::adj_threads(test_len))
    for (size_t t = start_t; t < decon_trimmed.size(); ++t) {
        double err = 0;
// #pragma omp parallel for num_threads(svr::adj_threads(DECON_LEVELS)) reduction(+:err)
        for (size_t l = 0; l < DECON_LEVELS; ++l) {
            err += decon[t + C_decon_offset]->at(l) - decon_trimmed[t]->at(l);
            values[l][t - start_t] = decon[t]->at(l);
        }

        if (err > TEST_TOL) LOG4_DEBUG("Time " << C_decon_offset << " variance " << err << " at " << t);
    }

#pragma omp parallel for num_threads(svr::adj_threads(values.size()))
    for (size_t l = 0; l < values.size(); ++l)
        LOG4_DEBUG("Power level " << l << " " << svr::common::meanabs(values[l]));

    const auto recon = svr::business::DeconQueueService::reconstruct(decon, svr::business::recon_type_e::ADDITIVE);

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