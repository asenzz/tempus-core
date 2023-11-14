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

#define INPUT_LIMIT 100000 // std::numeric_limits<size_t>::max()
#define DECON_LEVELS 32
#define TEST_TOL 1e-15


const std::string transform_input_filename = "online_emd_test_data/0.98_eurusd_avg_1_test_data.sql"; // "cvmd_test_data/transform_input";
const std::string transform_result_filename = "cvmd_test_data/transform_output_real";

namespace {

class cvmd_transform_test : public testing::Test
{
public:
    cvmd_transform_test()
    {
    }
    ~cvmd_transform_test() = default;

    virtual
    void SetUp(
            const std::string &input_filename,
            const std::string &exp_output_filename,
            const size_t levels = DECON_LEVELS)
    {
        transformer = svr::spectral_transform::create(std::string("cvmd"), levels);
        transformer2 = svr::spectral_transform::create(std::string("cvmd"), levels);
        auto input_f = open_data_file(input_filename);
        EXPECT_TRUE(input_f.good());
        std::string line;
        while(not input_f.eof()) {
            std::getline(input_f, line);
            if (line.empty()) continue;
            std::stringstream str(line);
            std::string word;
            size_t i = 0;
            while(getline(str, word, '\t')) {
                if (i == 3) input.push_back(std::atof(word.c_str()));
                ++i;
            }
            if (input.size() < 20) LOG4_DEBUG("Read " << input.back());
            if (input.size() >= INPUT_LIMIT) break;
        }
        LOG4_DEBUG("Read total " << input.size() << " lines.");
        //EXPECT_TRUE(input_f.eof());
    }

    //virtual void TearDown(){}
    std::vector<double> input;
    std::ifstream result_f;
    std::unique_ptr<svr::spectral_transform> transformer, transformer2;
};

TEST_F(cvmd_transform_test, test_transform_correctness)
{
    SetUp(transform_input_filename, transform_result_filename);
    std::vector<std::vector<double>> decon, decon_trimmed;
    std::vector<double> recon;
    auto start = std::chrono::steady_clock::now();
    const auto p_cvmd_transformer = dynamic_cast<svr::fast_cvmd *>(transformer.get());
    const auto p_cvmd_transformer2 = dynamic_cast<svr::fast_cvmd *>(transformer2.get());
    const size_t decon_offset = 5000;
    std::vector<double> input_trimmed(input.begin() + decon_offset, input.end());
    PROFILE_EXEC_TIME(p_cvmd_transformer->initialize({input.begin(), input.end()}, "Test table"), "VMD initialize");
    //PROFILE_EXEC_TIME(p_cvmd_transformer2->initialize(input_trimmed, "Test table 2"), "VMD initialize trimmed");
    PROFILE_EXEC_TIME(p_cvmd_transformer->transform(input, decon, "Test table"), "VMD batch transform");
    p_cvmd_transformer2->set_vmd_frequencies(p_cvmd_transformer->get_vmd_frequencies());
    PROFILE_EXEC_TIME(p_cvmd_transformer2->transform(input_trimmed, decon_trimmed, "Test table", decon[decon_offset - 1]), "VMD batch transform trimmed");
/*
    data_row_container decon_data;
    data_row_container input_data;
    for (size_t i = 0; i < input_trimmed.size(); ++i) {
        const boost::posix_time::ptime row_time = boost::posix_time::second_clock::local_time() + boost::posix_time::hours(4 * i);
        input_data.insert({row_time, std::make_shared<svr::datamodel::DataRow>(row_time, row_time, 0, std::vector<double>(input_trimmed[i]))});
    }
    //svr::business::DeconQueueService::copy_decon_data_to_container(decon_data, input_data, decon);
    const std::vector<double> scaling_factors(DECON_LEVELS, .0001);
    const std::vector<double> mean_values(DECON_LEVELS, 0);
    //auto delta_data = svr::business::DeconQueueService::decon_queue_delta(decon_data, true);
//    svr::business::DQScalingFactorService::scale_decon_queue(delta_data, scaling_factors, mean_values);
*/
    LOG4_INFO("Transformation took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, decon size " << decon.size() << " x " << decon[0].size());

    std::vector<double> averages(DECON_LEVELS, 0);
    for (size_t t = 0; t < decon_trimmed.size(); ++t) {
        double err = 0;
        for (size_t l = 0; l < DECON_LEVELS; ++l) {
            err += decon[t + decon_offset][l] - decon_trimmed[t][l];
        }
        if (err > TEST_TOL) LOG4_DEBUG("Offset decon error " << err << " at " << t);
    }

    const auto trans_decon = svr::common::transpose_matrix(decon);
    for (size_t l = 0; l < trans_decon.size(); ++l) {
        LOG4_DEBUG("Power level " << l << " " << svr::common::meanabs(trans_decon[l]));
    }

    std::vector<double> decon_flat(decon.size() * decon[0].size());
    __omp_pfor(t, 0, decon.size(),
        for (size_t l = 0; l < decon[t].size(); ++l)
        decon_flat[t + l * decon.size()] = decon[t][l];
    )
    p_cvmd_transformer->inverse_transform(decon_flat, recon, 0);
    double total_diff = 0;
    double highest_diff = 0;
    for (size_t i = 0; i < recon.size(); ++i) {
        const auto diff = std::abs(input[i] - recon[i]);
        if (diff > highest_diff) highest_diff = diff;
        if (diff > TEST_TOL) LOG4_WARN("Input value " << input[i] << " differs " << diff << " from recon value " << recon[i] << " at index " << i);
        total_diff += diff;
    }
    LOG4_INFO("Test took " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << " seconds, total reconstruction error " << total_diff <<
                ", average error " << total_diff / double(input.size() - decon_offset) << ", highest diff " << highest_diff);
}

}