//
// Created by sstoyanov on 3/15/18.
//
#include <gtest/gtest.h>
#include <fstream>
#include "../include/spectral_transform.hpp"
#include "../include/online_emd.hpp"
#include <chrono>
#include "DeconQueueService.hpp"
#include "test_harness.hpp"
#include "util/string_utils.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif

using namespace svr;

namespace {

// constexpr double oemd_epsilon = 0.0000011;
// constexpr double oemd_mask_sum = 1.00000;

const std::string transform_input_filename = "../SVRRoot/OnlineSVR/test/online_emd_test_data/transform_input_eurusd_avg_1_cvmd_level_0";
const std::string transform_result_filename = "online_emd_test_data/transform_output_real";
const std::string inv_transform_input_filename = "online_emd_test_data/inverse_transform_input";
const std::string inv_transform_result_filename = "online_emd_test_data/inverse_transform_output";

constexpr unsigned DECON_OFFSET  = 1500;
constexpr unsigned TEST_OEMD_PADDING = 1024;
constexpr unsigned INPUT_LIMIT = 64e4; // std::numeric_limits<size_t>::max()
constexpr unsigned TEST_LEVELS = MIN_LEVEL_COUNT;
const std::string TEST_QUEUE_NAME("test_queue");
constexpr double TEST_STRETCH_COEF = 1;
constexpr double TEST_ERROR_THRESHOLD = 1e-13;
const auto TEST_RES = bpt::hours(1);

class onlineEmdTransformTest : public testing::Test
{
public:
    onlineEmdTransformTest()
    = default;

    ~onlineEmdTransformTest() override = default;

    using testing::Test::SetUp;

    virtual void SetUp(const std::string &input_filename, const std::string &exp_output_filename, const unsigned levels)
    {
        transformer = std::make_unique<svr::oemd::online_emd>(levels, TEST_STRETCH_COEF);
        std::ifstream input_f(input_filename);
        std::string line;
#ifdef EMD_ONLY
        input = std::make_shared<datamodel::DeconQueue>(TEST_QUEUE_NAME, "input_table", "input_column", 0, levels);
        std::vector<double> vals(levels);
#else
        input = std::make_shared<datamodel::DeconQueue>(TEST_QUEUE_NAME, "input_table", "input_column", 0, 4 * levels);
        std::vector<double> vals(4 * levels);
#endif
        const auto time_now = bpt::second_clock::local_time();
        auto time_iter = time_now;
        EXPECT_TRUE(input_f && input_f.good());
        while (!input_f.eof()) {
            std::getline(input_f, line);
            if (line.empty()) continue;
#ifdef EMD_ONLY
            vals[levels] = std::strtod(line.c_str(), nullptr);
#else
            vals[levels * 2] = std::strtod(line.c_str(), nullptr);
#endif
            input->get_data().emplace_back(std::make_shared<datamodel::DataRow>(time_iter, time_now, 1., vals));
            time_iter += TEST_RES;
            if (input->size() > INPUT_LIMIT) break;
        }
        result_f.open(exp_output_filename);
        EXPECT_TRUE(result_f.good());
        LOG4_DEBUG("Read " << input->size() << " values.");
    }

    //virtual void TearDown(){}
    datamodel::DeconQueue_ptr input;
    std::ifstream result_f;
    std::unique_ptr<svr::oemd::online_emd> transformer;
};


#ifdef EXPERIMENTAL_FEATURES

TEST_F(onlineEmdTransformTest, show_mask)
{
    SetUp(transform_input_filename, transform_result_filename);
    const auto oemd_levels = online_emd::oemd_coefs_cache.find({TEST_LEVELS, TEST_QUEUE_NAME});
    if (oemd_levels == online_emd::oemd_coefs_cache.end())
        THROW_EX_FS(std::runtime_error, "Inappropriate number chosen for level parameter " << TEST_LEVELS);
    matplotlibcpp::plot(oemd_levels->second.masks.back(), {{"label", "Level 0 masks"}});
    // const auto new_mask = svr::common::stretch_cut_mask<double>(oemd_levels->second.masks.back(), 60, 10);
    // matplotlibcpp::plot(new_mask, {{"label", "Level 0 normalized modified masks"}});
    matplotlibcpp::legend();
    matplotlibcpp::show();
}

#endif

TEST_F(onlineEmdTransformTest, test_transform_correctness)
{
    SetUp(transform_input_filename, transform_result_filename, TEST_LEVELS);
    std::vector<double> tail;
    if (TEST_OEMD_PADDING) {
        business::DeconQueueService::mirror_tail(datamodel::datarow_crange{input->get_data()}, input->size() + TEST_OEMD_PADDING, tail, 0);
        OMP_FOR_i(TEST_OEMD_PADDING)
            if (tail[tail.size() - i - 1] != input->at(i)->at(2 * TEST_LEVELS))
                LOG4_WARN("Tail mismatch at " << i << " " << tail[tail.size() - i] << " != " << input->at(i)->at(2 * TEST_LEVELS));
    }

    const auto residuals_len = oemd::online_emd::get_residuals_length();

    const auto masks_siftings = transformer->get_masks(datamodel::datarow_crange{input->get_data()}, tail, input->get_table_name(), 0, std::identity(), TEST_RES, TEST_RES);
OMP_FOR_i_(masks_siftings->masks.size(), ordered) {
        const auto meanabs_mask = common::meanabs(masks_siftings->masks[i]);
        const auto sum_mask = common::sum(masks_siftings->masks[i]);
#pragma omp ordered
        LOG4_DEBUG("Power level of mask " << i << " " << meanabs_mask << ", sum " << sum_mask << " should be one!");
    }

    PROFILE_INFO(transformer->transform(*input, 0, 0, 0, TEST_RES, TEST_RES), "OEMD transformation.");

    auto delayed_input = input->clone(DECON_OFFSET);
    delayed_input->set_table_name(input->get_table_name());
    LOG4_DEBUG("Input size " << input->size() << ", delayed input size " << delayed_input->size() << ", residuals " << residuals_len << ", tail " << tail.size());
    PROFILE_INFO(transformer->transform(*delayed_input, 0, 0, 0, TEST_RES, TEST_RES), "OEMD transformation with " << DECON_OFFSET << " offset.");

    LOG4_DEBUG("Input queue: " << *input);
    LOG4_DEBUG("Delayed input queue: " << *delayed_input);
    // LOG4_FILE("/tmp/input.csv", input->data_to_string(input->size()));
    // LOG4_FILE("/tmp/delayed_input.csv", delayed_input->data_to_string(delayed_input->size()));

    const size_t start_t = residuals_len + DECON_OFFSET > input->size() ? 0 : residuals_len + DECON_OFFSET;
    const auto num_t = input->size() - start_t;
    std::deque<std::vector<double>> decon(TEST_LEVELS, std::vector<double>(num_t));
    std::vector<double> invec(num_t);
    bool test_result = true;
    double total_recon_diff = 0, total_recon_diff_off = 0;
    std::deque<double> total_delayed_diff_level(TEST_LEVELS, 0);
    tbb::mutex res_l;
    OMP_FOR(num_t)
    for (size_t t = start_t; t < input->size(); ++t) {
        if (t < DECON_OFFSET) {
            LOG4_TRACE("Skipping " << t << " < " << DECON_OFFSET);
            continue;
        }
        const auto off_t = t - DECON_OFFSET;
        double recon = 0, off_recon = 0;
        UNROLL(TEST_LEVELS)
        for (size_t l = 0; l < TEST_LEVELS; ++l) {
            const auto v = input->at(t)->at(2 * l);
            const auto off_v = delayed_input->at(off_t)->at(2 * l);
            if (!common::isnormalz(v) || !common::isnormalz(off_v)) {
                LOG4_ERROR("Decon at " << t << "x" << l << " not normal " << v);
                const tbb::mutex::scoped_lock lk(res_l);
                test_result = false;
            }
            decon[l][t - start_t] = v;
            recon += v;
            off_recon += off_v;
            const double diff_off = std::abs(v - off_v);
            if (diff_off > TEST_ERROR_THRESHOLD) {
                LOG4_ERROR("Offset decon difference " << diff_off << " at " << t << ", offset " << off_t << ", level " << l << ", val " << v << ", off val " << off_v);
                const tbb::mutex::scoped_lock lk(res_l);
                test_result = false;
            }

            const tbb::mutex::scoped_lock lk(res_l);
            total_delayed_diff_level[l] += diff_off;
        }
        invec[t - start_t] = input->at(t)->at(2 * TEST_LEVELS);
        const double recon_diff_off = std::abs(recon - off_recon);
        if (recon_diff_off > TEST_ERROR_THRESHOLD) {
            LOG4_WARN("Offset recon difference " << recon_diff_off << " at " << t << ", offset " << off_t << ", recon " << recon << ", off recon " << off_recon);
            const tbb::mutex::scoped_lock lk(res_l);
            test_result = false;
        }
        const auto in_v = input->at(t)->at(TEST_LEVELS * 2);
        if (!common::isnormalz(in_v) || !common::isnormalz(recon) || !common::isnormalz(off_recon)) {
            LOG4_WARN("Not zero-normal, position " << t << ". delayed position " << off_t << ", recon " << recon << ", input " << in_v << ", offset recon " << off_recon);
            const tbb::mutex::scoped_lock lk(res_l);
            test_result = false;
        }
        const auto recon_diff = std::abs(recon - in_v);
        if (recon_diff > TEST_ERROR_THRESHOLD) {
            LOG4_ERROR("Different, position " << t << ", delayed position " << off_t << ", recon " << recon << ", input " << in_v << ", recon diff " << recon_diff);
            const tbb::mutex::scoped_lock lk(res_l);
            test_result = false;
        }
        const tbb::mutex::scoped_lock lk(res_l);
        total_recon_diff += recon_diff;
        total_recon_diff_off += recon_diff_off;
    }
    auto mean_delayed_diff_level = total_delayed_diff_level;
    std::transform(mean_delayed_diff_level.begin(), mean_delayed_diff_level.end(), mean_delayed_diff_level.begin(), [num_t](const double val) { return val / num_t; });
    LOG4_DEBUG("Power of input " << svr::common::meanabs(invec) << ", validation window " << num_t << ", total recon error " << total_recon_diff << ", mean recon error " <<
        total_recon_diff / num_t << ", total offset recon error " << total_recon_diff_off << ", mean offset recon error " << total_recon_diff_off / num_t <<
        ", total delayed recon per-level error " << common::to_string(total_delayed_diff_level) << ", mean delayed recon per-level error " << common::to_string(mean_delayed_diff_level));
    OMP_FOR_(decon.size(), ordered)
    for (size_t l = 0; l < decon.size(); ++l) {
        const auto mean_abs_l = svr::common::meanabs(decon[l]);
#pragma omp ordered
        LOG4_DEBUG("Power of decon level " << l << " " << mean_abs_l);
#ifdef EXPERIMENTAL_FEATURES
        matplotlibcpp::plot(decon[l], {{"label", "Level " + std::to_string(l)}});
#endif
    }

#ifdef EXPERIMENTAL_FEATURES
    matplotlibcpp::legend();
    matplotlibcpp::show();
#endif
    EXPECT_TRUE(test_result);
}

TEST_F(onlineEmdTransformTest, TestBadLevels)
{
    ASSERT_THROW(SetUp(transform_input_filename, transform_result_filename, 10), std::runtime_error);
    //std::vector<std::vector<double>> decon;
    //ASSERT_THROW(transformer->transform(this->input, decon), std::runtime_error);

}

TEST_F(onlineEmdTransformTest, TestGoodLevels)
{
    ASSERT_NO_THROW(SetUp(transform_input_filename, transform_result_filename, 11));
    std::vector<std::vector<double>> decon;
    //transformer->transform(this->input, decon, 0);
}

TEST(OnlineEmdTestCoefficients, TestMasksSum)
{
#if 0
    double sum;
    for(auto oemd_coef : svr::oemd_coefs_cache) {
        for (size_t i = 0; i < oemd_coef.first.first - 1; ++i) {
            sum = 0.;
            for (size_t j = 0; j < oemd_coef.second->masks[i].size(); ++j)
                sum += oemd_coef.second->masks[i][j];
//	    std::cout.precision(std::numeric_limits< double >::max_digits10);
//        std::cout << "SUM of level " << i << " = " << std::fixed << sum << std::endl;
            ASSERT_NEAR(sum, svr::oemd_mask_sum, svr::oemd_epsilon) << "\nError at level " << i << std::endl;
//        if (sum - svr::oemd_mask_sum > svr::oemd_epsilon)
//            throw std::logic_error(svr::common::formatter() << "Invalid masks data at level" << i);
        }
    }
#endif
}

//TEST(OnlineEmdTestMatrix, TestMatrixIncreasingLevels)
//{
//   LOG4_DEBUG("From the optimisation, this test is not relevant.");
//    size_t upto_level = 3;
//    for (size_t i = 0; i < upto_level; ++i) {
//        for (size_t j = 0; j < svr::oemd_mask_size[i] - 1; ++j) {
//            ASSERT_LE(svr::oemd_mask[i][j], svr::oemd_mask[i][j + 1]);
//        }
//    }
//}


class onlineEmdInvTransformTest : public testing::Test
{
public:
    onlineEmdInvTransformTest() : input(1024 * 9)
    {
        omp_set_nested(true);
        omp_set_max_active_levels((int) C_n_cpu);
        this->transformer = svr::spectral_transform::create(std::string("oemd"), 8);
    }

    ~onlineEmdInvTransformTest() = default;

    using testing::Test::SetUp;

    virtual void SetUp(const std::string &input_filename, const std::string &exp_output_filename)
    {
        auto input_f = open_data_file(input_filename);
        EXPECT_TRUE(input_f.good());

        for (size_t i = 0; i < input.size(); ++i) {
            input_f >> input[i];
        }

        double new_line;
        input_f >> new_line; //skip new line at the end of input file
        EXPECT_TRUE(input_f.eof());

        this->result_f = open_data_file(exp_output_filename);
        EXPECT_TRUE(result_f.good());


    }

    //virtual void TearDown(){}
    std::vector<double> input;
    std::ifstream result_f;
    std::unique_ptr<svr::spectral_transform> transformer;
};

TEST_F(onlineEmdInvTransformTest, test_inverse_transform_correctnes)
{
    this->SetUp(inv_transform_input_filename, inv_transform_result_filename);

    std::vector<double> recon;
    auto start = std::chrono::steady_clock::now();

    transformer->inverse_transform(input, recon, 0);
    std::cout << "Inverse transformation took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";

    double val;
    for (size_t i = 0; i < recon.size(); ++i) {
        //std::cout << "RECON[" << i << "] = " << recon[i] << std::endl;
        result_f >> val;
        //ASSERT_DOUBLE_EQ(decon[i][j], val);
        ASSERT_NEAR(val, recon[i], 0.00001);
    }

    result_f >> val;
    EXPECT_TRUE(result_f.eof());
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}

}

