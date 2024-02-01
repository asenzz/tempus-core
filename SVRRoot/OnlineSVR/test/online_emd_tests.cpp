//
// Created by sstoyanov on 3/15/18.
//
#include <gtest/gtest.h>
#include <fstream>
#include "../include/spectral_transform.hpp"
#include "../include/online_emd.hpp"
#include <chrono>
#include "test_harness.hpp"
#include "util/string_utils.hpp"
#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif

using namespace svr;

namespace{


const std::string transform_input_filename = "../SVRRoot/OnlineSVR/test/online_emd_test_data/transform_input_eurusd_avg_1_cvmd_level_0";
const std::string transform_result_filename = "online_emd_test_data/transform_output_real";
const std::string inv_transform_input_filename = "online_emd_test_data/inverse_transform_input";
const std::string inv_transform_result_filename = "online_emd_test_data/inverse_transform_output";

#define DECON_OFFSET 1500
#define TEST_OEMD_PADDING 1024
#define INPUT_LIMIT std::numeric_limits<size_t>::max()
#define TEST_LEVELS 16
#define TEST_QUEUE_NAME "test_queue"
#define TEST_STRETCH_COEF 1
#define TEST_ERROR_THRESHOLD 1e-14

class onlineEmdTransformTest : public testing::Test {
 public:
  onlineEmdTransformTest() {
  }
  ~onlineEmdTransformTest() = default;

  virtual void SetUp(const std::string &input_filename, const std::string &exp_output_filename, const size_t levels = TEST_LEVELS)
  {
      transformer = std::make_unique<svr::online_emd>(levels, TEST_STRETCH_COEF);
      std::ifstream input_f(input_filename);
      std::string line;
      input = std::make_shared<datamodel::DeconQueue>(TEST_QUEUE_NAME, "input_table", "input_column", 0, 2 * TEST_LEVELS);
      std::vector<double> vals(2 * TEST_LEVELS);
      const auto time_now = bpt::second_clock::local_time();
      auto time_iter = time_now;
      EXPECT_TRUE(input_f && input_f.good());
      while (!input_f.eof()) {
          std::getline(input_f, line);
          if (line.empty()) continue;
          vals[TEST_LEVELS] = std::atof(line.c_str());
          input->get_data().emplace_back(std::make_shared<datamodel::DataRow>(time_now, time_iter, 0., vals));
          time_iter += bpt::hours(1);
          if (input->get_data().size() > INPUT_LIMIT) break;
      }
      result_f.open(exp_output_filename);
      EXPECT_TRUE(result_f.good());
      LOG4_DEBUG("Read " << input->get_data().size() << " values.");
  }

  //virtual void TearDown(){}
  datamodel::DeconQueue_ptr input;
  std::ifstream result_f;
  std::unique_ptr<svr::online_emd> transformer;
};


TEST_F(onlineEmdTransformTest, show_mask)
{
    SetUp(transform_input_filename, transform_result_filename);
#if 0
    const auto oemd_levels = online_emd::oemd_coefs_cache.find({TEST_LEVELS, TEST_QUEUE_NAME});
    if (oemd_levels == online_emd::oemd_coefs_cache.end())
        THROW_EX_FS(std::runtime_error, "Inappropriate number chosen for level parameter " << TEST_LEVELS);
#endif
#ifdef EXPERIMENTAL_FEATURES
    matplotlibcpp::plot(oemd_levels->second.masks.back(), {{"label", "Level 0 masks"}});
    // const auto new_mask = svr::common::stretch_cut_mask<double>(oemd_levels->second.masks.back(), 60, 10);
    // matplotlibcpp::plot(new_mask, {{"label", "Level 0 normalized modified masks"}});
    matplotlibcpp::legend();
    matplotlibcpp::show();
#endif
}

TEST_F(onlineEmdTransformTest, test_transform_correctness)
{
    SetUp(transform_input_filename, transform_result_filename);
    std::vector<double> tail;
    if (TEST_OEMD_PADDING) spectral_transform::mirror_tail(datamodel::datarow_range{input->get_data()}, input->get_data().size() + TEST_OEMD_PADDING, tail);
    const auto residuals_len = transformer->get_residuals_length();
    PROFILE_EXEC_TIME(transformer->transform(input), "OEMD transformation.");

    auto delayed_input = std::make_shared<datamodel::DeconQueue>(*input);
    delayed_input->get_data().erase(input->get_data().begin(), input->get_data().begin() + DECON_OFFSET);
    PROFILE_EXEC_TIME(transformer->transform(delayed_input), "OEMD transformation " << DECON_OFFSET << " offset.");
#if 0
    double val;
    for (size_t i = 0; i < decon.size(); ++i) {
        for (size_t j = 0; j < decon[i].size(); ++j) {
            this->result_f >> val;
            //ASSERT_DOUBLE_EQ(decon[i][j], val);
	    //Remove this when fixed the test files
            //EXPECT_NEAR (val, decon[i][j], 0.00001);
        }
        //myfile << std::endl;
    }
    //this->result_f >> val;
    //ASSERT_TRUE(result_f.eof());
#endif
    LOG4_TRACE("Residuals length " << residuals_len);
    const size_t start_t = residuals_len > input->get_data().size() ? input->get_data().size() - DECON_OFFSET : residuals_len;
    std::vector<std::vector<double>> decon(TEST_LEVELS, std::vector<double>(input->get_data().size() - start_t));
#pragma omp parallel for
    for (size_t t = start_t; t < input->get_data().size(); ++t) {
        double recon = 0;
        for (size_t l = 0; l < TEST_LEVELS; ++l) {
            if (!svr::common::isnormalz(input->get_data()[t]->get_value(l)))
                LOG4_WARN("Decon number " << t << "x" << l << " not normal " << input->get_data()[t]->get_value(l));
            decon[l][t - start_t] = input->get_data()[t]->get_value(l);
            recon += input->get_data()[t]->get_value(l);
        }
        const size_t off_t = t >= DECON_OFFSET ? t - DECON_OFFSET : 0;
        if (!svr::common::isnormalz(delayed_input->get_data()[off_t]->get_value(TEST_LEVELS))) {
            LOG4_WARN("Recon " << recon << " or padded input " << delayed_input->get_data()[off_t]->get_value(TEST_LEVELS) << " at " << t << " is not normal.");
        } else {
            if (!svr::common::isnormalz(recon))
                LOG4_WARN("Recon " << recon << " or padded input " << delayed_input->get_data()[off_t]->get_value(TEST_LEVELS) << " at " << t << " is not normal.");
        }
        if (std::abs(recon - delayed_input->get_data()[off_t]->get_value(TEST_LEVELS)) > TEST_ERROR_THRESHOLD)
            LOG4_WARN("Recon differs from delayed " << off_t << " input " << t << ", " << recon - delayed_input->get_data()[off_t]->get_value(TEST_LEVELS));

        // Time invariance check
        if (t < residuals_len + DECON_OFFSET) continue;
        double diff = 0;
        for (size_t l = 0; l < TEST_LEVELS; ++l)
            diff += std::abs(input->get_data()[t]->get_value(l) - delayed_input->get_data()[off_t]->get_value(l));
        if (diff > TEST_ERROR_THRESHOLD)
            LOG4_WARN("Decon differ at " << t << " " << diff);
    }

    for (size_t l = 0; l < decon.size(); ++l) { // Don't parallelize
        LOG4_DEBUG("Power level " << l << " " << svr::common::meanabs(decon[l]));
        if (l > 2) continue;
#ifdef EXPERIMENTAL_FEATURES
        matplotlibcpp::plot(trans_decon[l], {{"label", "Level " + std::to_string(l)}});
#endif
    }

#ifdef EXPERIMENTAL_FEATURES
    matplotlibcpp::legend();
    matplotlibcpp::show();
#endif
}

TEST_F(onlineEmdTransformTest, TestBadLevels){
    ASSERT_THROW(SetUp(transform_input_filename, transform_result_filename, 10),std::runtime_error);
    //std::vector<std::vector<double>> decon;
    //ASSERT_THROW(transformer->transform(this->input, decon), std::runtime_error);

}
TEST_F(onlineEmdTransformTest, TestGoodLevels){
    ASSERT_NO_THROW(SetUp(transform_input_filename, transform_result_filename, 11));
    std::vector<std::vector<double>> decon;
    //transformer->transform(this->input, decon, 0);

}

TEST(OnlineEmdTestCoefficients, TestMasksSum){
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


class onlineEmdInvTransformTest : public testing::Test {
 public:
  onlineEmdInvTransformTest() : input(1024*9){
      omp_set_nested(true);
      omp_set_max_active_levels((int) std::thread::hardware_concurrency());
      this->transformer =  svr::spectral_transform::create(std::string("oemd"), 8);
  }
  ~onlineEmdInvTransformTest() = default;

  virtual void SetUp(const std::string &input_filename, const std::string &exp_output_filename){
      auto input_f = open_data_file(input_filename);
      EXPECT_TRUE(input_f.good());

      for(size_t i = 0; i < input.size(); ++i){
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

TEST_F(onlineEmdInvTransformTest, test_inverse_transform_correctnes){
    this->SetUp(inv_transform_input_filename, inv_transform_result_filename);

    std::vector<double> recon;
    auto start = std::chrono::steady_clock::now();

    transformer->inverse_transform(input, recon, 0);
    std::cout << "Inverse transformation took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";

    double val;
    for(size_t i = 0; i < recon.size(); ++i)
    {
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

