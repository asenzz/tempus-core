#include <gtest/gtest.h>
#include <onlinesvr.hpp>
#include <sstream>
#include <util/CompressionUtils.hpp>
#include <spectral_transform.hpp>
#include "test_harness.hpp"
#include <fstream>

namespace
{
//    using TheMatrix = svr::datamodel::vmatrix<double>;
//    using TheVector = svr::datamodel::vektor<double>;

//    TheVector * fill( size_t n)
//    {
//        TheVector * result = new vektor<double>();
//        for(size_t i = 0; i < n; ++i)
//            result->add(0.3 + i);
//        return result;
//    }
//
//    void fill(TheMatrix * m1, size_t n, size_t m)
//    {
//        for(size_t i = 0; i < n; ++i)
//        {
//            auto row = new vektor<double>();
//            for(size_t j = 0; j < m; ++j)
//                row->add(0.4 + i * j);
//
//            m1->add_row_ref(row);
//        }
//    }

}


// Tests reconstruction error for the short-time fourier transform
TEST(OnlineSVRTests, STFT_Reconstruction)
{
    std::vector<double> input_large = read_test_data("stft_test_data/input_queue_etalon.txt");

    for (size_t input_size = 1000; input_size < std::min(size_t(10000), input_large.size() - 10000); input_size *= 2)
    {
        for (size_t offset = 0; offset <= 10000; offset += 4177)
        {
            std::vector<double> input;
            input.insert(input.begin(), input_large.begin() + offset, input_large.begin() + input_size + offset);
            for (int levels_2pow = 4; levels_2pow <= 5; ++levels_2pow)
            {
                int nLevels = 1;
                for (int i = 0; i < levels_2pow; ++i)
                {
                    nLevels *= 2;
                }
                nLevels -= 1;

                // deconstruct
                std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(std::string("stft_cpu"),nLevels));
                std::vector<std::vector<double>> decon;
                transformer->transform(input, decon, 0);
                EXPECT_TRUE(decon.size() == input.size() - nLevels);
                for (size_t i = 0; i < decon.size(); ++i)
                {
                    EXPECT_TRUE(int(decon[i].size()) == nLevels + 1);
                }
                std::vector<double> decon_linearized(decon.size() * (nLevels + 1));
                for (int level = 0; level < nLevels + 1; ++level)
                {
                    for (size_t row = 0; row < decon.size(); ++row)
                    {
                        decon_linearized[row + level*decon.size()] = decon[row][level];
                    }
                }

                std::vector<double> recon;
                transformer->inverse_transform(decon_linearized, recon, 0);

                EXPECT_TRUE(recon.size() == input.size());
                double max_residual = 0;
                for (size_t i = 0; i < input.size(); ++i)
                {
                    auto residual = fabs(recon[i] - input[i]);
                    if (residual > max_residual) max_residual = residual;
                    ASSERT_LE(max_residual, std::numeric_limits<float>::epsilon());
                }

            }
        }
    }

}

TEST(OnlineSVRTests, STFT_Stationarity) {

    std::vector<double> input_large = read_test_data("stft_test_data/input_queue_etalon.txt");

    size_t input_size = 3000;
    {
        for (size_t offset = 0; offset <= 79; offset += 17)
        {
            std::vector<double> input1, input2;
            input1.insert(input1.begin(), input_large.begin(), input_large.begin() + input_size);
            input2.insert(input2.begin(), input_large.begin() + offset, input_large.begin() + input_size + offset);
            for (int levels_2pow = 4; levels_2pow <= 6; ++levels_2pow)
            {
                int nLevels = 1;
                for (int i = 0; i < levels_2pow; ++i)
                {
                    nLevels *= 2;
                }
                nLevels -= 1;

                // deconstruct
                std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(std::string("stft_cpu"),nLevels));
                std::vector<std::vector<double>> decon1;
                transformer->transform(input1, decon1, 0);

                std::unique_ptr<svr::spectral_transform> transformer2(svr::spectral_transform::create(std::string("stft_cpu"),nLevels));
                std::vector<std::vector<double>> decon2;
                transformer2->transform(input2, decon2, 0);

                for (int level = 0; level < nLevels + 1; ++level)
                {
                    for (size_t row = offset; row < decon1.size(); ++row)
                    {
                        ASSERT_LE(decon1[row][level] - decon2[row-offset][level], 1e-10);
                    }
                }
            }
        }
    }
}
