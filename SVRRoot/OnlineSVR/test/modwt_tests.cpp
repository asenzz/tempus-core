//#include <gtest/gtest.h>
//#include "test_harness.hpp"
//#include <modwt_transform.hpp>
//#include <fstream>
//#include <chrono>
//#include <iostream>
//
//
//TEST(modwt, spectral_transform_test) //this test is no more valid, it should be modified
//{
//   std::vector<double> decon_input, recon, input, diff, input0 = read_test_data("modwt_test_data/inputqueue_long.txt");
//   ASSERT_FALSE(input0.empty());
//
//   std::vector<std::vector<double>> deconstructed;
//
//   std::string filterorder[5] = {"fk4", "fk6", "fk8", "fk14", "fk22"};
//
//   for(int test=4; test<=5; test++){
//
//        int dlv = test;
//
//        printf("\ndecomposition levels: %d \n", dlv);
//
//        for(int k=0; k<5; k++){
//
//             int filter_order = svr::common::modwt_filter_order_from(filterorder[k]);
//             int wvlen = filter_order * pow(2,dlv+1);
//             for(int i=0; i<wvlen; ++i)
//                 input.push_back( input0[i] );
//
//             printf("filter order: %d\n", filter_order);
//
//             std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(filterorder[k], dlv));
//             ASSERT_FALSE(!transformer);
//
//             transformer->transform(input, deconstructed, 0/*not used with modwt*/);
//             ASSERT_TRUE(!deconstructed.empty());
//
//             //linearize deconstructed matrix for inverse transform
//             for( auto i : deconstructed ){
//                 for(auto j : i)
//                     decon_input.push_back(j);
//             }
//
//             transformer->inverse_transform(decon_input, recon, 0/*not used with modwt*/);
//
//             ASSERT_TRUE(!recon.empty());
//             ASSERT_TRUE(input.size() == recon.size());
//
//             size_t q = 0; //input.size()-15;
//             for(; q < input.size(); ++q)
//                 diff.push_back( input[q]-recon[q] );
//
//             double min = *std::min_element( diff.begin(), diff.end() );
//             printf(" min element: %f\n", min );
//
//             input.clear();
//             deconstructed.clear();
//             decon_input.clear();
//             recon.clear();
//
//        }
//   }
//}
//
//namespace {
//
//std::string get_modwt_name(std::string filterorder, size_t levels)
//{
//    std::ostringstream ostr;
//    ostr << filterorder << "x" << levels;
//    return ostr.str();
//}
//
//void test_deconstruction(std::string filterorder, size_t levels)
//{
//    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(filterorder, levels));
//    ASSERT_NE(transformer.get(), nullptr);
//
//    auto const modwt_name = get_modwt_name(filterorder, levels);
//
//    std::vector<double> input = read_test_data("stft_test_data/input_queue_etalon.txt");
//    std::vector<std::vector<double>> decon;
//
//    transformer->transform(input, decon, 0);
//
////    std::ofstream de("deconstructed_etalon.txt" + modwt_name);
////    de.precision(std::numeric_limits< double >::max_digits10);
////    pretty_print(de, decon);
//
//    auto f = open_data_file("modwt_test_data/deconstructed_etalon.txt" + modwt_name);
//
//    ASSERT_TRUE(f.good());
//
//    for(size_t i = 0; i < decon.size(); ++i)
//        for(size_t j = 0; j < decon[i].size(); ++j)
//        {
//            double val; f >> val;
//            ASSERT_FP_EQ(decon[i][j], val) << " row: " << i << " col: " << j;
//        }
//
//    ASSERT_TRUE(f.eof());
//}
//
//void test_reconstruction(std::string filterorder, size_t levels, size_t minimal_data_length, size_t enough_data_length)
//{
//    auto const modwt_name = get_modwt_name(filterorder, levels);
//
//    std::vector<double> deconstructed_etalon = read_test_data("modwt_test_data/deconstructed_etalon.txt" + modwt_name);
//
//    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(filterorder, levels));
//    ASSERT_NE(transformer.get(), nullptr);
//    ASSERT_EQ(transformer->get_minimal_input_length(0, 15), minimal_data_length);
//    ASSERT_EQ(transformer->get_minimal_input_length(65000, 15), enough_data_length);
//
//    std::vector<double> decon_linearized(deconstructed_etalon.size());
//
//    size_t const decon_size = deconstructed_etalon.size() / (levels+1);
//
//    for (size_t level = 0; level <= levels; ++level)
//        for (size_t row = 0; row < decon_size; ++row)
//            decon_linearized[row + level * decon_size] = deconstructed_etalon[row * (levels + 1) + level];
//
//
//    std::vector<double> recon;
//
//    transformer->inverse_transform(decon_linearized, recon, 0);
//
//    auto f = open_data_file("stft_test_data/input_queue_etalon.txt");
//
//    ASSERT_TRUE(f.good());
//
//    for(size_t i = 0; i < recon.size(); ++i)
//    {
//        double val; f >> val;
//        ASSERT_FP_EQ(recon[i], val) << " at: " << i;
//    }
//
//    ASSERT_TRUE(f.eof());
//}
//
//}
//
//TEST(modwt_fk4x2, deconstruction)
//{
//    test_deconstruction("fk4", 2);
//}
//
//TEST(modwt_fk4x2, reconstruction)
//{
//    test_reconstruction("fk4", 2, 32+12UL, 65015+12UL);
//}
//
//TEST(modwt_fk4x8, deconstruction)
//{
//    test_deconstruction("fk4", 8);
//}
//
//TEST(modwt_fk4x8, reconstruction)
//{
//    test_reconstruction("fk4", 8, 4UL*pow(2, 9) + 1.5*pow(2,9), 65015UL + 1.5*pow(2,9));
//}
//
//TEST(modwt_fk8x8, deconstruction)
//{
//    test_deconstruction("fk8", 8);
//}
//
//TEST(modwt_fk8x8, reconstruction)
//{
//    test_reconstruction("fk8", 8, 8*pow(2, 9) + 1.5*pow(2,9), 65015+768UL );
//}
//
//TEST(modwt, decon_recon)
//{
//    std::string const filterorder = "fk8";
////    int fo=8;
//    size_t const levels = 8;
////    int wavelen = fo*pow(2, levels+1);
//
//    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(filterorder, levels));
//    ASSERT_NE(transformer, nullptr);
//
////    std::vector<double> input = read_length_of_test_data("stft_test_data/input_queue_etalon.txt", wavelen);
//    std::vector<double> input = read_test_data("stft_test_data/input_queue_etalon.txt");
//
//    std::vector<std::vector<double>> decon;
//    std::vector<double> recon, diff;
//
//    transformer->transform(input, decon, 0);
//
//    //==   copy decon to deconstructed_etalon as required for train
//    std::vector<double> deconstructed_etalon;
//    for(size_t i=0; i < decon.size(); ++i)
//        for(size_t j=0; j < decon[i].size(); ++j)
//        deconstructed_etalon.push_back(decon[i][j]);
//
//    std::vector<double> decon_linearized(deconstructed_etalon.size());
//    size_t const decon_size = deconstructed_etalon.size() / (levels+1);
//
//    //linearize
//    for (size_t level = 0; level <= levels; ++level)
//        for (size_t row = 0; row < decon_size; ++row)
//            decon_linearized[row + level * decon_size] = deconstructed_etalon[row * (levels + 1) + level];
//   //==
//
//    transformer->inverse_transform(decon_linearized, recon, 0/*not used with modwt*/);
//
//    ASSERT_TRUE(!recon.empty());
//
//    ASSERT_TRUE(input.size() == recon.size());
//
//    for(size_t k = 0; k < input.size(); ++k)
//        diff.push_back( std::fabs(input[k]-recon[k]) );
//
//    auto const min = *std::min_element( diff.begin(), diff.end() )
//             , max = *std::max_element( diff.begin(), diff.end() );
//
//    std::cout << "\nmin element: " << min;
//    std::cout << "\nmax element: " <<  max << std::endl<<std::endl;
//
//    ASSERT_LT(max, 1e-5);
//
//}
//
//TEST(modwt, direct_deconstruct_reconstruct)
//{
//    std::string const filterorder = "fk8";
//    int fo=8;
//    size_t const levels = 8;
////    int wavelen = fo*pow(2, levels+1);
//
//    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(filterorder, levels));
//    ASSERT_NE(transformer, nullptr);
//
////    std::vector<double> input = read_length_of_test_data("stft_test_data/input_queue_etalon.txt", wavelen);
//    std::vector<double> input = read_test_data("stft_test_data/input_queue_etalon.txt");
//
//    std::vector<std::vector<double>> decon;
//    std::vector<double> recon, diff;
//
//    svr::modwt_transform* mwt = dynamic_cast<svr::modwt_transform*>(transformer.get());
//    ASSERT_TRUE(mwt);
//
//    mwt->deconstruct1(fo, levels, input, decon);
//
//    //==
//    std::vector<double> deconstructed_etalon;
//    for(size_t i=0; i < decon.size(); ++i)
//        for(size_t j=0; j < decon[i].size(); ++j)
//        deconstructed_etalon.push_back(decon[i][j]);
//
//    std::vector<double> decon_linearized(deconstructed_etalon.size());
//
//    size_t const decon_size = deconstructed_etalon.size() / (levels+1);
//
//    for (size_t level = 0; level <= levels; ++level)
//        for (size_t row = 0; row < decon_size; ++row)
//            decon_linearized[row + level * decon_size] = deconstructed_etalon[row * (levels + 1) + level];
//
//   decon.clear();
//   decon.push_back(decon_linearized);
//   //==
//
//    mwt->reconstruct1(fo, levels, decon, recon);
//
//    ASSERT_TRUE(!recon.empty());
//
//    ASSERT_TRUE(input.size() == recon.size());
//
//    for(size_t k = 0; k < input.size(); ++k)
//        diff.push_back( fabs(input[k]-recon[k]) );
//
//    auto const max = *std::max_element( diff.begin(), diff.end() );
//    std::cout << "min element: " << max << std::endl;
//
//    ASSERT_LT(max, 1e-5);
//}