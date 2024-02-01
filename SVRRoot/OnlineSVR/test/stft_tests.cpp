#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "../include/spectral_transform.hpp"
#include <fstream>
#include <chrono>

namespace {
void test_fwd_transform(std::string && transform)
{
    std::vector<double> input = read_test_data("stft_test_data/input_queue_etalon.txt");

    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(std::string(transform), 15));
    std::vector<std::vector<double>> decon;

    transformer->transform(input, decon, 0);

    return;

//    std::cout.precision(std::numeric_limits< double >::max_digits10);
//    for(auto const & frame : decon)
//    {
//        std::cout << " { ";
//        for(auto f : frame)
//        {
//            std::cout << f << " ";
//        }
//        std::cout << " } " << std::endl;
//    }

//    std::ofstream de("deconstructed_etalon.txt");
//    de.precision(std::numeric_limits< double >::max_digits10);
//    pretty_print(std::cout, decon);

    auto f = open_data_file("stft_test_data/deconstructed_etalon.txt");

    ASSERT_TRUE(f.good());

    for(size_t i = 0; i < decon.size(); ++i)
        for(size_t j = 0; j < decon[i].size(); ++j)
        {
            double val; f >> val;
            ASSERT_FP_EQ(decon[i][j], val) << " row: " << i << " col: " << j;
        }

    ASSERT_TRUE(f.eof());
}
}

TEST(stft_tests, stft_transform_direct)
{
    test_fwd_transform("stft");
    auto start = std::chrono::steady_clock::now();
    test_fwd_transform("stft");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}

TEST(stft_tests_cpu, stft_transform_direct)
{
    auto start = std::chrono::steady_clock::now();
    test_fwd_transform("stft_cpu");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}

TEST(stft_tests_ocl, stft_transform_direct)
{
    test_fwd_transform("stft_ocl");
    auto start = std::chrono::steady_clock::now();
    test_fwd_transform("stft_ocl");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}


//void boot()
//{
//    std::vector<double> const X{
//      16.90555, -0.000189999999998136, -0.000513298046158522, 0.000138591170101923, -0.000319497474683061, 0.000246776695296239, -0.000295032729925771, 0.000270719941782995, -0.000179999999999403, 0.00014999999999965, -0.000169403778213177, 0.000143440721169525, -0.000220502525316924, 0.00010677669529621, -0.00026226544570266, 1.13119494884529e-05
//    , 16.90589, -0.00014999999999965, -0.000213143062540146, 6.17232543178527e-05, -0.000159999999999402, 0.000188994949366153, -0.000232904383999859, 0.000145144376985688, -0.00014999999999965, 0.00016000000000016, -0.000197806294119083, 0.000102717970114531, -0.00016000000000003, 8.99494936674973e-06, -7.61462593411876e-05, 1.92968474466959e-05
//    , 16.9061, -6.00000000012813e-05, -2.65242779693033e-05, 5.58220533832683e-05, -9.82842712470336e-05, 0.000168994949366596, -0.000142861047441437, 3.43834567911666e-05, -0.00016000000000016, 6.00000000012813e-05, -9.95653594296979e-05, -2.80429500801212e-05, -4.17157287529953e-05, 2.89949493656786e-05, -0.000131049315159518, 3.33956465122425e-05
//    , 16.90623, -7.00000000009027e-05, 7.42369267055778e-05, 9.1171297056703e-05, -9.70710678116132e-05, 0.000141923881555181, -3.66878817650584e-05, 1.27592076358646e-06, -6.00000000012813e-05, -2.99999999997524e-05, 1.42614748940326e-05, 3.88495138925097e-05, -8.29289321877896e-05, 4.19238815545258e-05, -1.181051983429e-05, -3.12551098145338e-05
//    , 16.90629, 9.99999999962142e-06, 8.91290041996177e-05, 0.000135601543165006, -0.000126568542495439, 7.41421356245307e-05, 7.74236434324504e-06, 2.20258626339337e-05, 2.99999999997524e-05, -8.88178419700125e-16, -6.43109068382254e-05, 5.37415913865495e-05, -1.34314575054782e-05, -4.58578643762555e-05, -3.25604617046373e-05, 4.73172719177242e-05
//    , 16.90633, -4.99999999981071e-05, 6.74071800612402e-05, 0.000174695020852106, -0.000113639610307848, -8.78679656433107e-06, -2.07903181728352e-06, 5.25371259640134e-05, 8.88178419700125e-16, 7.00000000000145e-05, -4.03473750538733e-05, -4.30263658971551e-05, 1.36396103071927e-05, 5.12132034351738e-05, -2.49807731900834e-05, -4.08684710089602e-05
//    , 16.9064, -1.99999999992428e-05, 6.00947910819113e-05, 0.000213980605494342, -2.46446609414621e-05, -3.70710678125524e-05, -2.25457461413918e-05, 8.28558800228812e-05, -7.00000000000145e-05, 7.00000000000145e-05, 2.84036105176992e-05, 4.38609306566931e-05, -9.5355339059324e-05, 2.29289321887288e-05, -2.59526554579566e-05, 5.49856561282562e-05
//    , 16.90644, -2.00000000027956e-05, 1.05886962214729e-05, 0.000235996919983703, 3.70710678119783e-05, -1.53553390601861e-05, -6.98693979253517e-05, 4.78332004540475e-05, -7.00000000000145e-05, -2.99999999997524e-05, -6.66991445697087e-05, 4.64118442167826e-05, 2.29289321875266e-05, -5.535533905956e-05, -3.40201537265726e-05, -4.54244362536202e-05
//    , 16.90645, 1.00000000013978e-05, -7.1290636326781e-05, 0.000225911677046627, 4.41421356247782e-05, 2.24264068710258e-05, -6.7103141565684e-05, -3.70071380361833e-05, 2.99999999997524e-05, -5.99999999995049e-05, -2.11811296819122e-05, -7.0144223026064e-05, 1.5857864376503e-05, 6.24264068712878e-05, 3.95749075744792e-05, 3.27745920565866e-05
//    , 16.90645, -1.00000000013978e-05, -0.000152316615745576, 0.000181433429173655, 1.53553390601861e-05, 4.70710678124879e-05, 8.51087685161321e-06, -7.61572376654381e-05, 5.99999999995049e-05, 2.99999999997524e-05, 7.29104793857317e-05, 7.27421983961054e-06, -5.535533905956e-05, -3.29289321880362e-05, -4.91047404916673e-05, -1.51351133213544e-05
//    };
//
//    size_t const decon_size = X.size() / 16;
//
//    std::vector<double> decon_linearized(X.size());
//    for (int level = 0; level < 16; ++level)
//        for (size_t row = 0; row < decon_size; ++row)
//            decon_linearized[row + level * decon_size] = X[row * 16 + level];
//
//    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(std::string("stft"), 15));
//    std::vector<double> reconstructed;
//
//    transformer->inverse_transform(decon_linearized, reconstructed, 0);
//
//    for(auto const & r : reconstructed)
//        std::cout << r << ", ";
//
//    std::cout << std::endl;
//}

namespace {
void test_back_transform(std::string && transform)
{
    std::vector<double> deconstructed_etalon = read_test_data("stft_test_data/deconstructed_etalon.txt");

    size_t const decon_size = deconstructed_etalon.size() / 16;

    std::vector<double> decon_linearized(deconstructed_etalon.size());
    for (int level = 0; level < 16; ++level)
        for (size_t row = 0; row < decon_size; ++row)
            decon_linearized[row + level * decon_size] = deconstructed_etalon[row * 16 + level];

    std::unique_ptr<svr::spectral_transform> transformer(svr::spectral_transform::create(std::string(transform), 15));
    std::vector<double> reconstructed;

    transformer->inverse_transform(decon_linearized, reconstructed, 0);

    auto f = open_data_file("stft_test_data/input_queue_etalon.txt");

    ASSERT_TRUE(f.good());

    for(size_t i = 0; i < reconstructed.size(); ++i)
    {
        double val; f >> val;
        if(!svr::common::equals(reconstructed[i], val))
            std::cout << "Not equal: " << reconstructed[i] << ", " << val << " at: " << i << '\n';
        ASSERT_FP_EQ(reconstructed[i], val) << " at: " << i;
    }

    ASSERT_TRUE(f.eof());
}

}

TEST(stft_tests, stft_transform_inverse)
{
    test_back_transform("stft");
    auto start = std::chrono::steady_clock::now();
    test_back_transform("stft");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}

TEST(stft_tests_cpu, stft_transform_inverse)
{
    auto start = std::chrono::steady_clock::now();
    test_back_transform("stft_cpu");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}

TEST(stft_tests_ocl, stft_transform_inverse)
{
    test_back_transform("stft_ocl");
    auto start = std::chrono::steady_clock::now();
    test_back_transform("stft_ocl");
    std::cout << "Test took: " << std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count() << "\n";
}
