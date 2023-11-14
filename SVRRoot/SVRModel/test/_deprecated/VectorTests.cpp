#include "../TestSuite.hpp"
#include "model/_deprecated/vektor.tcc"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <chrono>
#include <unordered_set>

using vektor = svr::datamodel::vektor<double>;

std::vector<double> initVector(size_t len)
{
    std::vector<double> v(len, 0);
    double a = -10;

    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = a;
        a += 0.2;
    }

    return v;
}

std::vector<double> initVectorRandom(size_t len)
{
    std::vector<double> v(len, 0);
    const int precision = 1e3;
    const int maxVal = 100;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(- maxVal * precision, maxVal * precision);

    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = double(distribution(generator)) / precision;
    }

    return v;
}

TEST(VectorTests, PowBasicTests)
{
    vektor v1(0.0, 77), v2(0.0, 77);
    for(int i = 0; i < v1.size(); ++i)
        v1[i] = double(i);

    v1.pow_scalar(2);
    for(int i = 0; i < v1.size(); ++i)
        ASSERT_LE(fabs(v1[i] - pow(double(i), 2)), std::numeric_limits<float>::epsilon());

    for(int i = 0; i < v2.size(); ++i)
        v2[i] = double(i);

    v2.square_scalar();
    for(int i = 0; i < v2.size(); ++i)
        ASSERT_LE(fabs(v2[i] - pow(double(i), 2)), std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, PowPerformanceTests)
{
    if ( !TestSuite::doPerformanceTests() )
        return;

    const size_t len = 1e+7;
    const int maxVal = 1000;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, maxVal);

    vektor v1(0.0, len), v2(0.0, len);


    using namespace std::chrono;

    for(size_t i = 0; i < len; ++i)
    {
        double u = double(distribution(generator));
        v1[i] = u / maxVal;
        v2[i] = u / maxVal;
    }

    /*  Test 1   **************************************************************/
    high_resolution_clock::time_point start = high_resolution_clock::now();

    v1.pow_scalar(2);

    high_resolution_clock::time_point finish = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(start - finish);
    std::cout << "TEST(VectorTests, PowPerformanceTests).PowScalar(2):" << " length: " << len << " took " << time_span.count() << "sec." << std::endl;

    /*  Test 2   **************************************************************/

    start = high_resolution_clock::now();

    v2.square_scalar();

    finish = high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(start - finish);
    std::cout << "TEST(VectorTests, PowPerformanceTests).SquareScalar():" << " length: " << len << " took " << time_span.count() << "sec." << std::endl;

    /*  Test 3   **************************************************************/

    start = high_resolution_clock::now();

    v2.square_scalar();

    finish = high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(start - finish);
    std::cout << "TEST(VectorTests, PowPerformanceTests).SquareScalar():" << " length: " << len << " took " << time_span.count() << "sec." << std::endl;

    /*  Test 4   **************************************************************/

    start = high_resolution_clock::now();

    v1.pow_scalar(2);

    finish = high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(start - finish);
    std::cout << "TEST(VectorTests, PowPerformanceTests).PowScalar(2):" << " length: " << len << " took " << time_span.count() << "sec." << std::endl;

    /*  Done   ****************************************************************/

    ASSERT_EQ(v1, v2);

}

TEST(VectorTests, Container)
{
    vektor v1(initVectorRandom(111));

    vektor * vClone = v1.clone();
    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(v1[i], vClone->get_value(i));
    }

    // Copy to STL vektor
    std::vector<double> stl_vec(v1.size());
    v1.copy_to(stl_vec);
    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(v1[i], stl_vec[i]);
    }

    // Resize
    vektor vResize(v1);
    ASSERT_EQ(vResize.size(), v1.size());
    vResize.memresize(v1.size() * 4);
    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(v1[i], vResize[i]);
    }

    // Add
    vektor vAdd(v1);
    vAdd.add(3.14);
    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(v1[i], vAdd[i]);
    }
    ASSERT_EQ(vAdd[v1.size()], 3.14);

    // AddAt
    vektor vAddAt(std::vector<double>{1.0, 2, 3, 4, 5, 6, 7});
    vektor v2(vAddAt);
    vAddAt.add_at(M_PI, 3);
    vAddAt.add_at(M_PI, 0);
    vAddAt.add_at(M_PI, 9);
    // TODO: Add more asserts to ensure add/remove correctness...
    // RemoveAt
    vAddAt.remove_at(9);
    vAddAt.remove_at(0);
    vAddAt.remove_at(3);
    for (int i = 0; i < vAddAt.size(); ++i) {
        ASSERT_LE(fabs(vAddAt[i] - v2[i]), std::numeric_limits<float>::epsilon());
    }

    // Set/Get
    v1.set_at(11, M_PI);
    ASSERT_LE(M_PI - v1.get_value(11),  std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, CTOR)
{
    const size_t n = 10;
    vektor* v1 = new vektor(n + 1);
    for (size_t i = 0; i < n + 1; i++) {
        v1->add(1);
    }
    vektor* v2 = new vektor(1, n + 1);
    for (int i = 0; i < v1->size(); ++i) {
        ASSERT_LE(fabs(v1->get_value(i) - v2->get_value(i)), std::numeric_limits<float>::epsilon());
    }

    // TODO: Add more CTOR tests...
}

TEST(VectorTests, SortRemoveDuplicates)
{
    vektor v1(std::vector<double>{1, 5, 3, 8, 34, 8, 12, 5, 14, 15, 3, 4, 5, 6, 7, 55, 4});
    std::unordered_set<double> uniques;

    v1.sort_and_remove_duplicates();

    int i0 = 0;
    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_EQ(uniques.count(v1[i]), static_cast<size_t>(0));
        uniques.insert(v1[i]);
        ASSERT_LE(i0, v1[i]);
        i0 = v1[i];
    }
}

TEST(VectorTests, Extract)
{
    vektor v1(initVectorRandom(33));

    const int from = 5;
    const int to = 15;

    vektor * v2 = v1.extract(from, to);
    ASSERT_EQ(v2->size(), to - from + 1);

    for (int i = from; i <= to; ++i) {
        ASSERT_LE(fabs(v2->get_value(i - from) - v1[i]), std::numeric_limits<float>::epsilon());
    }
}

TEST(VectorTests, ABS)
{
    vektor v1(initVector(111));
    vektor v2(v1);

    v1.abs();

    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_LE(fabs(fabs(v2[i]) - v1[i]), std::numeric_limits<float>::epsilon());
    }
}

TEST(VectorTests, Sum)
{
    vektor v1(initVectorRandom(33));

    double sum = v1.sum();
    double sum2 = 0;

    for (int i = 0; i < v1.size(); ++i) {
        sum2 += v1[i];
    }

    ASSERT_LE(fabs(sum - sum2), std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, AbsSum)
{
    vektor v1(initVectorRandom(33));
    vektor v2(v1);

    double sum = v1.abs_sum();
    double sum2 = 0;

    for (int i = 0; i < v1.size(); ++i) {
        sum2 += fabs(v2[i]);
    }

    ASSERT_LE(fabs(sum - sum2), std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, SumVector)
{
    vektor v1(initVectorRandom(33));
    vektor v2(initVectorRandom(33));
    vektor v3(v1);

    v1.sum_vector(&v2);

    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_LE(fabs(v1[i] - (v2[i] + v3[i])), std::numeric_limits<float>::epsilon());
    }
}

TEST(VectorTests, SumScalar)
{
    vektor v1(initVectorRandom(33));
    vektor v3(v1);
    const double scalar = 5.5;

    v1.sum_scalar(scalar);

    for (int i = 0; i < v1.size(); ++i) {
        ASSERT_LE(fabs(v1[i] - (v3[i] + scalar)), std::numeric_limits<float>::epsilon());
    }
}

TEST(VectorTests, ProductVector)
{
    vektor v1(initVectorRandom(55));
    vektor v2(initVectorRandom(55));
    vektor v3(v1);

    v1.product_vector(v2);

    for (int i = 0; i < v1.size(); ++i) {
        double prod2 = v2[i] * v3[i];
        ASSERT_LE(fabs(prod2 - v1[i]), std::numeric_limits<float>::epsilon());
    }
}

TEST(VectorTests, ProductVectorScalar)
{
    vektor v1(initVectorRandom(55));
    vektor v2(initVectorRandom(55));
    vektor v3(v1);

    double prod = v1.product_vector_scalar(v2);
    double prod2 = 0;

    for (int i = 0; i < v1.size(); ++i) {
        prod2 += v2[i] * v3[i];
    }

    ASSERT_LE(fabs(prod2 - prod), std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, Variance)
{
    vektor v1(initVectorRandom(111));

    double variance = v1.variance();
    double variance2 = 0;
    double mean = v1.mean();

    for (int i = 0; i < v1.size(); ++i) {
        variance2 += (fabs(v1[i] - mean)) * (fabs(v1[i] - mean));
    }
    variance2 /= v1.size();

    ASSERT_LE(fabs(variance2 - variance), std::numeric_limits<float>::epsilon());
}

TEST(VectorTests, MinMax)
{
    vektor v1(initVectorRandom(55));

    double min = v1.min();
    double max = v1.max();

    double min2 = v1[0];
    double max2 = v1[0];

    for (int i = 1; i < v1.size(); ++i) {
        if (min2 > v1[i]) {
            min2 = v1[i];
        }
        if (max2 < v1[i]) {
            max2 = v1[i];
        }
    }

    ASSERT_EQ(min, min2);
    ASSERT_EQ(max, max2);
}

TEST(VectorTests, vcl_container)
{
    vektor v1;

    for (size_t i = 0; i < 64; i++) {
        v1.add(i);
    }

    auto v2 = v1.vcl();

    for (size_t i = 0; i < 64; i++) {
        ASSERT_EQ(i, (*v2)[i]);
    }
}

#define ASSERT_FP_EQ(op1, op2) ASSERT_LT(fabs((op1) - (op2)), std::numeric_limits<float>::epsilon()) << " op1: " << (op1) << " op2: " << (op2);

TEST(VectorComparisonTest, DotProduct)
{
    vektor v1;

    for (size_t i = 0; i < 6400; i++) {
        v1.add(i);
    }

    double r1 = v1.product_vector_scalar_cpu(v1, v1);

    double r2 = v1.product_vector_scalar_gpu(v1, v1);

    std::cout << "r1: " << r1 << " r2: " << r2 << "\n";

    ASSERT_FP_EQ(r1, r2) ;
}