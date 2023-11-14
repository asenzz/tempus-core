#include "_deprecated/r_matrix.h"
#include "_deprecated/r_matrix_impl.hpp"
#include "gtest/gtest.h"
#include <sstream>
#include "../test_harness.hpp"
#if 0
namespace
{
viennacl::matrix<double> fill_matrix(size_t w, bool prn = false)
{
    viennacl::matrix<double> qm(w, w, viennacl::context(viennacl::MAIN_MEMORY));

    for (size_t i = 0; i < w; i++) {
        for (size_t j = 0; j < w; j++) {
            qm(i, j) = exp(-0.5 * fabs((double)i - (double)j)) ;
            //to make sure it is not singular!!
            if(prn)
                std::cout << qm(i, j) << ", ";
        }
        if(prn)
            std::cout << std::endl;
    }

    return qm;
}
}

TEST(RMatrixTests, RMatrixContainer)
{
    const size_t w = 64;
    auto qm = fill_matrix(w);

    svr::datamodel::vmatrix<double> vm;
    vm.copy_from(qm);

    std::vector<int64_t> sv_ind;
    for(size_t i = 0 ; i < w ; i++)
       sv_ind.push_back(i);

    svr::batch::r_matrix_cpu m(&vm, sv_ind);
    const svr::datamodel::vmatrix<double> & Q = *m.Q;

    for (size_t i = 0; i < w; i++) {
        for (size_t j = 0; j < w; j++) {
            ASSERT_EQ(Q.get_value(i, j), exp(-0.5 * fabs((double)i - (double)j)));
            ASSERT_EQ(m.Q->get_value(i, j), exp(-0.5 * fabs((double)i - (double)j)));
        }
    }
}

TEST(RMatrixTests, RMatrixMethods)
{
    const size_t w = 64;
    auto qm = fill_matrix(w);

    svr::datamodel::vmatrix<double> vm;
    vm.copy_from(qm);

    std::vector<int64_t> sv_ind;
    for(size_t i = 0 ; i < w ; i++) 
       sv_ind.push_back(i);

    svr::batch::r_matrix_cpu m(&vm, sv_ind);
    const svr::datamodel::vmatrix<double> & Q = *m.Q;

    std::vector<int64_t> sv_indexes = {3, 4, 7, 11, 15, 23, 32, 45};

    for (size_t SampleIndex = 0; SampleIndex < w; SampleIndex++)
    {
        auto v1 = m.get_qsi(sv_indexes, SampleIndex);

        ASSERT_EQ(v1->get_value(0), 1.0);
        for (size_t i = 0; i < sv_indexes.size()-1; i++) {
            //LOG4_DEBUG("i = " << i << " Q = " << Q.GetValue(SampleIndex, sv_indexes[i]) << " v1 = " << v1->GetValue(i));
            ASSERT_EQ(Q.get_value(SampleIndex, sv_indexes[i]), v1->get_value(i+1));
        }

        delete v1;
    }

    for (size_t SampleIndex = 0; SampleIndex < w - 1; SampleIndex++) {
        auto v1 = m.get_qxi(SampleIndex);
        for (size_t i = 0; i < SampleIndex + 1; i++) {
            ASSERT_EQ(Q.get_value(SampleIndex, i), v1->get_value(i));
        }
        delete v1;
    }

    return ; //This function does not exist anymore
    /* size_t SamplesTrainedNumber = w / 2;
    auto m1 = m.get_qxs(sv_indexes, SamplesTrainedNumber);
    for (size_t i = 0; i < SamplesTrainedNumber; i++)
    {
        ASSERT_EQ(m1.get_value(i, 0), 1.0);
        for (size_t j = 0; j < sv_indexes.size()-1; j++)
        {
            ASSERT_EQ(Q.get_value(i, sv_indexes[j]), m1.get_value(i, j+1));
        }
    }*/
}

void print_matrix(const svr::datamodel::vmatrix<double> & m)
{
    std::cout << std::endl;
    for (int i = 0; i < m.get_length_rows(); i++) {
        std::cout << "{ ";
        for (int j = 0; j < m.get_length_cols(); j++) {
            std::cout << std::setprecision(15) << m.get_value(i, j);
            if (j < m.get_length_cols() - 1) std::cout << ", ";
        }
        std::cout << " }";
        if (i < m.get_length_rows() - 1) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

TEST(RMatrixTests, RMatrixCompute)
{
    const size_t w = 200;
    std::vector<int64_t> sv_indexes;
    auto qm = fill_matrix(w);
    svr::datamodel::vmatrix<double> vm;
    vm.copy_from(qm);
    for (size_t i = 0; i < w ; i++) {
		sv_indexes.push_back(i); 
    }
    std::chrono::duration<float> td{0};
    svr::batch::r_matrix_cpu rm(&vm, sv_indexes);
    auto R = rm.get_r();
    std::cout << "Test took: " << td.count() << "\n";

    //print_matrix(*R);

    std::vector<std::vector<double>> refR = {
#include "../stft_test_data/r_matrix_200.h"
    };

    for (int i = 0; i < R->get_length_rows(); i++) {
        for (int j = 0; j < R->get_length_cols(); j++) {
            ASSERT_FP_EQ(refR[i][j], R->get_value(i, j));
        }
    }
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


TEST(RMatrixTests, RMatrixComputeThenUpdate)
{
    const size_t w = 200;
    const size_t samples_left = 10; 
    std::vector<int64_t> sv_indexes;
    auto qm = fill_matrix(w);

    svr::datamodel::vmatrix<double> vm;
    vm.copy_from(qm);
    for (size_t i = 0; i < w-samples_left ; i++) {
                sv_indexes.push_back(i);
    }
    svr::batch::r_matrix rm(&vm,sv_indexes);


    std::chrono::duration<float> td{0};

    for (size_t i = 0; i < samples_left; i++) {
        sv_indexes.push_back(w-samples_left+i);
        auto start = std::chrono::steady_clock::now();
        rm.update_r_matrix(sv_indexes, w-samples_left+i);
        if(i > 5)
            td += std::chrono::steady_clock::now() - start;
    }
    std::cout << "Test took: " << td.count() << "\n";

    auto R = rm.get_r();

    //print_matrix(*R);

    std::vector<std::vector<double>> refR = {
#include "../stft_test_data/r_matrix_200.h"
    };

    for (int i = 0; i < R->get_length_rows(); i++) {
        for (int j = 0; j < R->get_length_cols(); j++) {
            ASSERT_FP_EQ(refR[i][j], R->get_value(i, j));
        }
    }
}
#endif