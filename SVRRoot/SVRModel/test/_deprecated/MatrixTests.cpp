/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


#include "model/_deprecated/vektor.tcc"
#include "model/_deprecated/vmatrix.tcc"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <chrono>
#include <unordered_set>


TEST(VMatrixTest, VM_Matrix_TestsAll){

/*  vmatrix test:*/    
    
    const ssize_t n = 17;
    double r = 1;
    svr::datamodel::vektor<double> vr(0., n), evr(0., n);
    svr::datamodel::vektor<double> vrow(0., n);
    
    for(size_t i=0; i<n; i++) 
        vrow.set_at(i, exp(3*i)*pow(i, 2) );
    
    viennacl::matrix<double> copy_of_matrix(n, n);
    svr::datamodel::vmatrix<double> matrix(n, n);
    for(ssize_t i=0; i<(ssize_t)matrix.get_length_rows(); ++i){
        for(ssize_t j=0; j<(ssize_t)matrix.get_length_cols(); ++j){
            matrix.set_value( i, j, exp(3*j)*pow(j, 2) );
    }
    }
    
    matrix.copy_to(copy_of_matrix);
    for(ssize_t i=0; i<(ssize_t)matrix.get_length_rows(); ++i){
        for(ssize_t j=0; j<(ssize_t)matrix.get_length_cols(); ++j){
            matrix.set_value( i, j, exp(3*j)*pow(j, 2) );
        }
    }
    
    for(ssize_t i=0; i<(ssize_t)matrix.get_length_rows(); ++i){
        for(ssize_t j=0; j<(ssize_t)matrix.get_length_cols(); ++j){
            ASSERT_LE( std::fabs(matrix.get_value(i, j) - copy_of_matrix(i,j)), std::numeric_limits<long double>::epsilon() );
        }}
    
    svr::datamodel::vmatrix<double>sub_matrix = matrix.extract_rows(0, n-1);
    svr::datamodel::vektor<double>* ev = sub_matrix.get_row_ptr(0);
    for(size_t i=0; i<n-1; ++i){
        ASSERT_LE( std::fabs(ev->get_value(i) - vrow[i]), std::numeric_limits<long double>::epsilon() );
    }
    
    for(ssize_t i=0; i<(ssize_t)sub_matrix.get_length_rows(); ++i){
        for(ssize_t j=0; j<(ssize_t)sub_matrix.get_length_cols(); ++j){
            ASSERT_LE( std::fabs(sub_matrix.get_value(i, j) - matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
    }
    }

    svr::datamodel::vmatrix<double> kernel_matrix(n, n);          
    for(ssize_t row=0; row<(ssize_t)kernel_matrix.get_length_rows(); ++row){
        for(ssize_t row2=0; row2<(ssize_t)kernel_matrix.get_length_cols(); ++row2){
                r++;
                kernel_matrix.set_value(row, row2, r);
                evr[row2]=r;
        }
    }
            
    for(ssize_t row=0; row<(ssize_t)kernel_matrix.get_length_rows(); ++row){
        for(ssize_t row2=0; row2<(ssize_t)kernel_matrix.get_length_cols(); ++row2)
                vr[row2] = kernel_matrix.get_value(row, row2);
    }
    
    for(ssize_t i=0; i<vr.size(); ++i)
        ASSERT_LE( std::fabs(vr[i] - evr[i]), std::numeric_limits<long double>::epsilon() );
    
    svr::datamodel::vmatrix<double>* cloned_matrix = kernel_matrix.clone();
    for(ssize_t i=0; i<(ssize_t)kernel_matrix.get_length_rows(); ++i){
        for(ssize_t j=0; j<(ssize_t)kernel_matrix.get_length_cols(); ++j){
            ASSERT_LE( std::fabs(cloned_matrix->get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
        }
    
    svr::datamodel::vmatrix<double> test_matrix(0, 0);
    for(int i=0; i<n; ++i )  
        test_matrix.add_row_copy(kernel_matrix.get_row_copy(i));
    
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_row_copy(kernel_matrix.get_row_ptr(i));
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_row_copy(kernel_matrix.get_row_ref(i));
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_row_ref(kernel_matrix.get_row_copy(i));
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_row_ref_safe(kernel_matrix.get_row_copy(i));
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_col_copy(kernel_matrix.get_col_copy(i));
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
    test_matrix.clear();
    for(int i=0; i<n; ++i )  
        test_matrix.add_col_copy_at(kernel_matrix.get_col_copy(i), i);
    
    for(size_t i=0; i<n; ++i)
        for(size_t j=0; j<n; ++j){
            ASSERT_LE( std::fabs(test_matrix.get_value(i, j) - kernel_matrix.get_value(i, j)), std::numeric_limits<long double>::epsilon() );
        }
    
}
