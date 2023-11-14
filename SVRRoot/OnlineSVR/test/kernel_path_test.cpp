/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <sstream>
#include <gtest/gtest.h>

#include "test_harness.hpp"
#include "../include/onlinesvr.hpp"
#include "kernel_factory.hpp"


#if 0
static long double RbfKer(const svr::datamodel::vektor<double> &V1, const svr::datamodel::vektor<double> &V2, const svr::datamodel::SVRParameters& params) {
        
        auto V = vektor<double>::subtract_vector(V1, V2);
        V.square_scalar();
        auto K = V.sum();
        K *=  - params.get_svr_kernel_param();
        
    return std::exp(K);
}
#endif

TEST(kernel_path_rbf, k_path)
{
#if 0
    //todo: integrate with tests in "train_predict_tests.cpp"
    SVRParameters params;
    std::string modified_file_path = "../SVRRoot/OnlineSVR/test/test_data/vectors_200_path.txt";

    //arma::mat *p_learning_data = svr::datamodel::vmatrix<double>::load(modified_file_path.c_str());

    params.set_lag_count(9);//Must be a divisor of the number of columns in the file, which in this case is 549.

    params.set_kernel_type(svr::datamodel::kernel_type::PATH);
    params.set_svr_kernel_param(20); //gamma for PATH - must be big number 
    params.set_svr_kernel_param2(0.4); //this is between 0 and 1, controls how much weight for diagonal vs horizontal or vertital moves. Neutral is 0.333333
    params.set_svr_epsilon(1e-5);
    params.set_svr_C(100);
    SVRParameters rbf_params(params);
    rbf_params.set_svr_kernel_param(1./2*(params.get_svr_kernel_param()));
    std::vector<double> seq1, seq2;
    auto kpath = svr::IKernel<double>::get_kernel(svr::datamodel::kernel_type::PATH,params);
    
    std::cout<<"test PATH kernel computation, compare with RBF (there is no scenario to have them equal)\n";
    for(ssize_t v = 0; v < 1/* p_learning_data[0][0].size()-1 */; v++)  {
        
         std::vector<double> s1, s2;
        
        for(ssize_t k=0; k < p_learning_data[0][v].size(); k++){
            s1.push_back(p_learning_data[0][v][k]);
        }
        
        for(ssize_t k=0; k < p_learning_data[0][v+1].size(); k++){
            s2.push_back(p_learning_data[0][v+1][k]);
        }
    
        double path_val = (*kpath)(p_learning_data[0][v], p_learning_data[0][v+1]);
        double RBF = RbfKer(p_learning_data[0][v], p_learning_data[0][v+1], params);
       
        std::cout<< "PATH: " << path_val <<"\n";
        std::cout<< "RBF: " << RBF <<"\n";
        std::cout<<"\n";
    }
#endif
    return;
}
