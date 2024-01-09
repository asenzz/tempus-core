/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "../include/onlinesvr.hpp"
#include "kernel_basic_integration_test.hpp"


#define MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/GA_trainpredict-basic_integration-input.txt")
#define DUMP_FILE ("global_alignment_train_predict_dump.txt")
#define SAVED_MODEL_FILE ("../SVRRoot/OnlineSVR/test/test_data/online_svr_final_gak_test.txt")


void load_row(std::string row, svr::datamodel::vektor<double> &result)
{

    int yi = 0;
    std::string srow;
    std::stringstream ss(row);
    while (std::getline(ss, srow, ' ')) {
        result.add_at(std::atof(srow.c_str()), yi++);
    }
}

/*This function loads raw data of Y-labels and X-feature matrix from file*/
void load_srv_raw_data(
        const std::string &fileX,
        const std::string &fileY,
        svr::datamodel::vmatrix<double> &X,
        svr::datamodel::vektor<double> *Y)
{

    std::ifstream sX(fileX), sY(fileY);
    std::string field;
    svr::datamodel::vektor<double> row;

    //load X
    while (std::getline(sX, field, '\n')) {
        load_row(field, row);
        int ri = 0;
        X[ri++] = row;
    }

    //build Y
    field.clear();
    int yi = 0;
    while (std::getline(sY, field, '\n')) {
        Y->add_at(std::atof(field.c_str()), yi++);
    }

}

static long double RbfKer(const svr::datamodel::vektor<double> &V1, const svr::datamodel::vektor<double> &V2,
                          const svr::datamodel::SVRParameters &params)
{

    auto V = vektor<double>::subtract_vector(V1, V2);
    V.square_scalar();
    auto K = V.sum();
    K *= -params.get_svr_kernel_param();

    return std::exp(K);
}


TEST(kernel_ga_1122, k_ga)
{

    //todo: integrate with tests in "train_predict_tests.cpp"

    SVRParameters params;
    std::string modified_file_path = "../SVRRoot/OnlineSVR/test/test_data/vectors_200.txt";

    svr::datamodel::vmatrix<double> *p_learning_data = svr::datamodel::vmatrix<double>::load(
            modified_file_path.c_str());

    params.set_lag_count(9);//Must be a divisor of the number of columns in the file, which in this case is 549.

    params.set_kernel_type(svr::datamodel::kernel_type::GA);
    params.set_svr_kernel_param(20); //gamma for GA - must be big number 
    params.set_svr_kernel_param2(1); //this is the lambda, the triangular is fixed in the header file
    params.set_svr_epsilon(1e-5);
    params.set_svr_C(100);

    std::vector<double> seq1, seq2;
    auto kga = svr::IKernel<double>::get(params);

//    std::vector<double> s1{1, 2, 3}, s2{4, -7, 9};
//    double logKGA = kga.logGAK((double*)s1.data(), (double*)s2.data(), (int)s1.size(), (int)s2.size(), 1, 0.00001, 1);
//    double rbf = RbfKer(, , params);
//    std::cout<<logKGA<<"\n";
//    std::cout<<rbf<<"\n";

    std::cout << "test logGAK\n";
    for (ssize_t v = 0; v < 1/* p_learning_data[0][0].size()-1 */; v++) {

        std::vector<double> s1, s2;

        for (ssize_t k = 0; k < p_learning_data[0][v].size(); k++) {
            s1.push_back(p_learning_data[0][v][k]);
        }

        for (ssize_t k = 0; k < p_learning_data[0][v + 1].size(); k++) {
            s2.push_back(p_learning_data[0][v + 1][k]);
        }

        double logKGA = 0; // (*kga)(p_learning_data[0][v], p_learning_data[0][v + 1]);
        double RBF = RbfKer(p_learning_data[0][v], p_learning_data[0][v + 1], params);

        std::cout << "logKGA: " << logKGA << "\n";
        std::cout << "RBF: " << RBF << "\n";
        std::cout << "\n";
    }

    return;

//    single call test with our 'vektor' implementation
//      std::cout<<"KGA no input vector transpose test, using vektor\n";   
//      double dst_v = kga(p_learning_data[0][1], p_learning_data[0][1]);
//      std::cout<< "distance: "<< dst_v <<std::endl;
//      
//      std::cout<<"KGA input vector transpose test, using vektor\n";   
//      double dst_vt = kga(p_learning_data[0][1], p_learning_data[0][1], 1 /*use of last parameter calls transposed GA*/);
//      std::cout<< "transpose distance: "<< dst_vt <<std::endl;

//  test with our 'vektor' implementation using GA and GA with 'transpose'   

//    std::cout<<"KGA test between different vectors\n";
//    for(ssize_t v=0; v < p_learning_data[0][0].size()-1; v++){
//   
//        double dst_v_v1_rbf = RbfKer(p_learning_data[0][v], p_learning_data[0][v+1], params);
//        double dst_v_v1 = kga(p_learning_data[0][v], p_learning_data[0][v+1]);
//        double dst_v_v1_t = kga(p_learning_data[0][v], p_learning_data[0][v+1], 1 /*use of last parameter calls transposed GA*/);
//        
//        
//        std::cout<< "rbf  : " << dst_v_v1_rbf << std::endl;
//        std::cout<< "kga-t: " << dst_v_v1_t << std::endl;
//        std::cout<< "kga  : " << dst_v_v1 << std::endl;
//        
//        if(dst_v_v1 != dst_v_v1_t){
//            std::cout<< "\tdistance between "<<v<<" and "<<v+1<<" not equal" <<std::endl;
//            std::cout<< "\tdistance"<< dst_v_v1 <<std::endl;
//            std::cout<< "\ttranspose"<< dst_v_v1_t <<std::endl;
//            std::cout<< "\tdifference is: "<<abs(dst_v_v1 - dst_v_v1_t)<<std::endl;
//        }
//        else{
//            std::cout<< "distance between "<<v<<" and "<<v+1<<" : "<< dst_v_v1 <<std::endl;
//            std::cout<< "transpose distance between "<<v<<" and "<<v+1<<" : "<< dst_v_v1_t <<std::endl;
//        }
//    }

//    std::cout<<"KGA test between equal vectors\n";
//    for(ssize_t v=0; v < p_learning_data[0][0].size()-1; v++){
//   
//        double dst_v_v1 = kga(p_learning_data[0][v], p_learning_data[0][v]);
//        double dst_v_v1_t = kga(p_learning_data[0][v], p_learning_data[0][v], 1 /*use of last parameter calls transposed GA*/);
//
//        if(dst_v_v1 != dst_v_v1_t){
//            std::cout<< "\tdistance between "<<v<<" and "<<v+1<<" not equal" <<std::endl;
//            std::cout<< "\tdistance"<< dst_v_v1 <<std::endl;
//            std::cout<< "\ttranspose"<< dst_v_v1_t <<std::endl;
//            std::cout<< "\tdifference is: "<<abs(dst_v_v1 - dst_v_v1_t)<<std::endl;
//        }
//        else{
//            std::cout<< "distance between "<<v<<" and "<<v+1<<" : "<< dst_v_v1 <<std::endl;
//            std::cout<< "transpose distance between "<<v<<" and "<<v+1<<" : "<< dst_v_v1_t <<std::endl;
//        }
//    }

//    test with STL implementation    
//    std::cout<<"KGA test using STL\n";
//    for(ssize_t v=0; v < p_learning_data[0][0].size(); v++){
//    
//    //    std::cout<<"vector 0\n";
//        for(ssize_t i=0; i<p_learning_data[0][v].size(); i++){
//    //        printf("%lf ", p_learning_data[0][0][i]);
//            seq1.push_back(p_learning_data[0][v][i]);
//        }
//
//    //    std::cout<<"\nvector 1\n";
//        for(ssize_t i=0; i<p_learning_data[0][v+1].size(); i++){
//    //        printf("%lf ", p_learning_data[0][1][i]);
//            seq2.push_back(p_learning_data[0][v+1][i]);
//
//        }
//
//        double dst_v_v1 = kga(seq1, seq2);
//
//        std::cout<< "distance between "<<v<<" and "<<v+1<<": "<< dst_v_v1 <<std::endl;
//
//        seq1.clear();
//        seq2.clear();
//    
//    }

}

TEST(gak_train_predict, basic_integration)
{
    //kernel_basic_integration_test(MODEL_FILE, DUMP_FILE, SAVED_MODEL_FILE);
}
