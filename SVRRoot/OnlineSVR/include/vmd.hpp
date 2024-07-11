//
// Created by zarko on 7/2/24.
//

#ifndef SVR_VMD_HPP
#define SVR_VMD_HPP

#pragma once
#include <vector>
#include <cmath>
#include <ctime>
#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/FFT>
//#include <eigen3/Eigen/Core>
#define ARMA_DONT_USE_LAPACK
#define ARMA_DONT_USE_BLAS
#undef ARMA_USE_LAPACK
#undef ARMA_USE_BLAS
#include "model/DataRow.hpp"
#include "IQScalingFactorService.hpp"


#define pI acos(-1)
using namespace Eigen;
typedef std::vector<double> vectord;
typedef std::vector<std::complex<double> > vectorcd;
typedef std::vector<MatrixXcd> Matrix3DXd;

void VMD(MatrixXd &u, MatrixXcd &u_hat, MatrixXd &omega, svr::data_row_container::const_iterator iterin, const size_t in_ct, const unsigned input_column_index,
         const double alpha, const double tau, const int K, const int DC, const int init, const double tol, const double eps, const svr::business::t_iqscaler &scaler);
vectorcd circshift(vectorcd& data, int offset);
vectord omega_init_method2(int K, const double fs);
MatrixXcd vector_to_MatrixXcd_in_col(vectorcd& Input);
vectorcd ExtractColFromMatrixXcd(MatrixXcd& Input, const int k, const int T);
vectorcd ExtractRowFromMatrixXd(MatrixXd& Input, const int k, const int T);
MatrixXcd sum(Matrix3DXd& u_hat_plus, const int n);


void printMatrix(const MatrixXd& u);

#endif //SVR_VMD_HPP
