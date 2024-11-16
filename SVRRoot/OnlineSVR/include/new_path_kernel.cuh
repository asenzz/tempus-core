//
// Created by zarko on 19/10/2024.
//

#ifndef SVR_NEW_PATH_KERNEL_CUH
#define SVR_NEW_PATH_KERNEL_CUH

#include <armadillo>
#include "common/compatibility.hpp"

namespace svr {
namespace kernel {

double *cu_compute_path_distances(CRPTRd x_t, CRPTRd y_t, const unsigned n_rows_x, const unsigned n_rows_y, const unsigned n_cols_x, const unsigned n_cols_y, const unsigned lag,
                                  const double lambda, const double tau, const cudaStream_t custream);

arma::mat path_distances_t(const arma::mat &x_t, const arma::mat &y_t, const unsigned lag, const double lambda, const double tau);

arma::mat path_distances_t(const arma::mat &x_t, const unsigned lag, const double lambda, const double tau);

}
}

#endif //SVR_NEW_PATH_KERNEL_CUH
