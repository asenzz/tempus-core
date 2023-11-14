//
// Created by zarko on 12/22/22.
//

#ifndef SVR_CUDA_PATH_HPP
#define SVR_CUDA_PATH_HPP

namespace svr::kernel::path {

void cu_distances_xx(const size_t total_len_features, const size_t dim, const size_t size_X, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const double *X,
                     const double lambda, const double tau, const double w_sum_sym, double *Z);

void cu_distances_xy(const size_t total_len_features, const size_t dim, const size_t size_X, const size_t size_Y, const size_t startX, const size_t startY, const size_t numX, const size_t numY,
                     const double *X, const double *Y, const double lambda, const double tau, const double w_sum_sym, double *Z);

double score_distance_kernel(const size_t sizeX, double *Z_distances, double *Y);

}

#endif //SVR_CUDA_PATH_HPP
