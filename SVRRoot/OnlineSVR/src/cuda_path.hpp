//
// Created by zarko on 12/22/22.
//

#ifndef SVR_CUDA_PATH_HPP
#define SVR_CUDA_PATH_HPP

namespace svr::kernel::path {

void
cu_distances_xx(const unsigned long lag, const unsigned long cols, const unsigned long end_col, const unsigned long end_row, const double *X, const double lambda,
                double *Z);

void
cu_distances_xy(const unsigned long lag, const unsigned long X_cols, const unsigned long Xy_cols, const unsigned long end_col, const unsigned long end_row,
                const double *X, const double *Xy, const double lambda, double *Z);

}

#endif //SVR_CUDA_PATH_HPP
