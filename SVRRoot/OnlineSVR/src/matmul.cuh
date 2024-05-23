//
// Created by zarko on 5/11/24.
//

#ifndef SVR_MATMUL_CUH
#define SVR_MATMUL_CUH

namespace svr {

/*
 * parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C)
            to store the result
 */
void matmul(const double *d_a, const double *d_b, double *d_c, const unsigned m, const unsigned n, const unsigned k, const cudaStream_t &strm);

}

#endif //SVR_MATMUL_CUH
