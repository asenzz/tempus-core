//
// Created by zarko on 11/12/22.
//

#ifndef SVR_PATH_KERNEL_H
#define SVR_PATH_KERNEL_H

#include "common/defines.h"


//On AMD GPU
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif


#define blockX(i, j) (X[(startX + (i)) * input_internal_size2 + (j)])
#define blockY(i, j) (X[(startY + (i)) * input_internal_size2 + (j)])
#define w(i_j_plus_1) (1.0 / w_sum_sym * (i_j_plus_1) * (i_j_plus_1))

#define type_float double

#define DIFF_COEFF 0.25
#define Nx_local 1024



#endif //SVR_PATH_KERNEL_H
