//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SMO_KERNEL_H
#define THUNDERSVM_SMO_KERNEL_H

#include "deprecated/thundersvm.h"
#include "clion_cuda.h"
#include "deprecated/syncarray.hpp"

namespace svm_kernel {
    __host__ __device__ inline bool is_I_up(double a, double y, double Cp, double Cn) {
        return (y > 0 && a < Cp) || (y < 0 && a > 0);
    }

    __host__ __device__ inline bool is_I_low(double a, double y, double Cp, double Cn) {
        return (y > 0 && a > 0) || (y < 0 && a < Cn);
    }

    __host__ __device__ inline bool is_free(double a, double y, double Cp, double Cn) {
        return a > 0 && (y > 0 ? a < Cp : a < Cn);
    }

    void
    c_smo_solve(const SyncArray<int> &y, SyncArray<double> &f_val, SyncArray<double> &alpha,
                SyncArray<double> &alpha_diff,
                const SyncArray<int> &working_set, double Cp, double Cn,
                const SyncArray<double> &k_mat_rows,
                const SyncArray<double> &k_mat_diag, int row_len, double eps, SyncArray<double> &diff,
                int max_iter);

    void
    nu_smo_solve(const SyncArray<int> &y, SyncArray<double> &f_val, SyncArray<double> &alpha,
                 SyncArray<double> &alpha_diff,
                 const SyncArray<int> &working_set, double C, const SyncArray<double> &k_mat_rows,
                 const SyncArray<double> &k_mat_diag, int row_len, double eps, SyncArray<double> &diff,
                 int max_iter);

    void
    update_f(SyncArray<double> &f, const SyncArray<double> &alpha_diff, const SyncArray<double> &k_mat_rows,
             int n_instances);

    void sort_f(SyncArray<double> &f_val2sort, SyncArray<int> &f_idx2sort);
}

#endif //THUNDERSVM_SMO_KERNEL_H
