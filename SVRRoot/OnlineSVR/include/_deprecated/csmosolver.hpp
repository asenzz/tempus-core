//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_CSMOSOLVER_H
#define THUNDERSVM_CSMOSOLVER_H

#define DEFAULT_BATCH_SIZE 100

#include "thundersvm.h"
#include "smo.h"
#include "syncarray.hpp"

/**
 * @brief C-SMO solver for SVC, SVR and OneClassSVC
 */
class CSMOSolver {
public:
    void solve(const svr::smo::kernel_matrix &k_mat, const SyncArray<int> &y, SyncArray<double> &alpha, double &rho,
               SyncArray<double> &f_val, const double eps, const double Cp, const double Cn, int ws_size, int out_max_iter) const;

    virtual ~CSMOSolver() = default;

protected:
    void init_f(const SyncArray<double> &alpha, const SyncArray<int> &y, const svr::smo::kernel_matrix &k_mat,
                SyncArray<double> &f_val) const;

    virtual void
    select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                       const SyncArray<double> &alpha, double Cp, double Cn,
                       SyncArray<int> &working_set) const;

    virtual double
    calculate_rho(const SyncArray<double> &f_val, const SyncArray<int> &y, SyncArray<double> &alpha,
                  double Cp, double Cn) const;
    double calculate_obj(const SyncArray<double> &f_val, const SyncArray<double> &alpha,
                                 const SyncArray<int> &y) const;

    virtual void
    smo_kernel(const SyncArray<int> &y, SyncArray<double> &f_val, SyncArray<double> &alpha,
               SyncArray<double> &alpha_diff,
               const SyncArray<int> &working_set, double Cp, double Cn, const SyncArray<double> &k_mat_rows,
               const SyncArray<double> &k_mat_diag, int row_len, double eps, SyncArray<double> &diff,
               int max_iter) const;
};

#endif //THUNDERSVM_CSMOSOLVER_H
