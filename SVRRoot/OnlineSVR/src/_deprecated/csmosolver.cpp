//
// Created by jiashuai on 17-10-25.
//
#include "deprecated/csmosolver.hpp"
#include "deprecated/smo_kernel.h"
#include <climits>


void
CSMOSolver::solve(
        const svr::smo::kernel_matrix &k_mat, const SyncArray<int> &y, SyncArray<double> &alpha, double &rho,
        SyncArray<double> &f_val, const double eps, const double Cp, const double Cn, const int ws_size,
        const int out_max_iter) const
{
#ifndef USE_CUDA
    LOG4_THROW("Works on CUDA devices only!");
#endif

    const int n_instances = k_mat.size1();
    const int q = ws_size / 2;

    SyncArray<int> working_set(ws_size);
    SyncArray<int> working_set_first_half(q);
    SyncArray<int> working_set_last_half(q);
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);

    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncArray<int> f_idx(n_instances);
    SyncArray<int> f_idx2sort(n_instances);
    SyncArray<double> f_val2sort(n_instances);
    SyncArray<double> alpha_diff(ws_size);
    SyncArray<double> diff(2);

    SyncArray<double> k_mat_rows(ws_size * n_instances);
    SyncArray<double> k_mat_rows_first_half(q * n_instances);
    SyncArray<double> k_mat_rows_last_half(q * n_instances);

    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * n_instances]);

    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG4_DEBUG("Training start");
    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    long long local_iter = 0;

    //avoid infinite loop of repeated local diff
    int same_local_diff_cnt = 0;
    double previous_local_diff = INFINITY;
    int swap_local_diff_cnt = 0;
    double last_local_diff = INFINITY;
    double second_last_local_diff = INFINITY;

    for (int iter = 0;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        svm_kernel::sort_f(f_val2sort, f_idx2sort);
        vector<int> ws_indicator(n_instances, 0);
        if (0 == iter) {
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
            k_mat.get_rows(working_set, k_mat_rows);
        } else {
            working_set_first_half.copy_from(working_set_last_half);
            int *working_set_data = working_set.host_data();
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set_data[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
        }
        //local smo
        smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                   max_iter);
        //update f
        svm_kernel::update_f(f_val, alpha_diff, k_mat_rows, n_instances);
        double *diff_data = diff.host_data();
        local_iter += diff_data[1];

        //track unchanged diff
        if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
            same_local_diff_cnt++;
        } else {
            same_local_diff_cnt = 0;
            previous_local_diff = diff_data[0];
        }

        //track unchanged swapping diff
        if (fabs(diff_data[0] - second_last_local_diff) < eps * 0.001) {
            swap_local_diff_cnt++;
        } else {
            swap_local_diff_cnt = 0;
        }
        second_last_local_diff = last_local_diff;
        last_local_diff = diff_data[0];

        if (iter % 100 == 0)
            LOG4_DEBUG(
                    "Global iter = " << iter << ", total local iter = " << local_iter << ", diff = " << diff_data[0]);
        //todo find some other ways to deal unchanged diff
        //training terminates in three conditions: 1. diff stays unchanged; 2. diff is closed to 0; 3. training reaches the limit of iterations.
        //repeatedly swapping between two diffs
        if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) || diff_data[0] < eps ||
            ((out_max_iter != -1) && (iter == out_max_iter)) ||
            (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            LOG4_DEBUG(
                    "Global iter = " << iter << ", total local iter = " << local_iter << ", diff = " << diff_data[0]);
            LOG4_DEBUG("training finished");
            double obj = calculate_obj(f_val, alpha, y);
            LOG4_DEBUG("obj = " << obj);
            break;
        }
    }
}

void
CSMOSolver::select_working_set(
        vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
        const SyncArray<double> &alpha, double Cp, double Cn,
        SyncArray<int> &working_set) const
{
    const size_t n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    size_t n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const double *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < (ssize_t) n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !svm_kernel::is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == (ssize_t) n_instances) break;
                i = index[p_left];
            }
            if (p_left < (ssize_t) n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !svm_kernel::is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
    }
}

double
CSMOSolver::calculate_rho(
        const SyncArray<double> &f_val, const SyncArray<int> &y, SyncArray<double> &alpha,
        double Cp,
        double Cn) const
{
    int n_free = 0;
    double sum_free = 0;
    double up_value = INFINITY;
    double low_value = -INFINITY;
    const double *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    double *alpha_data = alpha.host_data();
    for (size_t i = 0; i < alpha.size(); ++i) {
        if (svm_kernel::is_free(alpha_data[i], y_data[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val_data[i];
        }
        if (svm_kernel::is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value = min(up_value, f_val_data[i]);
        if (svm_kernel::is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value = max(low_value, f_val_data[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void
CSMOSolver::init_f(const SyncArray<double> &alpha, const SyncArray<int> &y, const svr::smo::kernel_matrix &k_mat,
                   SyncArray<double> &f_val) const
{
    //todo auto set batch size
    const size_t batch_size = DEFAULT_BATCH_SIZE;
    vector<int> idx_vec;
    vector<double> alpha_diff_vec;
    const int *y_data = y.host_data();
    const double *alpha_data = alpha.host_data();
    for (size_t i = 0; i < alpha.size(); ++i) {
        if (alpha_data[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha_data[i] * y_data[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && !idx_vec.empty())) {
            SyncArray<int> idx(idx_vec.size());
            SyncArray<double> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncArray<double> kernel_rows(idx.size() * k_mat.size());
            k_mat.get_rows(idx, kernel_rows);
            svm_kernel::update_f(f_val, alpha_diff, kernel_rows, k_mat.size());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(
        const SyncArray<int> &y, SyncArray<double> &f_val, SyncArray<double> &alpha,
        SyncArray<double> &alpha_diff,
        const SyncArray<int> &working_set, double Cp, double Cn,
        const SyncArray<double> &k_mat_rows,
        const SyncArray<double> &k_mat_diag, int row_len, double eps,
        SyncArray<double> &diff,
        int max_iter) const
{
    svm_kernel::c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps,
                            diff, max_iter);
}

double CSMOSolver::calculate_obj(const SyncArray<double> &f_val, const SyncArray<double> &alpha,
                                 const SyncArray<int> &y) const
{
    //todo use parallel reduction for gpu and cpu
    int n_instances = f_val.size();
    double obj = 0;
    const double *f_val_data = f_val.host_data();
    const double *alpha_data = alpha.host_data();
    const int *y_data = y.host_data();
    /*cilk_*/for(int i = 0; i < n_instances; ++i) {
        obj += alpha_data[i] - (f_val_data[i] + y_data[i]) * alpha_data[i] * y_data[i] / 2.;
    }
    return -obj;
}

