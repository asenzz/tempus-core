#include <cstdio>
#include <iostream>
#include <cinttypes>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <set>
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <omp.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>
#include <limits>
#include "test_pos.h"
#include "test_pos_in.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


//#define USE_MPI

#include <complex>
//#define ARMA_NO_DEBUG
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include <fstream>

//inverse normal cdf

#include <sstream>
#include <string>

#include "fast_functions.hpp"
#include "firefly.hpp"

arma::mat call_gpu_oversolve(const arma::mat &Left, const arma::mat &Right)
{
    assert(arma::size(Left, 0) == arma::size(Right, 0));

    size_t Nrows = arma::size(Left, 0);
    size_t Ncols = arma::size(Left, 1);
    size_t Nrhs = arma::size(Right, 1);

    arma::mat output = arma::zeros(Ncols, Nrhs);
    //call_gpu_overdetermined(Nrows,Ncols,Nrhs,Left.memptr(),Right.memptr(),output.memptr());
    abort();
    return output;
}

extern int do_gpu_solve(size_t m, double *Left, double *Right, double *output);

arma::mat call_gpu_solve(arma::mat Left, arma::mat Right)
{
    size_t m = arma::size(Left, 0);
    arma::mat output = arma::zeros(m, 1);
    do_gpu_solve(m, Left.memptr(), Right.memptr(), output.memptr());
    return output;
}

int do_gpu_mmul(const arma::mat &A, const arma::mat &B, arma::mat &C)
{
    // Allocate 3 arrays on GPU
    double *d_A, *d_B, *d_C;
    size_t nr_rows_A = arma::size(A, 0);
    size_t nr_cols_A = arma::size(A, 1);
    size_t nr_rows_B = arma::size(B, 0);
    size_t nr_cols_B = arma::size(B, 1);

    cudaMalloc((void **) &d_A, nr_rows_A * nr_cols_A * sizeof(double));
    cudaMalloc((void **) &d_B, nr_rows_B * nr_cols_B * sizeof(double));
    cudaMalloc((void **) &d_C, nr_rows_A * nr_cols_B * sizeof(double));
    cudaMemcpy(d_A, A.memptr(), nr_rows_A * nr_cols_A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.memptr(), nr_rows_B * nr_cols_B * sizeof(double), cudaMemcpyHostToDevice);
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    cudaDeviceSynchronize();
    C.set_size(nr_rows_A, nr_cols_B);
    cudaMemcpy(C.memptr(), d_C, (size_t) nr_rows_A * nr_rows_B * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

extern arma::mat call_gpu_mmul(const arma::mat &A, const arma::mat &B);

arma::mat solve_irwls(double epsilon, const arma::mat &Z, const arma::mat &rhs)
{
    const int method = 0;
    switch (method) {
        case 0: {
            return call_gpu_solve(Z + epsilon * arma::eye(arma::size(Z)), rhs);
            break;
        }
        case 1: {
            const double delta = 0.00001;
            arma::mat x = call_gpu_solve(Z + epsilon * arma::eye(arma::size(Z)), rhs);
            const int irwls_iter = 10;
            for (int i = 0; i < irwls_iter; i++) {

                arma::mat errors = arma::abs(Z * x - rhs);
                arma::mat mult = 1. / arma::sqrt(errors + arma::ones(arma::size(rhs)) * delta / ((double) i + 1.));

                arma::mat Zmult = mult * arma::ones(1, arma::size(Z, 1));

                arma::mat Left = Zmult % Z;
                arma::mat Right = rhs % mult;

                arma::mat x_new = call_gpu_solve(Left + epsilon * arma::eye(arma::size(Left)), Right);
                x = x_new;
            }
            return x;
        }
        default: {
            abort();
        }
    }
}

int arma_multisolve(double epsilon, int Nrows, int lda, const double *Ainput, int start, const double *rhs, double *B)
{
    arma::mat Z(lda, lda);
    std::memcpy(Z.memptr(), Ainput, lda * lda * sizeof(double));
    arma::mat b(Nrows, 1);
    std::memcpy(b.memptr(), rhs, Nrows * sizeof(double));
    std::cout << "Where " << arma::size(Z) << " " << start << " " << Nrows << std::endl;
    Z = Z.rows(start, start + Nrows - 1);
    std::memset(B, 0, Nrows * sizeof(double));
    //arma::mat x = irwls_solve(Z.cols(start,start+Nrows-1), b ,epsilon);

    std::cout << "before " << arma::size(Z) << " " << start << " " << Nrows << std::endl;
    Z = Z.cols(start, start + Nrows - 1);
    std::cout << "after " << arma::size(Z) << " " << start << " " << Nrows << std::endl;
    const int multinum = 16;
    arma::mat x = 1. / (double) multinum * arma_singlesolve(epsilon, Z, b);
    for (int i = 0; i < multinum - 1; i++) {
        arma::mat ZZ = Z.rows(i + 1, Nrows - 1);
        arma::mat y = arma_singlesolve(epsilon, ZZ.cols(i + 1, Nrows - 1), b.rows(i + 1, Nrows - 1));
        x.rows(i + 1, Nrows - 1) += y * 1. / (double) multinum;
    }
    std::memcpy(B, x.memptr(), Nrows * sizeof(double));
}

size_t get_inversions_count(double *arr, size_t N)
{
    std::multiset<double> set1;
    set1.insert(arr[0]);
    size_t invcount = 0; //initializing result
    std::multiset<double>::iterator itset1;
    for (size_t i = 1; i < N; i++) {
        set1.insert(arr[i]);
        itset1 = set1.upper_bound(arr[i]);
        invcount += std::distance(itset1, set1.end());
    }
    return invcount;
}


size_t merge(double *arr, double *temp, size_t left, size_t mid, size_t right)
{
    size_t i, j, k;
    size_t inv_count = 0;

    i = left; /* i is index for left subarray*/
    j = mid; /* j is index for right subarray*/
    k = left; /* k is index for resultant merged subarray*/
    while ((i <= mid - 1) && (j <= right)) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            /* this is tricky -- see above
            explanation/diagram for merge()*/
            inv_count = inv_count + (mid - i);
        }
    }
    /* Copy the remaining elements of left subarray
(if there are any) to temp*/
    while (i <= mid - 1)
        temp[k++] = arr[i++];
    /* Copy the remaining elements of right subarray
    (if there are any) to temp*/
    while (j <= right)
        temp[k++] = arr[j++];
    /*Copy back the merged elements to original array*/
    for (i = left; i <= right; i++)
        arr[i] = temp[i];
    return inv_count;
}


/* An auxiliary recursive function
that sorts the input array and
returns the number of inversions in the array. */
size_t _merge_compute_inversions(double *arr, double *temp, size_t left, size_t right)
{
    size_t mid, inv_count = 0;
    if (right > left) {
        /* Divide the array into two parts and
        call _mergeSortAndCountInv()
        for each of the parts */
        mid = (right + left) / 2;

        /* Inversion count will be sum of
        inversions in left-part, right-part
        and number of inversions in merging */
        inv_count += _merge_compute_inversions(arr, temp, left, mid);
        inv_count += _merge_compute_inversions(arr, temp, mid + 1, right);

        /*Merge the two parts*/
        inv_count += merge(arr, temp, left, mid + 1, right);
    }
    return inv_count;
}

size_t merge_compute_inversions(double *arr, size_t N)
{
    double temp[N];
    size_t result = _merge_compute_inversions(arr, temp, 0, N - 1);
    return result;
}


double compute_score_metrics_arma(const arma::mat &K, const arma::mat Y_label, const int num_classes)
{
    size_t N = Y_label.n_elem;//number of training elements
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&Y_label](size_t i1, size_t i2) { return Y_label[i1] < Y_label[i2]; });
    // result = Y_label[idx[0]] is the smallest
    //
    std::vector<size_t> borders(num_classes + 1);


    for (int i = 0; i < num_classes; i++) {
        borders[i] = (size_t) ((double) (i * N) / (double) num_classes);
    }
    borders[num_classes] = N;


    //one-vs-all or one-vs-the rest strategy implemented - for each class
    //another option can be one vs one in pairs
    //
    //Remark - K(left_idx,left_idx) is constant 1. usually
    double score = 0.;
#pragma omp parallel for reduction(+:score)
    for (int i = 0; i < num_classes; i++) {
        size_t N1 = borders[i + 1] - borders[i];
        size_t N2 = N - N1;

        double E12 = 0.;
        for (size_t j = 0; j < N1; j++) {
            size_t left_idx = idx[borders[i] + j];
            for (size_t k = 0; k < N2; k++) {
                size_t right_idx = (k < borders[i]) ? idx[k] : idx[borders[i + 1] + k - borders[i]];
                E12 += pow(K(left_idx, left_idx) - 2 * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
            }
        }
        E12 = E12 / (double) N1 / (double) N2;
        double E11 = 0.;
        for (size_t j = 0; j < N1; j++) {
            size_t left_idx = idx[borders[i] + j];
            for (size_t k = 0; k < N1; k++) {
                size_t right_idx = idx[borders[i] + k];
                E11 += pow(K(left_idx, left_idx) - 2 * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
            }
        }
        E11 = E11 / (double) N1 / (double) N1;
        double E22 = 0.;
        for (size_t j = 0; j < N2; j++) {
            size_t left_idx = (j < borders[i]) ? idx[j] : idx[borders[i + 1] + j - borders[i]];
            for (size_t k = 0; k < N2; k++) {
                size_t right_idx = (k < borders[i]) ? idx[k] : idx[borders[i + 1] + k - borders[i]];
                E22 += pow(K(left_idx, left_idx) - 2 * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
            }
        }
        E22 = E22 / (double) N2 / (double) N2;
        score += E12 / ((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22);
    }
    score = score / (double) num_classes;
    return score;
}


extern double score_kernel_vs_labels(const arma::mat &K, const arma::mat &Y, const int num_classes, size_t L, int algotype);

double compute_score_metrics(size_t N, double *distance_matrix, double *Y)
{

    arma::mat K(N, N);

    arma::mat Y_label(N, 1);

    std::memcpy(K.memptr(), distance_matrix, sizeof(double) * N * N);

    //K=arma::ones(arma::size(K))-K/(2.*4096.*4096.);

    std::memcpy(Y_label.memptr(), Y, sizeof(double) * N);
    const int num_classes = 16;
#if 1
    return compute_score_metrics_arma(K, Y_label, num_classes);
#else
    return score_kernel_vs_labels(K,Y_label,num_classes,N/num_classes/2,2);
#endif
}

extern int
cuda_kernel_xx(size_t total_len_features, int dim, size_t size_X, size_t startX, size_t startY, size_t numX, size_t numY, const double *X, /* double sigma, */ double lambda, double tau, double w_sum_sym,
               double *Z, double *&d_Zptr);

extern int
cuda_kernel_xy(size_t total_len_features, int dim, size_t size_X, size_t size_Y, size_t startX, size_t startY, size_t numX, size_t numY, const double *X, const double *Y, double sigma, double lambda,
               double tau, double w_sum_sym, double *Z, double *&d_Zptr);

extern int check_gpu_malloc(double *&d_X, size_t size_X);

extern int check_gpu_free(double *d_X);

extern int check_gpu_copy_to_device(double *d_X, double *h_X, size_t size_X);

extern int check_gpu_copy_to_host(double *h_X, double *d_X, size_t size_X);


int do_gpu_distance_compute_mat_xx(size_t size_X, int dimension, double *sub_X, double *Z, int counter, double *&d_Z, double beta, double tau, double gamma)
{
    arma::mat sub_X_arma(dimension, size_X);
    std::memcpy(sub_X_arma.memptr(), sub_X, dimension * size_X * sizeof(double));

    arma::mat cum_X = arma::cumsum(sub_X_arma);

    int dim = 1;

    size_t total_len_features = dimension;
    size_t startX = 0;
    size_t startY = 0;
    size_t numX = size_X;
    size_t numY = size_X;
    double sigma = gamma;
    double lambda = beta;

    double w_sum_sym = 1.;
    double *X = cum_X.memptr();

    cuda_kernel_xx(dimension, dim, size_X, startX, startY, numX, numY, X, /* sigma, */ lambda, tau, w_sum_sym, Z, d_Z);
    return 0;
}


double compute_score_metrics_cuda(size_t N, double *distance_matrix, double *Y)
{
    return score_distance_kernel(N, distance_matrix, Y);
}

double compute_metrics(size_t N, const double *distance_matrix, const double *Y)
{
    //Compute sum of number of inverstions of  distance_matrix  vs Y,
    //i.e. i , j, k where i closer to k than j by distance_matrix, but Y(j) farther
    std::cout << "Start inversions " << std::endl;
    size_t inversions = 0;
#pragma omp parallel for reduction(+:inversions)
    for (size_t i = 0; i < N; i++) {

        // initialize original index locations
        if (i % 1000 == 0) {
            std::cout << "Start  " << i << std::endl;
        }
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);

        std::stable_sort(idx.begin(), idx.end(),
                         [&](size_t j, size_t k) { return distance_matrix[i * N + j] < distance_matrix[i * N + k]; });
        //idx are now sorted according to distance with i^th element


        std::vector<double> Ydistances(N);
        for (size_t j = 0; j < N; j++) {
            Ydistances[j] = fabs(Y[idx[j]] - Y[idx[i]]);
        }
        //inversions+=get_inversions_count(Ydistances.data(),N);
        inversions += merge_compute_inversions(Ydistances.data(), N);
    }
    return (double) inversions;

}

double adjacent_coef_old(double gau, int i, int the_level, int levels)
{
    double result = 1. / (1 + gau * abs(i - the_level));
    if (the_level > 0) {
        return result;
    }
    if (i == levels - 1) {
        return 1. / (1. + gau * 1.);
    }
    return result;
}

double adjacent_coef(double gau, int i, int the_level, int levels)
{
    //return  1./(1.+gau*(double)std::min(std::abs(i-the_level-levels),std::min(std::abs(i-the_level),std::abs(i+levels-the_level))));
    return 1.;
}

size_t first_valid_offset;
size_t last_valid_offset;

#define MULTIPLE_GU 10
#define GU_UPPER_LIMIT  0.0001

#define GU_STEP_MULTIPLE  0.01
#define GU_SKIP 1

#define GU_PART_TRAIN 0.8
#define GU_NUMTRY   10


#define GAMMA_SPAN 10


int arma_multisolve(double epsilon, int Nrows, int lda, const double *Ainput, int start, const double *rhs, double *B);

arma::mat make_model(const arma::mat &K, const arma::mat &B, double epsilon)
{
    //return  qp_model(K,B,epsilon);
    if (0) {
        const int irwls_iter = 4;

        const double delta = 0.00001;
        if (1) {
            arma::mat x = call_gpu_solve(K + epsilon * arma::eye(arma::size(K)), B);
            for (int i = 0; i < irwls_iter; i++) {

                arma::mat errors = arma::abs(K * x - B);

                //arma::mat mult = 1./arma::sqrt(arma::max(errors, arma::ones(arma::size(B))*delta));
                arma::mat mult = 1. / arma::sqrt(errors + arma::ones(arma::size(B)) * delta / ((double) i + 1.));

                arma::mat Kmult = mult * arma::ones(1, arma::size(K, 1));

                arma::mat Left = Kmult % K;
                arma::mat Right = B % mult;

                arma::mat x_new = call_gpu_solve(Left + epsilon * arma::eye(arma::size(K)), Right);
                x = x_new;
            }
            return x;
        } else {

            size_t len = arma::size(K, 0);

            int iterations = 4;
            arma::mat Left = arma::zeros(2 * len, len);
            Left.rows(0, len - 1) = K;
            Left.rows(len, 2 * len - 1) = epsilon * arma::eye(len, len);
            arma::mat Right = arma::zeros(2 * len, 1);
            Right.rows(0, len - 1) = B;
            arma::mat new_sol = arma::zeros(2 * len, 1);
            arma::mat err_column = arma::ones(2 * len, 1);
            double best_result = std::numeric_limits<double>::max();
            arma::mat x = call_gpu_solve(K + epsilon * arma::eye(arma::size(K)), B);
            best_result = arma::as_scalar(arma::sum(arma::abs(K * x - B)) + epsilon * arma::sum(arma::abs(x)));
            for (int i = 0; i < iterations; i++) {
                arma::mat new_left = (err_column * arma::ones(1, len)) % Left;
                arma::mat new_right = err_column % Right;
                new_sol = call_gpu_oversolve(new_left, new_right);
                arma::mat predicted = Left * new_sol;
                arma::mat errors = (predicted - Right);
                err_column = arma::max(arma::abs(errors), delta * arma::ones(arma::size(errors)));
                double temp = arma::as_scalar(arma::sum(arma::abs(K * new_sol - B)) + epsilon * arma::sum(arma::abs(new_sol)));
                int count = 0;
                while (temp > best_result) {
                    new_sol = (new_sol + x) / 2;
                    temp = arma::as_scalar(arma::sum(arma::abs(K * new_sol - B)) + epsilon * arma::sum(arma::abs(new_sol)));
                    count++;
                    if (count > 8) break;
                }
                if (temp < best_result) {
                    x = new_sol;
                    best_result = temp;
                }
                predicted = Left * new_sol;
                errors = (predicted - Right);
                err_column = 1. / arma::sqrt(arma::max(delta * arma::ones(arma::size(errors)), errors));
            }
            return x;

        }
    } else {
        if (1) {
            return call_gpu_solve(K + epsilon * arma::eye(arma::size(K)), B);
        } else {
            arma::mat result = B * 0.;
            arma_multisolve(epsilon, arma::size(K, 0), arma::size(K, 0), K.memptr(), 0, B.memptr(), result.memptr());
            return result;
        }
    }
}


double best_eval_score = std::numeric_limits<double>::max();


double compute_metrics(size_t N, double *distance_matrix, double *Y);

double eval_score_ratio(const int numtries, int level, const std::vector<arma::mat> &X_IN, const arma::mat &Y, std::vector<double> params)
{

    arma::mat cum_X = arma::cumsum(X_IN[level]);

    int dim = 1;
    size_t dimension = arma::size(cum_X, 0);
    size_t size_X = arma::size(cum_X, 1);
    size_t total_len_features = dimension;
    size_t startX = 0;
    size_t startY = 0;
    double beta = params[0];
    double lambda = beta;
    double tau = params[1];
    double gamma = params[2];
    double sigma = gamma;
    double epsilon = params[3];
    //double delta = params[4];
    double w_sum_sym = 1.;
    double *X = cum_X.memptr();
    double *d_Z;

    const size_t model_train_size = 6000;
    const size_t model_test_size = 100;
    const size_t max_j = 12;
    const size_t start_skip = 100;
    size_t short_size_X = numtries * start_skip * max_j + model_train_size + model_test_size;
    /*BAD*/
    short_size_X = 6000;
    arma::mat Z(short_size_X, short_size_X);
    arma::mat shortY = Y.rows(0, short_size_X - 1);
    size_t numX = short_size_X;
    size_t numY = short_size_X;
    cuda_kernel_xx(dimension, dim, short_size_X, startX, startY, numX, numY, cum_X.memptr(), /* sigma, */ lambda, tau, w_sum_sym, Z.memptr(), d_Z);

    //check_gpu_free(d_Z);
    const int num_classes = 16;
    double score = compute_score_metrics_arma(Z, shortY, num_classes);
    return score;
}

double score_kernel_vs_labels_referent(const arma::mat &Z, const arma::mat &Y);

double score_kernel_vs_labels_referent_jarko(const arma::mat &Z, const arma::mat &Y);

double eval_score_referent_jarko1(const int numtries, int level, const std::vector<arma::mat> &X_IN, const arma::mat &Y, std::vector<double> params)
{

    arma::mat cum_X = arma::cumsum(X_IN[level]);
    int dim = 1;
    size_t dimension = arma::size(cum_X, 0);
    size_t size_X = arma::size(cum_X, 1);
    size_t total_len_features = dimension;
    size_t startX = 0;
    size_t startY = 0;
    double beta = params[0];
    double lambda = beta;
    double tau = params[1];
    double gamma = params[2];
    double sigma = gamma;
    double epsilon = params[3];
    //double delta = params[4];
    double w_sum_sym = 1.;
    double *X = cum_X.memptr();
    double *d_Z;

    size_t short_size_X = 3000;

    arma::mat Z(short_size_X, short_size_X);
    size_t numX = short_size_X;
    size_t numY = short_size_X;
    cuda_kernel_xx(dimension, dim, short_size_X, startX, startY, numX, numY, cum_X.memptr(), sigma, /* lambda, */ tau, w_sum_sym, Z.memptr(), d_Z);
    check_gpu_free(d_Z);
    arma::mat shortY = Y.rows(0, short_size_X - 1);
    return score_kernel_vs_labels_referent(Z, shortY);
}


//GLOBAL
arma::mat KKK;
double old_lambda = 0.;
double old_tau = 0.;

double eval_score_direct(const int numtries, int level, const std::vector<arma::mat> &X_IN, const arma::mat &Y, std::vector<double> params)
{

    arma::mat cum_X = arma::cumsum(X_IN[level]);

    int dim = 1;
    size_t dimension = arma::size(cum_X, 0);
    size_t size_X = arma::size(cum_X, 1);
    size_t total_len_features = dimension;
    size_t startX = 0;
    size_t startY = 0;
    double beta = round(params[0] * 1000) / 1000.;
    double lambda = beta;
    double tau = round(params[1] * 1000) / 1000.;
    double gamma = params[2];
    double sigma = gamma;
    double epsilon = params[3];
    double mu = params[4];
    double nu = params[5];
    //double delta = params[4];
    double w_sum_sym = 1.;
    double *X = cum_X.memptr();
    double *d_Z;

    const size_t model_train_size = 6000;
    const size_t model_test_size = 400;
    const size_t max_j = 12;
    //BAD const size_t start_skip=100;
    const size_t start_skip = 10;
    size_t short_size_X = numtries * start_skip * max_j + model_train_size + model_test_size;
    assert(short_size_X <= arma::size(Y, 0));

    arma::mat Z = arma::zeros(short_size_X, short_size_X);
    size_t numX = short_size_X;
    size_t numY = short_size_X;
    size_t all_levels = X_IN.size();
    if ((old_lambda != lambda) || (old_tau != tau)) {
        old_lambda = lambda;
        old_tau = tau;
        Z = arma::zeros(short_size_X, short_size_X);
        for (int j = 0; j < all_levels; j++) {
            arma::mat ZZ = arma::zeros(short_size_X, short_size_X);
            arma::mat cum_XX = arma::cumsum(X_IN[j]);
            cuda_kernel_xx(dimension, dim, short_size_X, startX, startY, numX, numY, cum_XX.memptr(), /* sigma, */ lambda, tau, w_sum_sym, ZZ.memptr(), d_Z);
            Z += ZZ;
        }
        KKK = Z;
    };

    Z = arma::ones(numX, numY) - KKK / (2 * sigma * sigma);
    /* if (fabs(Z(0,1))<0.01){
        return 2+1./fabs(Z(0,1));
    }*/

    double score = 0.;
    size_t cntr = 0;
    for (int j = max_j / 2; j < max_j; j++) {
        //#pragma omp parallel for reduction(+:score)
        for (int i = 0; i < numtries; i++) {
            size_t x_train_start = i * start_skip + j * (start_skip * numtries);
            size_t x_train_final = x_train_start + model_train_size;

            size_t x_test_start = x_train_final;
            size_t x_test_final = x_test_start + model_test_size;
            arma::mat K = Z.submat(x_train_start, x_train_start, x_train_final - 1, x_train_final - 1);
            arma::mat testK = Z.submat(x_test_start, x_train_start, x_test_final - 1, x_train_final - 1);

#pragma omp critical
            {
                std::cout << "numtri " << (cntr + i) << " " << arma::size(Y) << " " << arma::size(Z) << " " << x_train_start << " " << x_train_final << " " << x_test_start << "  " << x_test_final << std::endl;
            }
            arma::mat B = Y.submat(x_train_start, 0, x_train_final - 1, 0);

            arma::mat testB = Y.submat(x_test_start, 0, x_test_final - 1, 0);
            double result_mae = 0.;
            //arma::mat coefs = make_model(K,B,epsilon);
            arma::mat coefs = call_gpu_solve(K + epsilon * arma::eye(arma::size(K)), B);
            arma::mat errors = testK * coefs - testB;
            result_mae = arma::as_scalar(arma::sum(arma::abs(errors))) / arma::as_scalar(arma::sum(arma::abs(testB)));
#pragma omp critical
            {
                std::cout << "eps " << beta << " " << tau << " " << gamma << " " << epsilon << " " << arma::as_scalar(arma::sum(arma::abs(coefs))) << " " << result_mae << std::endl;
            }
            score += result_mae;
#define USING_GRID_SEARCH 1
#ifdef USING_GRID_SEARCH
            if (result_mae > 2 * best_eval_score) {
                return result_mae;
            }
#endif
        }
        cntr += numtries;
        //if (score/(double)cntr > best_eval_score){
        //break;
        //}
    }
    if (score / (double) cntr < best_eval_score) {
        best_eval_score = score / (double) cntr;
    }
    return score / (double) cntr;
}

double eval_score(const int numtries, int level, const std::vector<arma::mat> &X_IN, const arma::mat &Y, std::vector<double> params)
{
    const int which_type = 0;
    double score = 0.;
    switch (which_type) {
        case 0:
            score = eval_score_direct(numtries, level, X_IN, Y, params);
            break;
            /*
            case 1: score= eval_score_inverse(numtries,level,X_IN,Y,params);
                            break;
            case 2: score= eval_score_ratio(numtries,level,X_IN,Y,params);
                            break;
            case 3: score= eval_score_referent_china(numtries,level,X_IN,Y,params,1);
                            break;
            case 4: score= eval_score_referent_china(numtries,level,X_IN,Y,params,2);
                            break;
            case 5: score= eval_score_referent_jarko1(numtries,level,X_IN,Y,params);
                            break;
            case 6: score= eval_score_referent_jarko2(numtries,level,X_IN,Y,params);
                            break;
            */
        default:
            score = std::numeric_limits<double>::max();
    }
    return score;
}

typedef double compute_score_func_t(const std::vector<double> &);

typedef std::function<compute_score_func_t> score_function_t;

extern std::vector<double>
optimise_params_pso(int pso_topol_select, int pso_iter, int pso_particlenum, score_function_t func, int numtries, const std::vector<double> &lower_limit, const std::vector<double> &upper_limit);

extern std::vector<double> optimise_params_grid(score_function_t func, int numtries, const std::vector<std::vector<double>> &discretized_points);


extern size_t start_from;


std::vector<double> optimise_params_grid(score_function_t func, int numtries, const std::vector<std::vector<double>> &discretized_points)
{
    size_t dim = discretized_points.size();
    double best_result = std::numeric_limits<double>::infinity();
    std::vector<double> result;

    size_t num_points = 1;
    for (size_t i = 0; i < dim; i++) {
        num_points = num_points * discretized_points[i].size();
    }
    std::vector<double> point(dim);
    std::vector<size_t> idx(dim, 0);
    for (size_t cntr = 0; cntr < num_points; cntr++) {
        for (size_t i = 0; i < dim; i++) {
            point[i] = discretized_points[i][idx[i]];
        }
        double val = func(point);
        if (val < best_result) {
            best_result = val;
            result = point;
        }
        for (size_t i = dim - 1; i >= 0; i--) {
            idx[i]++;
            if (idx[i] >= discretized_points[i].size()) {
                idx[i] = 0;
            } else {
                break;
            }
        }
    }
    return result;
}

arma::mat make_ideal_distance(const arma::mat &Y)
{
    //find the right gamma here
    size_t N = arma::size(Y, 0);
    arma::mat Z(N, N, arma::fill::zeros);
    //#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            Z(i, j) = fabs(Y(i, 0) - Y(j, 0));
        }
    }
    return Z;
}

arma::mat make_kernel_distance(const arma::mat &Z)
{
    //find the right gamma here
    size_t N = arma::size(Z, 0);
    arma::mat D(N, N, arma::fill::zeros);
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            D(i, j) = 0.5 * (Z(i, i) + Z(j, j) - 2 * Z(i, j));
        }
    }
    return Z;
}


double score_kernel_vs_labels_referent(const arma::mat &Z, const arma::mat &Y)
{
    arma::mat ideal_distance_Y = make_ideal_distance(Y);
    arma::mat real_distance_Z = make_kernel_distance(Z);
    double ymean_dist = arma::as_scalar(arma::mean(arma::mean(ideal_distance_Y)));
    double zmean_dist = arma::as_scalar(arma::mean(arma::mean(real_distance_Z)));
    arma::mat ematY = ideal_distance_Y / ymean_dist;
    arma::mat ematZ = real_distance_Z / zmean_dist;
    return arma::as_scalar(arma::mean(arma::mean(arma::abs(ematY - ematZ))));
}


int opti_param(
        int numtries, const std::vector<std::vector<double>> &xtrain, const std::vector<std::vector<double>> &ytrain,
        int the_level, int adjacent_left, int adjacent_right, std::vector<double> parameters)
{
    int levels = xtrain.size();
    size_t total_len = ytrain[0].size();
    std::cout << "opti param " << numtries << " " << levels << " " << xtrain.size() << " " << total_len << std::endl;

    size_t traintest_len = 11000;//BAD!
    size_t t_start = 0;//???? BAD
    //assert(traintest_len+t_start==11000);
    size_t dimension = xtrain[0].size() / total_len;
    size_t back_len = dimension; // lag
    std::cout << "opti param " << back_len << " " << traintest_len << std::endl;
    arma::mat X(back_len, traintest_len, arma::fill::zeros);
    std::cout << "opti param " << traintest_len << std::endl;
    arma::mat Y(traintest_len, 1, arma::fill::zeros);
    for (size_t i = 0; i < traintest_len; i++) {
        Y(i, 0) = ytrain[the_level][t_start + i];
    }
    std::vector<arma::mat> X_IN(levels);
    for (int i = 0; i < levels; i++) {
        X_IN[i] = X;
        for (size_t j = 0; j < traintest_len; j++) {
            for (int k = 0; k < back_len; k++) {
                X_IN[i](k, j) = xtrain[i][k + (t_start + j) * back_len];
            }
        }
    }
    double times0 = msecs();
    double *d_Z;
    int num_levels = 0;
    arma::mat Z(traintest_len, traintest_len);
    double best_result = std::numeric_limits<double>::max();
    const int pso_topol_select = 0;//ring
    const int pso_iter = 30;
    const int pso_particlenum = 100;
    std::vector<double> lower_limit, upper_limit, best_values;
    //lower_limit = {0.75000,0.250001, 1.,0.8e-11};
    //upper_limit = {0.75001,0.250002, 100000000., 8.192e-06};
    //lower_limit = {0.75000,0.250001, 1.,0.8e-12};
    //upper_limit = {0.75001,0.250002, 10000000., .00001};
    lower_limit = {0.75000, 0.250001, 1., 0.8e-12};
    upper_limit = {0.75001, 0.250002, 10000000., .00001};
    const score_function_t func = [&](const std::vector<double> &params) -> double {
        double result = eval_score(numtries, the_level, X_IN, Y, params);
        std::cout << "fitness " << result << " ";
        for (size_t i = 0; i < params.size(); i++) {
            std::cout << params[i] << " ";
        }
        std::cout << std::endl;

        //std::cout << "fitness " << params[0] << " " <<  params[1] << " " << params[2] << " " << params[3] << " " << result << std::endl;
        return result;
    };
    int optimisation_algorithm = 2;
    best_eval_score = std::numeric_limits<double>::max();
    switch (optimisation_algorithm) {
        case 0: {
            best_values = optimise_params_pso(pso_topol_select, pso_iter, pso_particlenum, func, numtries, lower_limit, upper_limit);
            break;
        }
        case 1: {
            std::vector<std::vector<double>> discretization_points({
                                                                           //{0.25,0.5,0.75,1.},
                                                                           {0.75},//lambda
                                                                           {0.25},
                                                                           //{1,2,4,8,16,32,64,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216},
                                                                           {1,       4,       16,       64,       256,      1024,      4096,          16384,          65536,          262144, 1048576, 4194304, 16777216},
                                                                           //{(1./5.26042e-06)/2,1./5.26042e-06 , 1./5.26042e-06*2,1/(4.64353e-07/2.),1./4.64353e-07,1/(4.64353e-07*2)},
                                                                           //{0.8e-11,1.6e-11,3.2e-11,6.4e-11,1.28e-10,2.56e-10,5.12e-10,1.024e-9,2.048e-9,4.096e-08,8.192e-08}
                                                                           {0.8e-11, 3.2e-11, 1.28e-10, 5.12e-10, 2.048e-9, 8.192e-08, 8.192e-08 * 4, 8.192e-08 * 16, 8.192e-08 * 64, 8.192e-08 * 256}
                                                                   });
            assert(discretization_points.size() == 4);
            best_values = optimise_params_grid(func, numtries, discretization_points);
            break;
        }
        case 2: {
            const int opt_particles = 10;
            const int max_iterations_opt = 100;
            const double ffa_alpha = 0.5;
            const double ffa_betamin = 1.;
            const double ffa_gamma = 1.;
            int dim = lower_limit.size();
            best_values = firefly(dim, opt_particles, max_iterations_opt, ffa_alpha, ffa_betamin, ffa_gamma, lower_limit, upper_limit, func);
            break;
        }
    }
    return 0;
}
