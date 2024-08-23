//
// Created by zarko on 5/29/22.
//

#include <set>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "calc_kernel_inversions.hpp"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"


namespace svr {

double score_distance_kernel(const size_t sizeX, double *Z_distances, double *Y)
{
    // labels = Y
    // kernel matrix - Z
    /* std::vector<double> Z_distances(sizeX*sizeX);
    for(int i=0;i<sizeX;i++){
            for(int j=0;j<sizeX;j++){
                    Z_distances[i*sizeX+j]=2.*(1.-Z[i*sizeX+j]);
            }
    }
    */
    size_t N1 = 0;
    size_t N2 = 0;
    for (size_t i = 0; i < sizeX; i++) {
        if (Y[i] < 0) ++N1;
        else if (Y[i] > 0) ++N2;
    }
    size_t N = N1 + N2;
    double E12 = 0.;
    size_t i_ctr = 0;
    for (size_t i = 0; i < N1; i++) {
        while (Y[i_ctr] >= 0) ++i_ctr;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N2; j++) {
            while (Y[j_ctr] <= 0) ++j_ctr;
            E12 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            ++j_ctr;
        }
        ++i_ctr;
    }
    E12 = E12 / (double) N1 / (double) N2;
    double E11 = 0;
    i_ctr = 0;
    for (size_t i = 0; i < N1; i++) {
        while (Y[i_ctr] >= 0) ++i_ctr;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N1; j++) {
            while (Y[j_ctr] >= 0) ++j_ctr;
            E11 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            ++j_ctr;
        }
        ++i_ctr;
    }
    E11 = E11 / double(N1) / double(N1);
    double E22 = 0;
    i_ctr = 0;
    for (size_t i = 0; i < N2; i++) {
        while (Y[i_ctr] <= 0) ++i_ctr;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N2; j++) {
            while (Y[j_ctr] <= 0) ++j_ctr;
            E22 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            ++j_ctr;
        }
        ++i_ctr;
    }
    E22 = E22 / double(N2) / double(N2);
    return E12 / ((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22);
    //alternative?
    //return E12 - (((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22)) / 2.;
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


size_t merge(double *arr, double *temp, const size_t left, const size_t mid, const size_t right)
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
size_t _merge_compute_inversions(double *arr, double *temp, const size_t left, const size_t right)
{
    size_t inv_count = 0;
    if (right > left) {
        /* Divide the array into two parts and
        call _mergeSortAndCountInv()
        for each of the parts */
        const size_t mid = (right + left) / 2;

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
    std::vector<double> temp(N);
    return _merge_compute_inversions(arr, temp.data(), 0, N - 1);
}


calc_kernel_inversions::calc_kernel_inversions(const double *Z, const arma::mat &Y)
{
    //Compute sum of number of inversions of the distance_matrix vs Y,
    //i.e. i , j, k where i closer to k than j by distance_matrix, but Y(j) farther
    size_t inversions = 0;
    const size_t N = Y.n_rows;
#pragma omp parallel for reduction(+:inversions) num_threads(adj_threads(N))
    for (size_t i = 0; i < N; ++i) {
        // initialize original index locations
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(C_default_exec_policy, idx.begin(), idx.end(),
                         [&](const size_t j, const size_t k) { return Z[i * N + j] > Z[i * N + k]; });
        //idx are now sorted according to distance with i^th element
        std::vector<double> Ydistances(N);
        for (size_t j = 0; j < N; ++j)
            Ydistances[j] = std::fabs(arma::mean(Y.row(idx[j]) - Y.row(idx[i])));
        //merge_compute_inversions is faster, gives the same result
        //inversions+=get_inversions_count(Ydistances.data(),N);
        inversions += merge_compute_inversions(Ydistances.data(), N);
    }
    _weight = (double) inversions;
}

}
