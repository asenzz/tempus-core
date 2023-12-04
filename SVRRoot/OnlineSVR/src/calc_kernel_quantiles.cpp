//
// Created by zarko on 5/29/22.
//
#include "calc_kernel_quantiles.hpp"
#include "common/compatibility.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <common.hpp>


namespace svr {

calc_kernel_quantiles::calc_kernel_quantiles(const arma::mat &K, const arma::mat &Y, const size_t num_classes)
{
    const size_t N = Y.n_elem; // number of training elements
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&Y](const size_t i1, const size_t i2) { return Y[i1] < Y[i2];});
    // result = Y[idx[0]] is the smallest
    //
    std::vector<size_t> borders(num_classes + 1);
    __tbb_pfor_i (0, num_classes, borders[i] = size_t(double(i * N) / double(num_classes)) )
    borders[num_classes] = N;

    //one-vs-all or one-vs-the rest strategy implemented - for each class
    //another option can be one vs one in pairs
    //
    //Remark - K(left_idx,left_idx) is constant 1. usually
    const auto score = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_classes), double(0), [&](const tbb::blocked_range<size_t> &r, double score)
    {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            const size_t N1 = borders[i + 1] - borders[i];
            const size_t N2 = N - N1;

            double E12 = 0;
            for (size_t j = 0; j < N1; ++j) {
                const size_t left_idx = idx[borders[i] + j];
                for (size_t k = 0; k < N2; ++k) {
                    const size_t right_idx = k < borders[i] ? idx[k] : idx[borders[i + 1] + k - borders[i]];
                    E12 += std::pow(K(left_idx, left_idx) - 2. * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
                }
            }

            E12 = E12 / (double) N1 / (double) N2;
            double E11 = 0;
            for (size_t j = 0; j < N1; ++j) {
                const size_t left_idx = idx[borders[i] + j];
                for (size_t k = 0; k < N1; ++k) {
                    const size_t right_idx = idx[borders[i] + k];
                    E11 += std::pow(K(left_idx, left_idx) - 2. * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
                }
            }
            E11 = E11 / (double) N1 / (double) N1;
            double E22 = 0;
            for (size_t j = 0; j < N2; ++j) {
                const size_t left_idx = j < borders[i] ? idx[j] : idx[borders[i + 1] + j - borders[i]];
                for (size_t k = 0; k < N2; ++k) {
                    const size_t right_idx = k < borders[i] ? idx[k] : idx[borders[i + 1] + k - borders[i]];
                    E22 += std::pow(K(left_idx, left_idx) - 2. * K(left_idx, right_idx) + K(right_idx, right_idx), 2);
                }
            }
            E22 = E22 / (double) N2 / (double) N2;
            score += E12 / ((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22);
        }
        return score;
    }, std::plus<double>() );
    _weight = score;// / double(num_classes);
    LOG4_DEBUG("Returning " << score << " for " << num_classes);
}

}
