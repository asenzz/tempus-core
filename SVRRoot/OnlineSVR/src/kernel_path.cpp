//
// Created by zarko on 19/03/2025.
//

#include <armadillo>
#include "kernel_path.hpp"
#include "util/math_utils.hpp"
#include "common/parallelism.hpp"
#include "cuda_path.hpp"

namespace svr {
namespace kernel {

#define T double

template<>
arma::Mat<T> kernel_path<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    arma::Mat<T> Ky(X.n_cols, Xy.n_cols, arma::fill::none);
    kernel::path::kernel_xy(X.n_cols, Xy.n_cols, X.n_rows, parameters.get_lag_count(), parameters.get_svr_kernel_param(), parameters.get_svr_kernel_param2(),
                            parameters.get_kernel_param3(), parameters.get_min_Z(), X.mem, Xy.mem, Ky.memptr());
    return Ky;
}

template<>
arma::Mat<T> kernel_path<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    arma::Mat<T> Zy(X.n_cols, Xy.n_cols, arma::fill::none);
    kernel::path::distances_xy(
            X.n_cols, Xy.n_cols, X.n_rows, parameters.get_lag_count(), parameters.get_svr_kernel_param2(), parameters.get_kernel_param3(), X.mem, Xy.mem, Zy.memptr());
    return Zy;
}

template<>
void kernel_path<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    kernel::d_kernel_from_distances(d_K, d_Z, m, m, parameters.get_svr_kernel_param(), parameters.get_min_Z(), custream);
}

template<>
void kernel_path<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    const auto lag = parameters.get_lag_count();
    kernel::path::cu_distances_xy(n_X, n_Xy, m, lag, m / lag, CDIVI(lag, common::C_cu_tile_width), parameters.get_svr_kernel_param2(), parameters.get_kernel_param3(), d_X, d_Xy,
                                  d_Z, custream);
}


template<> // Path kernel takes cumulatives of features
arma::Mat<T> kernel_path<T>::all_cumulatives(const arma::Mat<T> &features_t) const
{
    const auto lag = parameters.get_lag_count();
    assert(features_t.n_rows % lag == 0);
    const DTYPE(lag) levels = features_t.n_rows / lag;
    arma::mat cuml(arma::size(features_t), arma::fill::none);
    OMP_FOR_i(levels) {
        const auto start_row = i * lag;
        const auto end_row = std::min<DTYPE(start_row) >(start_row + lag, features_t.n_rows) - 1;
        cuml.rows(start_row, end_row) = arma::cumsum(features_t.rows(start_row, end_row));
    }
    LOG4_TRACE("Prepared " << levels << " cumulatives " << common::present(cuml) << " with " << lag << " lag, from features t " << common::present(features_t));
    return cuml;
}

template<> // Separated in deque elements
std::shared_ptr<std::deque<arma::Mat<T>>> kernel_path<T>::prepare_cumulatives(const arma::Mat<T> &features_t) const
{
    const auto lag = parameters.get_lag_count();
    const DTYPE(lag) levels = features_t.n_rows / lag;
    const auto p_cums = ptr<std::deque<arma::Mat<T>>>(levels);
    OMP_FOR_i(levels) p_cums->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, parameters " << parameters << ", from features_t " << arma::size(features_t));
    return p_cums;
}

}
}