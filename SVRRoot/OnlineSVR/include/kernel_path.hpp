#ifndef SVR_KERNEL_PATH_HPP
#define SVR_KERNEL_PATH_HPP

#include "common/gpu_handler.hpp"
#include "kernel_base.hpp"

/*
 * Kernel matrix format, where x is X_cols and y is Y_cols:
 * output size X_cols x Y_cols

 L0 - L0, L0 - L1, L0 - L2, ... L0 - Ly
 L1 - L0, L1 - L1, L1 - L2, ... L1 - Ly
 L2 - L0, L2 - L1, L2 - L2, ... L2 - Ly
 ...
 Lx - L0, Lx - L1, Lx - L2, ... Lx - Ly
*/

// #define PATHS_AVERAGE

namespace svr {
namespace kernel {

template<typename T> class kernel_path : public kernel_base<T>
{
public:
    explicit kernel_path(const datamodel::SVRParameters &p);

    explicit kernel_path(const kernel_base<T> &k);


    arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const override;

    static void cu_distances_xy(
        const uint32_t X_cols, const uint32_t Xy_cols, const uint16_t lag, const uint16_t dim, float tau, const float H, const float D, const float V,
        CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z, const cudaStream_t custream);

    static void distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, float tau, const float H, const float D, const float V,
                             CRPTR(T) X, CRPTR(T) Xy,RPTR(T) Z);


    arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    void d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const override;

    static void cu_kernel_xy(
        const uint32_t X_cols, const uint32_t Xy_cols, const uint16_t lag, const uint16_t dim, const T gamma, const T mean, float lambda, float tau,
        const float H, const float D, const float V, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K, const cudaStream_t custream);

    static void kernel_xy(
        const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const T gamma, const T mean, float lambda, float tau,
        const float H, const float D, const float V, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K);
};

constexpr uint8_t C_dist_pow = 4;
#define DIST(x) pow(abs(x), C_dist_pow)


#ifdef DEPRECATED_KERNEL_API

/* original labels to features before matrix transposition
  = {   label 0 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ,
        label 1 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ..
        ...
        label 6000 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 }
     }
*/

template<typename T> class kernel_path : public kernel_base<T> {
public:
    static constexpr T C_default_tau = .25;

    // Path kernel takes cumulatives of features
    std::shared_ptr<arma::Mat<T>> all_cumulatives(const arma::Mat<T> &features_t)
    {
        const auto lag = p.get_lag_count();
        const DTYPE(lag) levels = features_t.n_rows / lag;
        const auto p_cuml = ptr<arma::mat>(arma::size(features_t), ARMA_DEFAULT_FILL);
        OMP_FOR_i(levels) p_cuml->rows(i * lag, (i + 1) * lag - 1) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
        LOG4_TRACE("Prepared " << levels << " cumulatives " << common::present(*p_cuml) << " with " << lag << " lag, from features t " << common::present(features_t));
        return p_cuml;
    }

    // Separated in deque elements
    std::shared_ptr<std::deque<arma::Mat<T>>> prepare_cumulatives(const SVRParameters &params, const arma::Mat<T> &features_t)
    {
        const auto lag = params.get_lag_count();
        const DTYPE(lag) levels = features_t.n_rows / lag;
        const auto p_cums = ptr<std::deque<arma::Mat<T>>>(levels);
        OMP_FOR_i(levels) p_cums->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
        LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, parameters " << params << ", from features_t " << arma::size(features_t));
        return p_cums;
    }

    explicit kernel_path(const SVRParameters &p) : kernel_base<T>(p) {}
};

#endif
}
}

#endif //SVR_KERNEL_PATH_HPP
