#ifndef SVR_KERNEL_PATH_HPP
#define SVR_KERNEL_PATH_HPP

#include <vector>
#include "common/gpu_handler.hpp"
#include "kernel_base.hpp"
#include "cuda_path.hpp"

namespace svr {
namespace kernel {

template<typename T>
class kernel_path : public kernel_base<T> {
public:
    static constexpr T C_default_tau = .25;

    arma::Mat<T> all_cumulatives(const arma::Mat<T> &features_t) const;

    std::shared_ptr<std::deque<arma::Mat<T>>> prepare_cumulatives(const arma::Mat<T> &features_t) const;

    explicit kernel_path(const datamodel::SVRParameters &p) : kernel_base<T>(p)
    {}

    virtual arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const;

    virtual arma::Mat<T> distances(const arma::Mat<T> &X_cuml, const arma::Mat<T> &Xy) const;

    virtual void d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const;

    virtual void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const;
};


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
        const auto p_cuml = ptr<arma::mat>(arma::size(features_t), arma::fill::none);
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

//     kernel::path::cu_distances_xy(train_len, train_len, train_F_rows, lag, dim, lag_tile_width, lambda, tau, dx_.d_train_cuml, dx_.d_train_cuml, dxsx.d_K_train, dxsx.custream);

#endif

}
}

#endif //SVR_KERNEL_PATH_HPP