#pragma once

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

#define PATHS_AVERAGE

namespace svr {
namespace kernel {

template<typename T> class kernel_deep_path : public kernel_base<T>
{
public:
    explicit kernel_deep_path(const datamodel::SVRParameters &p);

    explicit kernel_deep_path(const kernel_base<T> &k);


    void init_manifold(const mat_ptr X, const mat_ptr Y);

    arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const override;

    arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    void d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const override;
};

}
}