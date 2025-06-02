//
// Created by zarko on 19/03/2025.
//

#include <oneapi/mkl/types.hpp>
#include <armadillo>
#include "kernel_deep_path.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace kernel {
// Specializations
#define T double

template<> kernel_deep_path<T>::kernel_deep_path(const datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

template<> kernel_deep_path<T>::kernel_deep_path(const kernel_base<T> &k) : kernel_base<T>(k.get_parameters())
{
}

template<> void kernel_deep_path<double>::init_manifold(const mat_ptr X, const mat_ptr Y)
{
    // STUB
}


template<> arma::Mat<T> kernel_deep_path<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    auto p_manifold = parameters.get_manifold();
    assert(p_manifold);
    // X and Xy are transposed therefore the acrobatics below
    arma::mat Ky(X.n_cols, Xy.n_cols);
    OMP_FOR(X.n_cols)
    for (uint32_t i = 0; i < X.n_cols; ++i) Ky.row(i) = p_manifold->predict_t(arma::join_cols(common::extrude_cols(X.col(i), Xy.n_cols), Xy)).t();
    LOG4_TRACE("Prepared kernel " << common::present(Ky) << " with parameters " << parameters << ", from X " << common::present(X) << " and Xy " << common::present(Xy));
    return Ky;
}

template<> arma::Mat<T> kernel_deep_path<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    LOG4_THROW("This kernel does not implement distances.");
    return {};
}


template<> void kernel_deep_path<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    LOG4_THROW("This kernel does not implement distances.");
}

template<> void kernel_deep_path<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    LOG4_THROW("This kernel does not implement distances.");
}

}
}
