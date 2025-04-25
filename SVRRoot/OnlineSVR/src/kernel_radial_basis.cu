//
// Created by zarko on 12/04/2025.
//
#if 0
#include "kernel_radial_basis.hpp"

#define T double

namespace svr {
namespace kernel {

template<> kernel_radial_basis<T>::d_distances(CRPTR(T) X, CRPTR(T), Xy, m, n_X, n_Xy, RPTR(Z), const cudaStream_t custream) const
{

}

}
}
#endif