//
// Created by zarko on 16/06/2025.
//

#ifndef KERNEL_GBM_HPP
#define KERNEL_GBM_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {

template<typename T> class kernel_gbm final : public kernel_base<T>
{
public:
    void init(datamodel::OnlineSVR &svrmod, uint32_t chunk_ix) override;

    explicit kernel_gbm(datamodel::SVRParameters &p);

    arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    void d_kernel(CRPTR (T) d_Z, uint32_t m, RPTR (T) d_K, cudaStream_t custream) const override;

    void d_distances(CRPTR (T) d_X, CRPTR (T) &d_Xy, uint32_t m, uint32_t n_X, uint32_t n_Xy, RPTR (T) d_Z, cudaStream_t custream) const override;
    void update(datamodel::OnlineSVR &svrmod, uint32_t chunk_ix, const arma::Mat<T> &x, const arma::Mat<T> &y) override;
};

std::string get_lgbm_core_parameters(uint16_t gpu_id);

std::string get_lgbm_dataset_parameters();

} // kernel
} // svr

#endif //KERNEL_GBM_HPP
