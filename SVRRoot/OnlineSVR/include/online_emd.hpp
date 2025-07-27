#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "model/DataRow.hpp"
#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "model/DeconQueue.hpp"
#include "model/InputQueue.hpp"
#include "common/cuda_util.cuh"

// Algorithm to use FIR OEMD is default (slower and higher quality) or FFT OEMD faster but loss of precision on reconstruction
// #define OEMDFFT
#define OEMD_CUDA

namespace svr {
namespace oemd {

class online_emd final : public spectral_transform {
    const uint16_t levels;
    const double stretch_coef;
    static t_coefs_cache oemd_coefs_cache;

    static void transform(
            const datamodel::datarow_crange &in,
            datamodel::datarow_range &out,
            const std::vector<double> &tail,
            const std::deque<uint16_t> &siftings,
            const std::deque<std::vector<double>> &mask,
            const double stretch_coef,
            uint16_t oemd_levels,
            uint16_t in_colix_,
            const datamodel::t_iqscaler &scaler);

public:
    explicit online_emd(uint16_t levels, double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    t_oemd_coefficients_ptr get_masks(const datamodel::datarow_crange &input, const std::vector<double> &tail, std::string queue_name, uint16_t in_colix_,
                                      const datamodel::t_iqscaler &scaler, const boost::posix_time::time_duration &resolution,
                                      const boost::posix_time::time_duration &main_resolution) const;

    void transform(const std::vector<double> &input, std::vector<std::vector<double>> &decon,
                   const size_t padding /* = 0 */) override
    {}; // Dummy

    void transform(datamodel::DeconQueue &decon_queue, uint32_t decon_start_ix, uint32_t test_offset, uint32_t custom_residuals_ct,
                   const boost::posix_time::time_duration &resolution, const boost::posix_time::time_duration &main_resolution);

    void transform(const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue, uint16_t in_colix, uint32_t test_offset,
                   const datamodel::t_iqscaler &scaler, uint32_t custom_residuals_ct, const boost::posix_time::time_duration &main_resolution);

    void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon, const size_t padding /* = 0 */) const override;

    static uint32_t get_residuals_length(double _stretch_coef = oemd_coefficients::C_oemd_stretch_coef, uint16_t siftings = oemd_coefficients::C_default_siftings);

    static uint32_t get_residuals_length(const oemd_coefficients &coefs, double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    uint32_t get_residuals_length(const std::string &queue_name) const noexcept;

    static void expand_the_mask(uint32_t mask_size, uint32_t input_size, CPTRd dev_mask, double *dev_expanded_mask, cudaStream_t custream);
};

__global__ void G_subtract_I(RPTR(double) x, double y, uint32_t n);

__global__ void G_subtract_I(RPTR(double) x, CRPTRd y, uint32_t n);

__global__ void G_apply_fir(double stretch_coef, CRPTRd in, uint32_t len, CRPTRd mask, uint32_t mask_len, uint32_t stretched_mask_len, RPTR(double) out, uint32_t in_start);

}
}
