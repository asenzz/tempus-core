#pragma once

#include <vector>
#include <common/cuda_util.cuh>

#include "model/DataRow.hpp"
#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "common/defines.h"
#include "model/DeconQueue.hpp"
#include "model/InputQueue.hpp"

// Algorithm to use FIR OEMD is default (slower and higher quality) or FFT OEMD faster but loss of precision on reconstruction
// #define OEMDFFT
#define OEMD_CUDA

namespace svr {
namespace oemd {

class online_emd final : public spectral_transform {
    const unsigned levels;
    const double stretch_coef;
    static t_coefs_cache oemd_coefs_cache;

    static void transform(
            const datamodel::datarow_crange &in,
            datamodel::datarow_range &out,
            const std::vector<double> &tail,
            const std::deque<unsigned> &siftings,
            const std::deque<std::vector<double>> &mask,
            const double stretch_coef,
            const unsigned oemd_levels,
            const unsigned in_colix_,
            const datamodel::t_iqscaler &scaler);

public:
    explicit online_emd(const unsigned levels, const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    t_oemd_coefficients_ptr get_masks(const datamodel::datarow_crange &input, const std::vector<double> &tail, const std::string &queue_name, const unsigned in_colix_,
                                      const datamodel::t_iqscaler &scaler, const boost::posix_time::time_duration &resolution,
                                      const boost::posix_time::time_duration &main_resolution) const;

    void transform(const std::vector<double> &input, std::vector<std::vector<double>> &decon,
                   const size_t padding /* = 0 */) override
    {}; // Dummy

    void transform(datamodel::DeconQueue &decon_queue, const unsigned decon_start_ix, const unsigned test_offset, const unsigned custom_residuals_ct,
                   const boost::posix_time::time_duration &resolution, const boost::posix_time::time_duration &main_resolution);

    void transform(const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue, const unsigned in_colix, const unsigned test_offset,
                   const datamodel::t_iqscaler &scaler, const unsigned custom_residuals_ct, const boost::posix_time::time_duration &main_resolution);

    void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon, const size_t padding /* = 0 */) const override;

    static unsigned get_residuals_length(const double _stretch_coef = oemd_coefficients::C_oemd_stretch_coef, const unsigned siftings = oemd_coefficients::default_siftings);

    static unsigned get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    unsigned get_residuals_length(const std::string &queue_name);

    static void expand_the_mask(const unsigned mask_size, const unsigned input_size, CPTR(double) dev_mask, double *const dev_expanded_mask, const cudaStream_t custream);
};

__global__ void G_subtract_inplace(RPTR(double) x, const double y, const unsigned n);

__global__ void G_subtract_inplace(RPTR(double) x, CRPTR(double) y, const unsigned n);

__global__ void G_apply_fir(
        const double stretch_coef,
        CRPTR(double) in,
        const unsigned len,
        CRPTR(double) mask,
        const unsigned mask_len,
        const unsigned stretched_mask_len,
        RPTR(double) out,
        const unsigned in_start);

}
}
