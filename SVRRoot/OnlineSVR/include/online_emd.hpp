#pragma once

#include <vector>
#include <common/cuda_util.cuh>

#include "model/DataRow.hpp"
#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "common/defines.h"
#include "model/DeconQueue.hpp"

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
            datamodel::datarow_range &inout,
            const std::vector<double> &tail,
            const std::deque<unsigned> &siftings,
            const std::deque<std::vector<double>> &mask,
            const double stretch_coef);

public:
    explicit online_emd(const unsigned levels, const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    t_oemd_coefficients_ptr get_masks(const datamodel::datarow_range &input, const std::vector<double> &tail, const std::string &queue_name) const;

    void transform(const std::vector<double> &input, std::vector<std::vector<double>> &decon,
                   const size_t padding /* = 0 */) override
    {}; // Dummy

    void transform(
            datamodel::DeconQueue &decon_queue,
            const unsigned decon_start_ix = 0,
            const unsigned test_offset = 0,
            const unsigned custom_residuals_ct = std::numeric_limits<unsigned>::max());

    void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon, const size_t padding /* = 0 */) const override;

    static unsigned get_residuals_length(const double _stretch_coef = oemd_coefficients::C_oemd_stretch_coef, const unsigned siftings = oemd_coefficients::default_siftings);

    static unsigned get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef);

    unsigned get_residuals_length(const std::string &queue_name);

    static void expand_the_mask(const unsigned mask_size, const unsigned input_size, const double *dev_mask, double *dev_expanded_mask, const cudaStream_t custream);
};

__global__ void G_subtract_inplace(double *__restrict__ const x, CRPTR(double) y, const unsigned n);

__global__ void G_apply_mask(
        const double stretch_coef,
        CRPTR(double) rx,
        const unsigned x_size,
        CRPTR(double) mask,
        const unsigned mask_size,
        const unsigned stretched_mask_size,
        double *__restrict__ const rx2,
        const unsigned start_x);

}
}
