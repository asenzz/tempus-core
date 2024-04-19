#pragma once

#include <vector>
#include "model/DataRow.hpp"
#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "common/defines.h"
#include "model/DeconQueue.hpp"
#include "oemd_coefficients_search.hpp"

// Algorithm to use FIR OEMD is default (slower and higher quality) or FFT OEMD faster but loss of precision on reconstruction
// #define OEMDFFT
#define OEMD_CUDA

namespace svr {
namespace oemd {

constexpr double oemd_epsilon = 0.0000011;
constexpr double oemd_mask_sum = 1.00000;

using oemd_fptr = void (*)(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);

class online_emd final : public spectral_transform
{
    const size_t levels;
    const double stretch_coef;
    static t_coefs_cache oemd_coefs_cache;

    static void transform(
            datamodel::datarow_range &inout,
            const std::vector<double> &tail,
            const std::deque<size_t> &siftings,
            const std::deque<std::vector<double>> &mask,
            const double stretch_coef);

public:
    explicit online_emd(const size_t levels, const double stretch_coef = OEMD_STRETCH_COEF);


    t_oemd_coefficients_ptr get_masks(
            const datamodel::datarow_range &input, const std::vector<double> &tail, const std::string &queue_name) const;

    void transform(const std::vector<double> &input, std::vector<std::vector<double>> &decon,
                   const size_t padding /* = 0 */) override
    {}; // Dummy

    void transform(
            datamodel::DeconQueue &decon_queue,
            const size_t decon_start_ix = 0,
            const size_t test_offset = 0,
            const size_t custom_residuals_ct = std::numeric_limits<size_t>::max());

    void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon,
                           const size_t padding /* = 0 */) const override;

    static size_t get_residuals_length(const double _stretch_coef = OEMD_STRETCH_COEF, const size_t siftings = DEFAULT_SIFTINGS);

    static size_t get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef = OEMD_STRETCH_COEF);

    size_t get_residuals_length(const std::string &queue_name);

    static void
    expand_the_mask(const size_t mask_size, const size_t input_size, const double *dev_mask, double *dev_expanded_mask);
};


__global__ void vec_power(
        const cufftDoubleComplex *__restrict__ x,
        cufftDoubleComplex *__restrict__ y,
        const size_t x_size,
        const size_t siftings);

__global__ void
gpu_multiply_complex(
        const size_t input_size,
        const cufftDoubleComplex *__restrict__ multiplier,
        cufftDoubleComplex *__restrict__ output);

__global__ void
vec_subtract_inplace(
        double *__restrict__ x,
        const double *__restrict__ y,
        const size_t x_size);

}
}
