#pragma once

#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "common/defines.h"
#include <unordered_map>

#include <common/Logging.hpp>

#define OEMD_TAIL_COMPUTE
#define OEMD_CUDA


namespace svr {

const double oemd_epsilon = 0.0000011;
const double oemd_mask_sum = 1.00000;

using oemd_fptr =  void (*)(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);

class online_emd final : public spectral_transform
{
    bool fir_coefs_initialized = false;

public:
    bool get_fir_coefs_initialized() const { return fir_coefs_initialized; }
    void set_fir_coefs_initialized(const bool new_state) { fir_coefs_initialized = new_state; }

    void find_fir_coefficients(const std::vector<double> &input);

    explicit online_emd(const size_t levels, const double stretch_coef = 1, const bool force_find_fir_coefs = false);

    virtual ~online_emd() final
    {}

    virtual void transform(const std::vector<double> &input, std::vector<std::vector<double> > &decon,
                           const size_t padding /* = 0 */) override;

    virtual void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon,
                                   const size_t padding /* = 0 */) const override;

    static size_t get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef = OEMD_STRETCH_COEF);
    static size_t get_residuals_length(const size_t levels);
    size_t get_residuals_length();

    static size_t get_frame_size(const size_t transformation_levels, const size_t max_lag_count = 1)
    {
        return 1;
    }

    //oemd_fptr select_method(size_t level, size_t input_length) const;
private:
    const size_t levels;
    const double stretch_coef;
    std::shared_ptr<oemd_coefficients> p_oemd_coef;
};

}
