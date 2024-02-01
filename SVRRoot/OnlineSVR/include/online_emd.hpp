#pragma once

#include "spectral_transform.hpp"
#include "oemd_coefficients.hpp"
#include "common/defines.h"
#include "model/DeconQueue.hpp"
#include "oemd_coefficients_search.hpp"


#define OEMD_TAIL_COMPUTE
#define OEMD_CUDA


namespace svr {

constexpr double oemd_epsilon = 0.0000011;
constexpr double oemd_mask_sum = 1.00000;

using oemd_fptr =  void (*)(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);

class online_emd final : public spectral_transform
{
    const size_t levels;
    const double stretch_coef;
    static t_coefs_cache oemd_coefs_cache;

    t_oemd_coefficients_ptr get_coefs(
            const datamodel::datarow_range &input, const std::vector<double> &tail, const std::string &queue_name) const;

public:
    explicit online_emd(const size_t levels, const double stretch_coef = OEMD_STRETCH_COEF);

    void transform(const std::vector<double> &input, std::vector<std::vector<double>> &decon,
                           const size_t padding /* = 0 */) override {}; // Dummy

    void transform(
            datamodel::DeconQueue_ptr &p_decon_queue, const size_t test_offset = 0, const size_t custom_residuals_ct = std::numeric_limits<size_t>::max(), const bpt::ptime &oemd_start_time = bpt::min_date_time);

    void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon,
                                   const size_t padding /* = 0 */) const override;

    static size_t get_residuals_length(const double _stretch_coef = OEMD_STRETCH_COEF, const size_t siftings = DEFAULT_SIFTINGS);
    static size_t get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef = OEMD_STRETCH_COEF);
    size_t get_residuals_length(const std::string &queue_name);
};

}
