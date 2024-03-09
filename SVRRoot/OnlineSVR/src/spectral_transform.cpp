#include "spectral_transform.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "short_term_fourier_transform.hpp"
#include "online_emd.hpp"
#include "fast_cvmd.hpp"
#include "sw_transform.hpp"
#include "../include/modwt_transform.hpp"

#include <util/math_utils.hpp>

#include "common/Logging.hpp"

namespace svr {

std::unique_ptr<spectral_transform>
spectral_transform::create(const std::string &transformation_name, const size_t levels, const double stretch_coef, const bool force_find_oemd_coefs)
{
    LOG4_DEBUG("Request to create a \"" << transformation_name << "\" transformer.");

    if (transformation_name.find("fk") != std::string::npos) {
        auto fkfo = spectral_transform::modwt_filter_order_from(transformation_name);
        return std::unique_ptr<spectral_transform>(new modwt_transform(fkfo, levels, false));
    }

    if (transformation_name == "stft")
        return std::unique_ptr<spectral_transform>(new short_term_fourier_transform(levels));
    if (transformation_name == "stft_cpu")
        return std::unique_ptr<spectral_transform>(new short_term_fourier_transform_cpu(levels));
#ifdef VIENNACL_WITH_OPENCL
    if (transformation_name == "stft_ocl")
        return std::unique_ptr<spectral_transform>(new short_term_fourier_transform_opencl(levels));
#endif
    if (transformation_name == "oemd")
        return std::unique_ptr<spectral_transform>(new oemd::online_emd(levels, stretch_coef));
    if (transformation_name == "cvmd")
        return std::unique_ptr<spectral_transform>(new vmd::fast_cvmd(levels));


    return std::unique_ptr<spectral_transform>(new sw_transform(transformation_name, levels));
}

size_t spectral_transform::get_min_frame_length(
        const size_t transformation_levels,
        const size_t max_lag_count,
        const size_t filter_order,
        const std::string &transformation_name)
{
    size_t min_frame_length = 0;
    if (transformation_name == "stft")
        min_frame_length = transformation_levels + 1;
    else if (transformation_name == "oemd")
        min_frame_length = 1;
    else if (transformation_name == "cvmd")
        min_frame_length = 1;
    else
        min_frame_length = spectral_transform::modwt_levels_to_frame_length(transformation_levels, filter_order);

    return min_frame_length;
}

spectral_transform::spectral_transform(const std::string &transformation_name, const size_t &levels)
        : transformation_name_(std::move(transformation_name)), levels_(levels)
{
}

spectral_transform::~spectral_transform()
{

}

void spectral_transform::summary() const
{

}

size_t spectral_transform::modwt_levels_to_frame_length(const size_t modwt_levels, const size_t wavelet_order)
{
    return size_t(wavelet_order * std::pow(2, modwt_levels + 1));
}

size_t spectral_transform::modwt_residuals_length(const size_t modwt_levels)
{
    return size_t(std::pow(2, modwt_levels) * 1.5);
}

//returns existing filter order for MODWT, otherwise return 0
size_t spectral_transform::modwt_filter_order_from(const std::string &filter_order)
{
    const char *c;
    for (c = filter_order.c_str(); *c == '\0'; ++c) if (*c >= '0' && *c <= '9') break;
    return *c == '\0' ? 0 : (unsigned) atol(c);
}


void spectral_transform::mirror_tail(const datamodel::datarow_range &input, const size_t needed_data_ct, std::vector<double> &tail)
{
    const size_t in_colix = input.begin()->get()->get_values().size() / 2;
    const auto input_size = input.distance();
    LOG4_WARN("Adding mirrored tail of size " << needed_data_ct - input_size << ", to input of size " <<
                                              input_size << ", total size " << needed_data_ct);
    const auto empty_ct = needed_data_ct - input_size;
    tail.resize(empty_ct);
#pragma omp parallel for num_threads(adj_threads(empty_ct))
    for (size_t i = 0; i < empty_ct; ++i) {
        const auto phi = double(i) / double(input_size);
        tail[empty_ct - 1 - i] = input[(size_t) std::round((input_size - 1) * std::abs(std::round(phi) - phi))]->get_value(in_colix);
    }
}

}