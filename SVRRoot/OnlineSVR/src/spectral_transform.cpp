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


std::unique_ptr<svr::spectral_transform> svr::spectral_transform::create(const std::string& transformation_name, const size_t levels, const double stretch_coef, const bool force_find_oemd_coefs)
{
    LOG4_DEBUG("Request to create a \""<<transformation_name<<"\" transformer.");

    if(transformation_name.find("fk") != std::string::npos)
    {
        auto fkfo = svr::spectral_transform::modwt_filter_order_from(transformation_name);
        return std::unique_ptr<svr::spectral_transform>(new svr::modwt_transform(fkfo, levels, false));
    }

    if (transformation_name == "stft")
        return std::unique_ptr<svr::spectral_transform>(new svr::short_term_fourier_transform(levels));
    if (transformation_name == "stft_cpu")
        return std::unique_ptr<svr::spectral_transform>(new svr::short_term_fourier_transform_cpu(levels));
#ifdef VIENNACL_WITH_OPENCL
    if (transformation_name == "stft_ocl")
        return std::unique_ptr<svr::spectral_transform>(new svr::short_term_fourier_transform_opencl(levels));
#endif
    if (transformation_name == "oemd")
        return std::unique_ptr<svr::spectral_transform>(new svr::online_emd(levels, stretch_coef, force_find_oemd_coefs));
    if (transformation_name == "cvmd")
        return std::unique_ptr<svr::spectral_transform>(new svr::fast_cvmd(levels));


    return std::unique_ptr<svr::spectral_transform>(new svr::sw_transform(transformation_name, levels));
}

size_t svr::spectral_transform::get_min_frame_length(
        const size_t transformation_levels,
        const size_t max_lag_count,
        const size_t filter_order,
        const std::string &transformation_name)
{
    size_t min_frame_length = 0;
    if (transformation_name == "stft")
        min_frame_length = transformation_levels + 1;
    else if (transformation_name == "oemd")
        min_frame_length = svr::online_emd::get_frame_size(transformation_levels, max_lag_count);
    else if (transformation_name == "cvmd")
        min_frame_length = svr::online_emd::get_frame_size(transformation_levels, max_lag_count);
    else
        min_frame_length = svr::spectral_transform::modwt_levels_to_frame_length(transformation_levels, filter_order);

    return min_frame_length;
}

svr::spectral_transform::spectral_transform(const std::string &transformation_name, const size_t& levels)
: transformation_name_(std::move(transformation_name)), levels_(levels)
{
}

svr::spectral_transform::~spectral_transform()
{

}
void svr::spectral_transform::summary() const
{

}

size_t svr::spectral_transform::modwt_levels_to_frame_length(const size_t modwt_levels, const size_t wavelet_order)
{
    return size_t(wavelet_order * std::pow(2, modwt_levels + 1));
}

size_t svr::spectral_transform::modwt_residuals_length(const size_t modwt_levels)
{
    return size_t(std::pow(2, modwt_levels) * 1.5);
}

//returns existing filter order for MODWT, otherwise return 0
size_t svr::spectral_transform::modwt_filter_order_from(const std::string &filter_order)
{
    const char *c;
    for (c = filter_order.c_str(); *c == '\0'; ++c) if (*c >= '0' && *c <= '9') break;
    return *c == '\0' ? 0 : (unsigned) atol(c);
}
