/*
 * File:   short_term_fourier_transform.hpp
 * Author: Boyko Perfanov
 * Implementation of a maximum overlap fourier transform based on libfftw3.
 * Properties of the transform:
 * Stationary for stationary signals - adjacent deconstructed spectra are stationary.
 * Implanted signals from another signal or timeframe (i.e. discontinuous derivatives of any order)
 *      will produce reconstruction artifacts. This is important in terms of stationarity testing.
 * The number of levels must be a power of 2, recommended > 8.
 *
 * Created on April 6, 2017, 6:22 PM
 */

#pragma once

#include "spectral_transform.hpp"
#include <mutex>

namespace svr {

class short_term_fourier_transform_cpu final: public spectral_transform
{
public:
    explicit short_term_fourier_transform_cpu(const size_t levels);

    virtual ~short_term_fourier_transform_cpu() final
    {}

    // TODO: refactor the base class's transform method - padding is not used in stft.
    virtual void transform(const std::vector<double> &input, std::vector<std::vector<double> > &decon,
                           const size_t padding /* = 0 */) override;

    virtual void inverse_transform(const std::vector<double> &decon, std::vector<double> &recon,
                                   const size_t padding /* = 0 */) const override;


    // Singleton that provides thread-safety when creating and destroying libfftw3 plans.
    static std::mutex &get_mutex_for_fftw3()
    {
        static std::mutex fftw_mutex;
        return fftw_mutex;
    }

private:
    const size_t frame_size;
};

#ifdef ENABLE_OPENCL
class short_term_fourier_transform_opencl : public spectral_transform
{

public:
    explicit short_term_fourier_transform_opencl(const size_t levels);

    virtual void
    transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const size_t padding = 0) override;

    virtual void
    inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding = 0) const override;


    static size_t constexpr can_handle_levels = 15;
private:
    const size_t gpu_parallelism;
    const size_t frame_size;

};

#endif

class short_term_fourier_transform : public spectral_transform
{
public:
    explicit short_term_fourier_transform(const size_t levels);

    virtual void
    transform(
            const std::vector<double> &input,
            std::vector<std::vector<double> > &decon,
            const size_t padding = 0) override;

    virtual void
    inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding = 0) const override;

    static size_t get_frame_size(const size_t transformation_levels)
    {
        return transformation_levels + 1;
    }

    static size_t get_residuals_length(const size_t transformation_levels)
    {
        return transformation_levels / 2;
    }

private:
    static auto const gpu_faster_threshold = (size_t) 3e+5;

    short_term_fourier_transform_cpu cpu_transformer;
#ifdef ENABLE_OPENCL
    short_term_fourier_transform_opencl gpu_transformer;
#endif
};


}
