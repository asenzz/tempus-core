#include <fftw3.h>
#include "short_term_fourier_transform.hpp"
#include "common/thread_pool.hpp"
#include "short_term_fourier_transform_ocl_impl.hpp"

namespace svr
{
short_term_fourier_transform_cpu::short_term_fourier_transform_cpu(const size_t levels): spectral_transform(std::string("stft_cpu"), levels), frame_size(levels + 1)
{
    //frame_size(levels+1): this is because wavelet transform transforms into (Levels + 1) models.
    // It's a "feature". Otherwise, go through all the project code to use (Levels) models, and give (Levels+1) number to wavelet_transform.
    auto l = levels + 1;
    while (l > 1) {
        if (l % 2) THROW_EX_FS(std::logic_error, "Invalid number of levels " << l);
        l /= 2;
    }
}

void
short_term_fourier_transform_cpu::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double> > &decon,
        const size_t padding = 0)
{
    LOG4_BEGIN();
    const auto signal_length = static_cast<size_t>(input.size());
    if (signal_length < 2) throw std::runtime_error("Transform of invalid number of elements.");
    auto in = (double *) fftw_malloc(sizeof(double) * (frame_size));
    auto out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (frame_size / 2 + 1));
    decon.resize(input.size() - frame_size + 1);
    auto &m = get_mutex_for_fftw3();
    for (size_t frame_slide_offset = 0; frame_slide_offset < signal_length - frame_size + 1; ++frame_slide_offset) {
        // Prepare input frame
        for (size_t i = 0; i < frame_size; ++i) in[i] = input[i + frame_slide_offset];

        // Run the transform
        // TODO: try with different plan flags
        //fftw_plan plan = fftw_plan_dft_1d(frame_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        fftw_plan plan;
        {
            auto fft_lock = std::unique_lock<std::mutex>(m);
            plan = fftw_plan_dft_r2c_1d(frame_size, in, out, FFTW_ESTIMATE);//FFTW_ESTIMATE);
        }

        fftw_execute(plan);

        {
            auto fft_lock = std::unique_lock<std::mutex>(m);
            fftw_destroy_plan(plan);
        }


        // Normalize the output values corresponding to scipy.fft output. In the future, we can get rid of that normalization - this is for
        // debugging and testing purposes.
        // Currently, the output signal does not correspond to the scipy.fft output - it does only at rought scale but it looks like different "noise".
        // It might be that we need to choose a specific parameter to get the same results.

        // The 0th phase coefficient is always zero, and the last phase coefficient is also always zero.
        // Pack the values by moving the last phase coeff to the 0th position, so we have frame_size values - to save into database.
        // Note that we need to unpack the values appropriately for reconstruction.
        std::vector<double> frame(frame_size);
        for (size_t i = 1; i < frame_size / 2; ++i) {
            frame[2 * i] = out[i][0];
            frame[2 * i + 1] = out[i][1];
        }
        frame[0] = out[0][0];
        frame[1] = out[frame_size / 2][0];

        decon[frame_slide_offset] = frame;

    }
    fftw_free(in);
    fftw_free(out);

    LOG4_DEBUG("Deconstruction yielded " << decon.size() << "x" << decon[0].size() << " matrix.");
    LOG4_END();
}

void short_term_fourier_transform_cpu::inverse_transform(
        const std::vector<double> &decon, std::vector<double> &recon,
        const size_t padding = 0) const
{
    LOG4_BEGIN();
    const auto signal_length = static_cast<size_t>(decon.size()) / frame_size;
    if (signal_length < 2) throw std::runtime_error("Transform of invalid number of elements.");

    recon.resize(signal_length + frame_size - 1);
    for (auto &r: recon) r = 0;
    auto &m = get_mutex_for_fftw3();
    auto in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (frame_size / 2 + 1));
    auto out = (double *) fftw_malloc(sizeof(double) * frame_size);

    for (size_t frame_slide_offset = 0; frame_slide_offset < signal_length; ++frame_slide_offset) {
        // Prepare input frame
        std::vector<double> decon_frame(frame_size);
        for (size_t level = 0; level < frame_size; ++level) {
            decon_frame[level] = decon[level * signal_length + frame_slide_offset];
        }

        // Prepare the fftw input frame
        for (size_t i = 1; i < frame_size / 2; ++i) {
            in[i][0] = decon_frame[2 * i + 0];
            in[i][1] = decon_frame[2 * i + 1];
        }
        in[0][0] = decon_frame[0];
        in[0][1] = 0;
        in[(frame_size / 2)][0] = decon_frame[1];
        in[(frame_size / 2)][1] = 0;

        // Run the transform
        fftw_plan plan;
        {
            auto fft_lock = std::unique_lock<std::mutex>(m);
            plan = fftw_plan_dft_c2r_1d(frame_size, in, out, FFTW_ESTIMATE);
        }

        fftw_execute(plan);
        {
            auto fft_lock = std::unique_lock<std::mutex>(m);
            fftw_destroy_plan(plan);
        }

        // Add the reconstructed values for each relevant index
        for (size_t value_idx = 0; value_idx < frame_size; ++value_idx)
            recon[value_idx + frame_slide_offset] = out[value_idx];
    }
    fftw_free(in);
    fftw_free(out);

    // Normalize the reconstructed values (e.g. the last value participates in only one fft, so do not multiply.
    // An inner value, in turn, participates in frame_size reconstructions, so we need to divide by that.
/*
    for (size_t i = 0; i < frame_size - 1; ++i) {
        recon[i] *= double(frame_size) / (i + 1);
        recon[signal_length + frame_size - i - 2] *= double(frame_size) / (i + 1);
    }
*/
    //auto sqr_frame_size = frame_size * frame_size;
    for (auto &r: recon) r /= frame_size;

    LOG4_DEBUG("Reconstructed " << decon.size() << " sized flattened matrix to " << recon.size()
                                << " reconstructed vector.");
    LOG4_END();
}



/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
#ifdef VIENNACL_WITH_OPENCL

short_term_fourier_transform_opencl::short_term_fourier_transform_opencl(const size_t levels)
        : spectral_transform(std::string("stft_ocl"), levels),
          gpu_parallelism{common::gpu_handler_hid::get().get_max_gpu_threads()},
          frame_size{levels + 1}
{
    if (levels != 15)
        throw std::logic_error("This code is for 15 levels only");

    if (gpu_parallelism == 0)
        throw std::logic_error("This code requires GPU execution");
}

void short_term_fourier_transform_opencl::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double> > &decon,
        const size_t padding)
{
    const auto input_size = input.size()
    , output_size = input_size - frame_size + 1
    , parallelism = std::min(std::max(output_size / 30000, 1UL), gpu_parallelism) /*30000 is an experimental number*/
    , chunk_size = (output_size < parallelism * 1024) ? output_size : output_size / parallelism
    , chunks = output_size / chunk_size
    , rest_size = input_size - chunk_size * (chunks - 1);

    std::vector<svr::future<void>> futures;

    decon.resize(output_size);

    for (size_t ch = 0; ch < chunks - 1; ++ch) {
        auto input_begin = input.begin() + chunk_size * ch;
        auto output_begin = decon.begin() + chunk_size * ch;

        futures.emplace_back(
                svr::async(
                        [this, input_begin, output_begin, chunk_size]
                                ()
                        {
                            short_term_fourier_transform_ocl_impl impl(frame_size - 1);
                            impl.transform(input_begin, output_begin, chunk_size + frame_size);
                        }
                )
        );
    }

    auto input_begin = input.begin() + chunk_size * (chunks - 1);
    auto output_begin = decon.begin() + chunk_size * (chunks - 1);

    futures.emplace_back(
            svr::async(
                    [this, input_begin, output_begin, rest_size]
                            ()
                    {
                        short_term_fourier_transform_ocl_impl impl(frame_size - 1);
                        impl.transform(input_begin, output_begin, rest_size);
                    }
            )
    );

    for (auto &fut : futures)
        fut.get();
}



namespace {

std::vector<double> linearize(std::vector<double> const &what, size_t const frame_size)
{
    const auto decon_size = what.size() / frame_size
    , gpu_size = frame_size * 2
    , linear_size = gpu_size * decon_size;

    std::vector<double> linearized(linear_size);

    for (size_t row = 0; row < decon_size; ++row) {
        for (size_t level = 0; level < frame_size; ++level)
            linearized[gpu_size * row + level] = what[row + decon_size * level];
        linearized[gpu_size * row + frame_size] = what[row + decon_size * 1];
        linearized[gpu_size * row + 1] = 0;
    }
    return linearized;
}

}


void
short_term_fourier_transform_opencl::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    std::vector<double> linearized = linearize(decon, frame_size);

    const auto input_size = linearized.size()
    , gpu_frame_size = frame_size * 2
    , output_size = input_size / gpu_frame_size + frame_size - 1
    , parallelism = std::min(std::max(output_size / 30000, 1UL), gpu_parallelism) //30000 is an experimental number
    , inp_chunk_size = ((input_size < parallelism * 1024) ? input_size : input_size / parallelism) / gpu_frame_size *
                       gpu_frame_size //Chunk size should be 32 proportional
    , chunks = input_size / inp_chunk_size
    , inp_rest_size = input_size - inp_chunk_size * (chunks - 1)
    , out_chunk_size = (output_size - frame_size + 1) / chunks;

    std::vector<svr::future<void>> futures;

    recon.resize(output_size);

    for (size_t ch = 0; ch < chunks - 1; ++ch) {
        auto input_begin = linearized.begin() + inp_chunk_size * ch;
        auto output_begin = recon.begin() + out_chunk_size * ch;

        futures.emplace_back(
                svr::async(
                        [this, input_begin, output_begin, inp_chunk_size]
                                ()
                        {
                            short_term_fourier_transform_ocl_impl impl(frame_size - 1);
                            impl.inverse_transform(input_begin, output_begin, inp_chunk_size);
                        }
                )
        );
    }

    auto input_begin = linearized.begin() + inp_chunk_size * (chunks - 1);
    auto output_begin = recon.begin() + out_chunk_size * (chunks - 1);

    futures.emplace_back(
            svr::async(
                    [this, input_begin, output_begin, inp_rest_size]
                            ()
                    {
                        short_term_fourier_transform_ocl_impl impl(frame_size - 1);
                        impl.inverse_transform(input_begin, output_begin, inp_rest_size);
                    }
            )
    );

    for (auto &fut : futures)
        fut.get();
}


#endif
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

short_term_fourier_transform::short_term_fourier_transform(const size_t levels)
        : spectral_transform("stft", levels), levels(levels), cpu_transformer(levels)
#ifdef VIENNACL_WITH_OPENCL
, gpu_transformer(levels)
#endif

{}

void
short_term_fourier_transform::transform(
        const std::vector<double> &input, std::vector<std::vector<double>> &decon,
        const size_t padding)
{
#ifdef VIENNACL_WITH_OPENCL
    if (input.size() < gpu_faster_threshold || levels != short_term_fourier_transform_opencl::can_handle_levels)
        cpu_transformer.transform(input, decon, padding);
    else
        gpu_transformer.transform(input, decon, padding);
#else
    cpu_transformer.transform(input, decon, padding);
#endif
}

void short_term_fourier_transform::inverse_transform(const std::vector<double> &decon, std::vector<double> &recon,
                                                     const size_t padding) const
{
#ifdef VIENNACL_WITH_OPENCL
    if (decon.size() / levels < gpu_faster_threshold ||
        levels != short_term_fourier_transform_opencl::can_handle_levels)
        cpu_transformer.inverse_transform(decon, recon, padding);
    else
        gpu_transformer.inverse_transform(decon, recon, padding);
#else
    cpu_transformer.inverse_transform(decon, recon, padding);
#endif
}

}
