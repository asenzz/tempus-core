#ifdef VIENNACL_WITH_OPENCL
#include "short_term_fourier_transform_ocl_impl.hpp"


namespace svr {

short_term_fourier_transform_ocl_impl::short_term_fourier_transform_ocl_impl(size_t levels)
: frame_size(levels+1)
, gpu_frame_size(frame_size * 2)
, gpu_frame_buffer_size(gpu_frame_size * sizeof(double))
, gpu_kernel("mostft")
, ctx(gpu_kernel.ctx())
, context(ctx.handle())
, queue(ctx.get_queue().handle())
, const_buffer(context, CL_MEM_READ_ONLY, 32 * sizeof (int32_t))
{
    size_t const const_buffer_size = 32;
    std::array<int32_t, const_buffer_size> const constant = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    cl_int err = queue.enqueueWriteBuffer(const_buffer, CL_TRUE, 0, const_buffer_size * sizeof(int), constant.data());
    CL_CHECK(err);
}


void short_term_fourier_transform_ocl_impl::transform(std::vector<double>::const_iterator input_begin, std::vector<std::vector<double>>::iterator output_begin, size_t input_size) const
{
    if(input_size < frame_size)
        throw std::runtime_error("Input does not provide enough data for the transform.");

    size_t const  input_buffer_size =  input_size * sizeof(double)
                , out_frame_number = input_size - frame_size + 1
                , output_buffer_size = out_frame_number * gpu_frame_buffer_size;

    cl_int err = CL_SUCCESS;
 
    cl::Buffer out(context, CL_MEM_READ_WRITE, output_buffer_size, NULL, &err); CL_CHECK(err);
    cl::Buffer in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_buffer_size, const_cast<double*>(&*input_begin), &err); CL_CHECK(err);

    for(size_t frame = 0; frame < out_frame_number; ++frame)
    {
        svr::cl12::ocl_kernel forward(ctx.get_kernel("mostft", "fft_fwd").handle());
        err = forward.set_args(const_buffer, in, out, frame, frame * gpu_frame_size / 2)
               .enqueue(queue, 0, 64, 64);
        CL_CHECK(err);
    }

    queue.finish();

    std::vector<double> buff(out_frame_number * gpu_frame_size);
    err = queue.enqueueReadBuffer(out, CL_TRUE, 0, buff.size() * sizeof(double), buff.data());
    CL_CHECK(err);

    for(size_t frameno = 0; frameno < out_frame_number; ++frameno)
    {
        std::vector<double> & frame = *output_begin++;
        frame.resize(frame_size);
        auto const frame_start = buff.begin() + frameno * gpu_frame_size
                 , frame_end = frame_start + frame_size;
        std::copy(frame_start, frame_end, frame.begin());
        frame[1] = *frame_end;
    }
}


void short_term_fourier_transform_ocl_impl::inverse_transform(std::vector<double>::const_iterator decon_begin, std::vector<double>::iterator recon, size_t decon_size) const
{
    if(decon_size < frame_size)
        throw std::runtime_error("Input does not provide enough data for the transform.");

    size_t const  input_buffer_size =  decon_size * sizeof(double)
                , signal_length = decon_size / gpu_frame_size
                , output_size = signal_length + frame_size - 1
                , output_buffer_size = output_size * sizeof(double)
                ;

    cl_int err = CL_SUCCESS;

    cl::Buffer in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_buffer_size, const_cast<double*>(&*decon_begin), &err); CL_CHECK(err);

    std::vector<double> zeros(output_size, 0.0);
    cl::Buffer out(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, output_buffer_size, const_cast<double*>(zeros.data()), &err); CL_CHECK(err);


    for(size_t frame = 0; frame < signal_length; ++frame)
    {
        svr::cl12::ocl_kernel forward(ctx.get_kernel("mostft", "fft_back").handle());
            err = forward.set_args(const_buffer, in, out, frame * gpu_frame_size / 2, frame)
               .enqueue(queue, 0, 64, 64);
        CL_CHECK(err);
    }

    queue.finish();

    err = queue.enqueueReadBuffer(out, CL_TRUE, 0, output_buffer_size, &*recon);
    CL_CHECK(err);

    size_t const div = std::min(frame_size, signal_length);
    for(size_t i = 0; i < div; ++i)
        *recon++ /= (i+1);

    for(size_t i = div; i < output_size - div; ++i)
        *recon++ /= div;

    for(size_t i = output_size - div; i < output_size; ++i)
        *recon++ /= (output_size - i);
}

}

#endif