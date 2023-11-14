#ifndef SHORT_TERM_FOURIER_TRANSFORM_OCL_IMPL_HPP
#define SHORT_TERM_FOURIER_TRANSFORM_OCL_IMPL_HPP


#include <common/Logging.hpp>
#include <common/gpu_handler.hpp>


namespace svr {

class short_term_fourier_transform_ocl_impl
{
#ifdef VIENNACL_WITH_OPENCL
public:
    short_term_fourier_transform_ocl_impl(size_t levels);

    void transform(std::vector<double>::const_iterator input_begin, std::vector<std::vector<double>>::iterator output_begin, size_t size) const;
    void inverse_transform(std::vector<double>::const_iterator decon_begin, std::vector<double>::iterator recon, size_t size) const;

private:
    const size_t frame_size;
    const size_t gpu_frame_size;
    const size_t gpu_frame_buffer_size;
    const size_t gpu_effective_frame_size;

    common::gpu_kernel gpu_kernel;
    mutable viennacl::ocl::context ctx;
    mutable cl12::ocl_context context;
    mutable cl12::command_queue queue;
    cl::Buffer const_buffer;
#else
public:
        short_term_fourier_transform_ocl_impl(size_t levels){
            abort();
        }

        void transform(std::vector<double>::const_iterator input_begin, std::vector<std::vector<double>>::iterator output_begin, size_t size) const{
            abort();
        }
        void inverse_transform(std::vector<double>::const_iterator decon_begin, std::vector<double>::iterator recon, size_t size) const{
            abort();
        }
#endif
};


}


#endif /* SHORT_TERM_FOURIER_TRANSFORM_OCL_IMPL_HPP */

