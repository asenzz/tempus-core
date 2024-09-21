#include <cuda_runtime_api.h>
#include "common/semaphore.hpp"
#include "common/gpu_handler.hpp"
#include <mutex>

namespace svr {
namespace common {

// GPU Context, set the worker as blocked during its lifetime.
__attribute_noinline__ gpu_context::gpu_context() :
        context_id_(gpu_handler_hid::get().get_free_gpu()), dev_ct(gpu_handler_hid::get().get_gpu_devices_count())
{
    LOG4_TRACE("Enter ctx " << context_id_);
}

__attribute_noinline__ gpu_context::gpu_context(const gpu_context &context) :
        context_id_(context.context_id_), dev_ct(gpu_handler_hid::get().get_gpu_devices_count())
{};

unsigned gpu_context::id() const
{
    return context_id_;
}

unsigned gpu_context::phy_id() const
{
    const unsigned gpu_id = context_id_ % dev_ct;
    LOG4_TRACE("Returning GPU with ID " << gpu_id);
    return gpu_id;
}

viennacl::ocl::context &gpu_context::ctx() const
{
    return viennacl::ocl::get_context(context_id_);
}

gpu_context::~gpu_context()
{
#ifdef GPU_QUEUE
    gpu_handler_hid::get().return_gpu(context_id_);
#else
    gpu_handler_hid::get().return_gpu();
#endif
    LOG4_TRACE("Exit ctx " << context_id_);
}

namespace {
std::unordered_map<std::string, std::stringstream> kernel_file_cache;
tbb::mutex kernel_file_cache_mx;
}

void
gpu_kernel::add_ctx_kernel(
        viennacl::ocl::context &ctx,
        const std::string &kernel_name,
        const std::stringstream &kernel_program)
{
//    ctx.build_options(OCL_BUILD_OPTIONS);
    ctx.add_program(kernel_program.str(), kernel_name);
    LOG4_DEBUG("Kernel " << kernel_name << " loaded successfully.");
}


gpu_kernel::gpu_kernel(const std::string &kernel_name) : kernel_name_(kernel_name)
{
    const tbb::mutex::scoped_lock lk(kernel_file_cache_mx);
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(context_id_);
    const auto iter_kernels_cache = kernel_file_cache.find(kernel_name);

    if (iter_kernels_cache != kernel_file_cache.end()) {
        add_ctx_kernel(ctx, kernel_name, iter_kernels_cache->second);
        return;
    }

    std::string kernels_path("../SVRRoot/opencl-libsvm/libsvm/kernels/" + kernel_name + ".cl");
    std::ifstream file_kernels(kernels_path);
    std::stringstream ss_kernels;
    ss_kernels << file_kernels.rdbuf();
    add_ctx_kernel(ctx, kernel_name, ss_kernels);
}


gpu_kernel::~gpu_kernel()
{
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(context_id_);
    ctx.delete_program(kernel_name_);
}


void gpu_helper::enqueue(const viennacl::ocl::kernel &kernel)
{
    enqueue(kernel, 32, 64);
}


void gpu_helper::enqueue(const viennacl::ocl::kernel &kernel, const size_t local_work_size, const size_t global_work_group_size)
{
    const size_t global_work_size = global_work_group_size * local_work_size;
    const auto err = clEnqueueNDRangeKernel(
            kernel.context().get_queue().handle().get(), kernel.handle().get(), 1, NULL,
            &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) LOG4_THROW("clEnqueueNDRangeKernel failed, " << get_error_string(err));
}

} //namespace common


namespace cl12 {

cl_int kernel_helper::enqueue(
        const command_queue &queue,
        const ::cl::NDRange &offset,
        const ::cl::NDRange &global,
        const ::cl::NDRange &local,
        const ::cl::vector<::cl::Event> *events,
        ::cl::Event *event)
{
    const cl_int code = queue.enqueueNDRangeKernel(*this, offset, global, local, events, event);
    if (code != CL_SUCCESS)
        LOG4_THROW("OpenCL execution failed. Error: " << svr::common::gpu_helper::get_error_string(code));
    return code;
}

cl::NDRange ndrange(const range_args2_t &range_args)
{
    return {range_args[0], range_args[1]};
}

}

} //namespace svr
