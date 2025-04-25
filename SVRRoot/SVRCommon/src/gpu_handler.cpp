#include <cuda_runtime_api.h>
#include <mutex>
#include "common/semaphore.hpp"
#include "common/gpu_handler.hpp"

namespace svr {

#ifdef ENABLE_OPENCL

namespace common {

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
    ctx.build_options(OCL_BUILD_OPTIONS);
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

    const std::string kernels_path("../SVRRoot/opencl-libsvm/libsvm/kernels/" + kernel_name + ".cl");
    std::ifstream file_kernels(kernels_path);
    std::stringstream ss_kernels;
    ss_kernels << file_kernels.rdbuf();
    add_ctx_kernel(ctx, kernel_name, ss_kernels);
}


void gpu_kernel::ensure_compiled_kernel(viennacl::ocl::context &ctx, const std::string &kernel_name)
{
    LOG4_BEGIN();

    if (ctx.has_program(kernel_name)) return;

    const std::string kernel_path(KERNEL_DIRECTORY_PATH + kernel_name + ".cl");
    LOG4_DEBUG("Looking for the source of " << kernel_name << " in " << KERNEL_DIRECTORY_PATH);
    std::ifstream file_kernel(kernel_path);
    std::stringstream ss_kernel;
    ss_kernel << file_kernel.rdbuf();
//        ctx.build_options(OCL_BUILD_OPTIONS);
    ctx.add_program(ss_kernel.str(), kernel_name);

    LOG4_END();
}


void gpu_kernel::ensure_compiled_kernel(viennacl::ocl::context &ctx, const std::string &kernel_name, const std::string &kernel_file_name)
{
    LOG4_BEGIN();

    if (ctx.has_program(kernel_name)) return;

    const std::string kernel_file_path(KERNEL_DIRECTORY_PATH + kernel_file_name + ".cl");

    LOG4_DEBUG("Looking for the source of " << kernel_name << " at " << kernel_file_path);
    std::ifstream file_kernel(kernel_file_path);
    std::stringstream ss_kernel;
    ss_kernel << file_kernel.rdbuf();
    ctx.build_options(OCL_BUILD_OPTIONS);
    ctx.add_program(ss_kernel.str(), kernel_file_name);

    LOG4_END();
}


gpu_kernel::~gpu_kernel()
{
    viennacl::ocl::get_context(context_id_).delete_program(kernel_name_);
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

#endif

} //namespace svr


