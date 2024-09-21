#pragma once

#ifdef VIENNACL_WITH_OPENCL

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

#endif // VIENNACL_WITH_OPENCL

#include <stack>
#include <mutex>
#include <thread>
#include <boost/thread/shared_mutex.hpp>
#include "logging.hpp"
#include "parallelism.hpp"
#include "semaphore.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#include <CL/opencl.hpp>

#pragma GCC diagnostic pop

// #define IPC_SEMAPHORE // GPU counter semaphore is IPC
#define GPU_QUEUE
// OpenCL kernel options
#define KERNEL_DIRECTORY_PATH   "../SVRRoot/opencl-libsvm/libsvm/kernels/"
#define COMMON_PATH             "../SVRRoot/SVRCommon/include"
#define OCL_BUILD_OPTIONS       " -I\"" KERNEL_DIRECTORY_PATH "\" -I\"" COMMON_PATH "\""
#define CTX_PER_GPU 4

#ifdef GPU_QUEUE

#include <oneapi/tbb/concurrent_queue.h>

#endif

#ifdef IPC_SEMAPHORE
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#define SVRWAVE_GPU_SEM         "svrwave_gpu_sem"
#endif

namespace svr {
namespace common {

class gpu_context;

class fat_gpu_context;

template<const unsigned context_per_gpu>
class gpu_handler : boost::noncopyable
{
    friend gpu_context;
    friend fat_gpu_context;

    unsigned get_free_gpu();

    unsigned get_free_gpus(const unsigned gpu_ct);

#ifdef GPU_QUEUE

    void return_gpu(const unsigned id);

#else
    void return_gpu();
#endif

    void return_gpus(const unsigned gpu_ct);

    void sort_free_gpus();

    gpu_handler(const gpu_handler &) = delete;

    gpu_handler &operator=(const gpu_handler &) = delete;

    void init_devices(const int device_type);

#ifndef GPU_QUEUE
#ifdef IPC_SEMAPHORE
    std::unique_ptr<boost::interprocess::named_semaphore> p_gpu_sem_;
#else
    std::unique_ptr<svr::fast_semaphore> p_gpu_sem_;
#endif
#endif
    unsigned max_running_gpu_threads_number_;
    unsigned m_max_gpu_kernels_;
    unsigned max_gpu_data_chunk_size_;
#ifdef VIENNACL_WITH_OPENCL
    std::deque<viennacl::ocl::device> devices_;
#endif //VIENNACL_WITH_OPENCL

#ifdef GPU_QUEUE
    tbb::concurrent_bounded_queue<unsigned> available_devices_; // TODO Compare to next_device atomic counter for even distribution of workload and resource usage
#else
    std::atomic<unsigned> next_device_;
#endif
    mutable boost::shared_mutex devices_mutex_;

public:
    unsigned get_max_gpu_threads() const;

    unsigned get_gpu_devices_count() const;

    unsigned get_max_gpu_kernels() const;

    size_t get_max_gpu_data_chunk_size() const;

#ifdef VIENNACL_WITH_OPENCL

    const viennacl::ocl::device &device(const unsigned idx) const;

#endif //VIENNACL_WITH_OPENCL

    static gpu_handler &get();

    gpu_handler();

    ~gpu_handler();
};

using gpu_handler_hid = gpu_handler<CTX_PER_GPU>;

// TODO  Make CUDA stream context
#ifdef VIENNACL_WITH_OPENCL

class gpu_context
{
protected:
    const unsigned context_id_, dev_ct;
public:
    __attribute_noinline__ gpu_context();

    __attribute_noinline__ gpu_context(const gpu_context &context);

    virtual ~gpu_context();

    unsigned id() const;

    unsigned phy_id() const;

    viennacl::ocl::context &ctx() const;
};

class gpu_kernel : public gpu_context
{
public:
    explicit gpu_kernel(const std::string &kernel_name);

    virtual ~gpu_kernel();

    static void
    ensure_compiled_kernel(
            viennacl::ocl::context &ctx, const std::string &kernel_name)
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


    static void
    ensure_compiled_kernel(
            viennacl::ocl::context &ctx, const std::string &kernel_name, const std::string &kernel_file_name)
    {
        LOG4_BEGIN();

        if (ctx.has_program(kernel_name)) return;

        const std::string kernel_file_path(KERNEL_DIRECTORY_PATH + kernel_file_name + ".cl");

        LOG4_DEBUG("Looking for the source of " << kernel_name << " at " << kernel_file_path);
        std::ifstream file_kernel(kernel_file_path);
        std::stringstream ss_kernel;
        ss_kernel << file_kernel.rdbuf();
//        ctx.build_options(OCL_BUILD_OPTIONS);
        ctx.add_program(ss_kernel.str(), kernel_file_name);

        LOG4_END();
    }

private:
    std::string kernel_name_;

    void add_ctx_kernel(viennacl::ocl::context &ctx, const std::string &kernel_name, const std::stringstream &kernel_program);
};

class gpu_helper
{
public:
    static std::string get_error_string(const cl_int error);

    static void enqueue(const viennacl::ocl::kernel &kernel);

    static void enqueue(const viennacl::ocl::kernel &kernel, const size_t local_work_size, const size_t global_work_group_size);
};

} //namespace common

namespace cl12 {
template<typename cl_class>
struct naf : public cl_class// Null at finish
{
    template<class handler_type>
    naf(handler_type const &handler) : cl_class(handler)
    {}

    ~naf()
    {
        cl_class::operator()() = NULL;
    }
};

using ocl_context = naf<::cl::Context>;
using command_queue = naf<::cl::CommandQueue>;

class kernel_helper : public ::cl::Kernel
{
public:
    template<typename handler_type> kernel_helper(handler_type const &handler);

    template<class... Args> kernel_helper &set_args(Args... args);

    template<class T> kernel_helper &set_arg(const size_t position, T t);

    cl_int enqueue(const command_queue &queue, const ::cl::NDRange &offset, const ::cl::NDRange &global, const ::cl::NDRange &local = ::cl::NullRange,
                   const ::cl::vector<::cl::Event> *events = NULL, ::cl::Event *event = NULL);

private:
    template<class T, class... Args>
    void set_args(const size_t args_sz, T t, Args... args);

    template<class T> void set_args(const size_t args_sz, T t);
};

using ocl_kernel = naf<kernel_helper>;

typedef size_t range_args2_t[2];

cl::NDRange ndrange(const range_args2_t &range_args);

#define CL_CHECK(cl_call) { cl_int code = (cl_call); if (code != CL_SUCCESS /* == clblasSuccess */ )                     \
            LOG4_THROW("OpenCL call failed with error " << svr::common::gpu_helper::get_error_string(code)); }

#endif //VIENNACL_WITH_OPENCL
}  //namespace cl12
}

#include "gpu_handler.tpp"