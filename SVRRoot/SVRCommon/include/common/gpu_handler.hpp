#pragma once

#ifdef ENABLE_OPENCL

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"

#endif // ENABLE_OPENCL

#include <stack>
#include <mutex>
#include <thread>
#include <boost/thread/shared_mutex.hpp>
#include "logging.hpp"
#include "parallelism.hpp"
#include "semaphore.hpp"

// #define IPC_GPU

#ifdef ENABLE_OPENCL

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#include <CL/opencl.hpp>

#pragma GCC diagnostic pop

// OpenCL kernel options
#define KERNEL_DIRECTORY_PATH   "../SVRRoot/opencl-libsvm/libsvm/kernels/"
#define COMMON_PATH             "../SVRRoot/SVRCommon/include"
#define OCL_BUILD_OPTIONS       " -I\"" KERNEL_DIRECTORY_PATH "\" -I\"" COMMON_PATH "\""

#endif

#ifdef IPC_GPU
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#define SVRWAVE_GPU_SEM         "svrwave_gpu_sem"
#endif

namespace svr {
namespace common {

template<const uint16_t ctx_per_gpu> class gpu_context_;
using gpu_context = gpu_context_<CTX_PER_GPU>;
using gpu_context_4 = gpu_context_<4>;

struct device_info
{
    uint16_t id;
    std::shared_ptr<tbb::mutex> p_mx;
    uint16_t running_threads = 0;
#ifdef ENABLE_OPENCL
    std::shared_ptr<viennacl::ocl::device> p_ocl_info;
#endif
};

template<const uint16_t ctx_per_gpu = 1> class gpu_handler : boost::noncopyable
{
#ifdef IPC_GPU
    std::unique_ptr<boost::interprocess::named_semaphore> p_gpu_sem_;
#endif
    uint16_t max_running_gpu_threads_number_;
#ifdef ENABLE_OPENCL
    uint32_t m_max_gpu_kernels_ = 0;
#endif
    size_t max_gpu_data_chunk_size_;

    std::mutex cv_mx_;

    std::condition_variable cv_;

    std::deque<device_info> available_devices_;

#ifdef ENABLE_OPENCL
void init_devices(const cl_device_type device_type);
#else
    void init_devices();
#endif

public:
    static constexpr auto C_no_gpu_id = std::numeric_limits<uint16_t>::max();

    uint16_t get_free_gpu();

    uint16_t try_free_gpu();

    std::deque<uint16_t> get_free_gpus(const uint16_t gpu_ct);

    void return_gpu(const uint16_t context_id);

    void return_gpus(uint16_t gpu_ct);

    uint16_t context_id(const DTYPE(available_devices_)::const_iterator &it) const;

    static uint16_t dev_id(const uint16_t context_id);

    static uint16_t stream_id(const uint16_t context_id);

    gpu_handler(const gpu_handler &) = delete;

    gpu_handler &operator=(const gpu_handler &) = delete;

    uint16_t get_max_gpu_threads() const;

    uint16_t get_gpu_devices_count() const;

    size_t get_max_gpu_data_chunk_size() const;

#ifdef ENABLE_OPENCL

    uint32_t get_max_gpu_kernels() const;
    const viennacl::ocl::device &device(const uint16_t idx) const;

#endif //ENABLE_OPENCL

    static gpu_handler &get();

    gpu_handler();

    ~gpu_handler();
};

using gpu_handler_1 = gpu_handler<CTX_PER_GPU>;
using gpu_handler_4 = gpu_handler<4>;

template<const uint16_t ctx_per_gpu> class gpu_context_
{
protected:
    const uint16_t context_id_;
public:
    __attribute_noinline__ gpu_context_();

    __attribute_noinline__ gpu_context_(const bool try_init);

    __attribute_noinline__ gpu_context_(const gpu_context &context);

    virtual ~gpu_context_();

    uint16_t id() const;

    uint16_t phy_id() const;

    uint16_t stream_id() const;

#ifdef ENABLE_OPENCL
    viennacl::ocl::context &ctx() const;
#endif

    operator bool() const;
};

#ifdef ENABLE_OPENCL

class gpu_kernel : public gpu_context
{
public:
    explicit gpu_kernel(const std::string &kernel_name);

    virtual ~gpu_kernel();

    static void ensure_compiled_kernel(viennacl::ocl::context &ctx, const std::string &kernel_name);
    static void ensure_compiled_kernel(viennacl::ocl::context &ctx, const std::string &kernel_name, const std::string &kernel_file_name);

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

#endif

} // namespace common

#ifdef ENABLE_OPENCL

namespace cl12 {
template<typename cl_class>
struct naf : public cl_class // Null at finish
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

} namespace c12

#endif //ENABLE_OPENCL

} // namespace svr

#include "gpu_handler.tpp"
