#include <common/gpu_handler.hpp>
#include <common/common.hpp>


namespace svr {
namespace common {


gpu_handler::gpu_handler()
{
#ifdef VIENNACL_WITH_OPENCL
    bool try_ocl_cpu = false;
    try {
        init_devices(CL_DEVICE_TYPE_GPU);
    } catch (...) {
        try_ocl_cpu = true;
        LOG4_WARN("No GPU device could be found. Using CPUs instead.");
    }
    try {
        if (try_ocl_cpu) init_devices(CL_DEVICE_TYPE_CPU);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Failed initializing OpenCL, " << ex.what());
        throw;
    }
#endif // VIENNACL_WITH_OPENCL

    LOG4_DEBUG("Max running GPU threads " << max_running_gpu_threads_number_);
    p_gpu_sem_ = std::make_unique<boost::interprocess::named_semaphore>(
            boost::interprocess::open_or_create_t(), SVRWAVE_GPU_SEM, max_running_gpu_threads_number_, boost::interprocess::permissions(0x1FF));
    // This hack is needed as the semaphore created with sudo privileges, does not have W rights.
    std::shared_ptr<FILE> pipe_gpu(popen("chmod a+rw /dev/shm/sem." SVRWAVE_GPU_SEM, "r"), pclose);
}

gpu_handler::~gpu_handler()
{
#if 0
    try {
        if (!named_semaphore::remove(SVRWAVE_GPU_SEM))
            LOG4_WARN("Failed removing named semaphore " << SVRWAVE_GPU_SEM);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Exception " << ex.what());
    }
#endif
}


void gpu_handler::gpu_sem_enter()
{
    p_gpu_sem_->wait();
}


bool gpu_handler::gpu_sem_try_enter()
{
    return p_gpu_sem_->try_wait();
}


void gpu_handler::gpu_sem_leave()
{
    p_gpu_sem_->post();
}


gpu_handler &gpu_handler::get()
{
    static gpu_handler handler;
    return handler;
}

void gpu_handler::init_devices(const int device_type)
{
    std::scoped_lock wl(devices_mutex_);
    devices_.clear();
    max_running_gpu_threads_number_ = 0;
    max_gpu_data_chunk_size_ = std::numeric_limits<size_t>::max();
    m_max_gpu_kernels_ = std::numeric_limits<size_t>::max();
#ifdef VIENNACL_WITH_OPENCL
    auto ocl_platforms = viennacl::ocl::get_platforms();
    LOG4_INFO("Found " << ocl_platforms.size() << " platforms.");
    for (auto &pf: ocl_platforms) {
        LOG4_INFO("Platform info " << pf.info());
        try {
            const auto new_devices = pf.devices(device_type);
            const auto prev_dev_size = devices_.size();
            for (ushort mult = 0; mult < ushort(CTX_PER_GPU); ++mult)
                devices_.insert(devices_.begin(), new_devices.begin(), new_devices.end());

            max_running_gpu_threads_number_ += new_devices.size() * CTX_PER_GPU;
            for (const auto &device: new_devices) {
                if (device.max_mem_alloc_size() && device.max_mem_alloc_size() < max_gpu_data_chunk_size_)
                    max_gpu_data_chunk_size_ = device.max_mem_alloc_size();
                size_t prod = 1;
                std::vector<size_t> item_sizes = device.max_work_item_sizes();
                for (const size_t i: item_sizes) prod *= i;
                if (prod < m_max_gpu_kernels_) m_max_gpu_kernels_ = prod;
            }
            LOG4_INFO("Devices available " << max_running_gpu_threads_number_ << ", device max allocate size is " << max_gpu_data_chunk_size_ << " bytes, max GPU kernels " << m_max_gpu_kernels_);
            for (size_t i = 0; i < devices_.size(); ++i) {
                setup_context(prev_dev_size + i, devices_[prev_dev_size + i]);
                free_gpus_.push(prev_dev_size + i);
            }
        } catch (...) {
            LOG4_ERROR("Failed enumerating platform " << pf.info() << " for devices!");
        }
    }
#else
    max_running_gpu_threads_number_ = DEFAULT_GPU_NUM;
    m_max_gpu_kernels = 1;
    max_gpu_data_chunk_size_ = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < DEFAULT_GPU_NUM; ++i) {
        free_gpus_.push(i);
    }
#endif //VIENNACL_WITH_OPENCL
}


size_t gpu_handler::get_free_gpus() const
{
    const std::scoped_lock l(free_gpu_mutex_);
    return free_gpus_.size();
}


size_t gpu_handler::get_free_gpu()
{
    const std::scoped_lock l(free_gpu_mutex_);
    const auto free_gpu_index = free_gpus_.top();
    free_gpus_.pop();
    return free_gpu_index;
}


void gpu_handler::return_gpu(const size_t gpu_index)
{
    std::scoped_lock l(free_gpu_mutex_);
    free_gpus_.push(gpu_index);
}


size_t gpu_handler::get_max_running_gpu_threads_number() const
{
    return max_running_gpu_threads_number_;
}


size_t gpu_handler::get_gpu_devices_count() const
{
    return max_running_gpu_threads_number_ / CTX_PER_GPU;
}


size_t gpu_handler::get_max_gpu_kernels() const
{
    return m_max_gpu_kernels_;
}


size_t gpu_handler::get_max_gpu_data_chunk_size() const
{
    return max_gpu_data_chunk_size_;
}


#ifdef VIENNACL_WITH_OPENCL

const viennacl::ocl::device &gpu_handler::device(const size_t idx) const
{
    const boost::shared_lock<boost::shared_mutex> rl(devices_mutex_);
    if (idx >= devices_.size()) LOG4_THROW("Wrong device index " << idx << " of available " << devices_.size());
    return devices_[idx];
}

// GPU Context, set the worker as blocked during its lifetime.
gpu_context::gpu_context()
{
    gpu_handler::get().gpu_sem_enter();
    context_id_ = gpu_handler::get().get_free_gpu();
    LOG4_TRACE("Enter ctx " << context_id_);
}


gpu_context::~gpu_context()
{
    gpu_handler::get().return_gpu(context_id_);
    gpu_handler::get().gpu_sem_leave();
    LOG4_TRACE("Exit ctx " << context_id_);
}

static std::map<std::string, std::stringstream> kernel_file_cache;

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


gpu_kernel::gpu_kernel(const std::string &kernel_name)
        : kernel_name_(kernel_name)
{
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
    if (err != CL_SUCCESS)
        LOG4_THROW("clEnqueueNDRangeKernel failed, " << get_error_string(err));
}

} //namespace common

#endif //VIENNACL_WITH_OPENCL

namespace cl12 {
cl_int
kernel_helper::enqueue(
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
}

} //namespace svr
