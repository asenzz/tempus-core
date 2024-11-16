//
// Created by zarko on 5/5/24.
//
// TODO Refactor and clean up old logic
#pragma once

#include "gpu_handler.hpp"
#include "compatibility.hpp"

namespace svr {
namespace common {

// GPU Context, set the worker as blocked during its lifetime.
template<const uint16_t ctx_per_gpu> __attribute_noinline__ gpu_context_<ctx_per_gpu>::gpu_context_() :
        context_id_(gpu_handler<ctx_per_gpu>::get().get_free_gpu()), dev_ct(gpu_handler<ctx_per_gpu>::get().get_gpu_devices_count())
{
    LOG4_TRACE("Enter ctx " << context_id_);
}

template<const uint16_t ctx_per_gpu> __attribute_noinline__ gpu_context_<ctx_per_gpu>::gpu_context_(const bool try_init) :
        context_id_(try_init ? gpu_handler<ctx_per_gpu>::get().try_free_gpu() : gpu_handler<ctx_per_gpu>::get().get_free_gpu()),
        dev_ct(gpu_handler<ctx_per_gpu>::get().get_gpu_devices_count())
{
    LOG4_TRACE("Enter ctx " << context_id_);
}

template<const uint16_t ctx_per_gpu> __attribute_noinline__ gpu_context_<ctx_per_gpu>::gpu_context_(const gpu_context &context) :
        context_id_(context.context_id_), dev_ct(gpu_handler<ctx_per_gpu>::get().get_gpu_devices_count())
{};

template<const uint16_t ctx_per_gpu> uint16_t gpu_context_<ctx_per_gpu>::id() const
{
    return context_id_;
}

template<const uint16_t ctx_per_gpu> uint16_t gpu_context_<ctx_per_gpu>::phy_id() const
{
    const uint16_t gpu_id = context_id_ % dev_ct;
    LOG4_TRACE("Returning GPU with ID " << gpu_id);
    return gpu_id;
}

template<const uint16_t ctx_per_gpu> viennacl::ocl::context &gpu_context_<ctx_per_gpu>::ctx() const
{
    return viennacl::ocl::get_context(context_id_);
}

template<const uint16_t ctx_per_gpu> gpu_context_<ctx_per_gpu>::operator bool() const
{
    return context_id_ != gpu_handler<ctx_per_gpu>::C_no_gpu_id;
}

template<const uint16_t ctx_per_gpu> gpu_context_<ctx_per_gpu>::~gpu_context_()
{
#ifdef GPU_QUEUE
    if (context_id_ != gpu_handler<ctx_per_gpu>::C_no_gpu_id) gpu_handler<ctx_per_gpu>::get().return_gpu(context_id_);
#else
    gpu_handler<ctx_per_gpu>::get().return_gpu();
#endif
    LOG4_TRACE("Exit ctx " << context_id_);
}


template<const uint16_t ctx_per_gpu>
gpu_handler<ctx_per_gpu>::gpu_handler()
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
#ifdef GPU_QUEUE
    available_devices_.set_capacity(devices_.size());
    for (uint16_t devid = 0; devid < devices_.size(); ++devid) available_devices_.emplace(devid);
#else
    p_gpu_sem_ = std::make_unique<typename DTYPE(p_gpu_sem_)::element_type>(
#ifdef IPC_SEMAPHORE
            boost::interprocess::open_or_create_t(), SVRWAVE_GPU_SEM, max_running_gpu_threads_number_, boost::interprocess::permissions(0x1FF));
    std::shared_ptr<FILE> pipe_gpu(popen("chmod a+rw /dev/shm/sem." SVRWAVE_GPU_SEM, "r"), pclose); // This hack is needed as the semaphore created with sudo privileges, does not have W rights.
#else
        max_running_gpu_threads_number_);
#endif
#endif
}

template<const uint16_t ctx_per_gpu>
gpu_handler<ctx_per_gpu>::~gpu_handler()
{
#ifdef IPC_SEMAPHORE
#if 0
    try {
        if (!named_semaphore::remove(SVRWAVE_GPU_SEM))
            LOG4_WARN("Failed removing named semaphore " << SVRWAVE_GPU_SEM);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Exception " << ex.what());
    }
#endif
#endif
}

template<const uint16_t ctx_per_gpu>
gpu_handler<ctx_per_gpu> &gpu_handler<ctx_per_gpu>::get()
{
    static gpu_handler handler;
    return handler;
}

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::init_devices(const int device_type)
{
    const std::scoped_lock wl(devices_mutex_);
    devices_.clear();
    max_running_gpu_threads_number_ = 0;
    max_gpu_data_chunk_size_ = std::numeric_limits<uint32_t>::max();
    m_max_gpu_kernels_ = std::numeric_limits<uint32_t>::max();
#ifdef VIENNACL_WITH_OPENCL
    auto ocl_platforms = viennacl::ocl::get_platforms();
    LOG4_INFO("Found " << ocl_platforms.size() << " platforms.");
    for (auto &pf: ocl_platforms) {
        LOG4_INFO("Platform " << pf.info());
        try {
            const auto new_devices = pf.devices(device_type);
            const auto prev_dev_size = devices_.size();
            UNROLL(ctx_per_gpu)
            for (uint16_t mult = 0; mult < ctx_per_gpu; ++mult)
                devices_.insert(devices_.begin(), new_devices.cbegin(), new_devices.cend());

            max_running_gpu_threads_number_ += new_devices.size() * ctx_per_gpu;
            UNROLL(ctx_per_gpu)
            for (const auto &device: new_devices) {
                if (device.max_mem_alloc_size() && device.max_mem_alloc_size() < max_gpu_data_chunk_size_)
                    max_gpu_data_chunk_size_ = device.max_mem_alloc_size();
                uint32_t prod = 1;
                const auto item_sizes = device.max_work_item_sizes();
                for (const auto i: item_sizes) prod *= i;
                if (prod < m_max_gpu_kernels_) m_max_gpu_kernels_ = prod;
            }
            LOG4_INFO("Devices available " << max_running_gpu_threads_number_ << ", device max allocate size is " << max_gpu_data_chunk_size_ <<
                                           " bytes, max GPU kernels " << m_max_gpu_kernels_);
            UNROLL(ctx_per_gpu)
            for (uint16_t i = 0; i < devices_.size(); ++i) {
                const auto devix = prev_dev_size + i;
                setup_context(devix, devices_[devix]);
            }
        } catch (const std::exception &ex) {
            LOG4_ERROR("Failed enumerating platform " << pf.info() << " for devices, error " << ex.what());
        }
    }
#else
    max_running_gpu_threads_number_ = DEFAULT_GPU_NUM;
    m_max_gpu_kernels = 1;
    max_gpu_data_chunk_size_ = std::numeric_limits<uint32_t>::max();
#endif //VIENNACL_WITH_OPENCL
}

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::sort_free_gpus()
{
#if 0
    std::sort(C_default_exec_policy, free_gpus_.begin(), free_gpus_.end(), [&](const auto lhs, const auto rhs) {
        uint32_t lhs_free, lhs_total, rhs_free, rhs_total;
#pragma omp parallel num_threads(2)
#pragma omp single
        {
#pragma omp task
            {
                cu_errchk(cudaSetDevice(lhs));
                cu_errchk(cudaMemGetInfo(&lhs_free, &lhs_total));
            }
#pragma omp task
            {
                cu_errchk(cudaSetDevice(rhs));
                cu_errchk(cudaMemGetInfo(&rhs_free, &rhs_total));
            }
        }
        if (lhs_free < rhs_free) return true;
        return double(lhs_free) / double(lhs_total) < double(rhs_free) / double(rhs_total);
    });
#endif
}

template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_free_gpu()
{
#ifdef GPU_QUEUE
    uint16_t gpu_id;
    available_devices_.pop(gpu_id);
    return gpu_id;
#else
    (void) p_gpu_sem_->wait();
    return next_device_++ % max_running_gpu_threads_number_;
#endif
}

template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::try_free_gpu()
{
    uint16_t gpu_id;
    return available_devices_.try_pop(gpu_id) ? gpu_id : C_no_gpu_id;
}

template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_free_gpus(const uint16_t gpu_ct)
{
#ifdef GPU_QUEUE
    LOG4_THROW("Not implemented."); // TODO..
    return 0;
#else
    for (uint16_t i = 0; i < gpu_ct; ++i)
        (void) p_gpu_sem_->wait();
    return (next_device_ += gpu_ct) % max_running_gpu_threads_number_;
#endif
}

#ifdef GPU_QUEUE

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::return_gpu(const uint16_t id)
{
    if (id == C_no_gpu_id) return;
    available_devices_.push(id);
}

#else

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::return_gpu()
{
    p_gpu_sem_->post();
}

#endif

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::return_gpus(const uint16_t gpu_ct)
{
#ifdef GPU_QUEUE
    LOG4_THROW("Not implemented."); // TODO..
#else
    for (uint16_t i = 0; i < gpu_ct; ++i)
        (void) p_gpu_sem_->wait();
    p_gpu_sem_->post();
#endif
}


template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_max_gpu_threads() const
{
    return max_running_gpu_threads_number_;
}


template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_gpu_devices_count() const
{
    return max_running_gpu_threads_number_ / ctx_per_gpu;
}


template<const uint16_t ctx_per_gpu>
uint32_t gpu_handler<ctx_per_gpu>::get_max_gpu_kernels() const
{
    return m_max_gpu_kernels_ / ctx_per_gpu;
}


template<const uint16_t ctx_per_gpu>
size_t gpu_handler<ctx_per_gpu>::get_max_gpu_data_chunk_size() const
{
    return max_gpu_data_chunk_size_ / ctx_per_gpu;
}

#ifdef VIENNACL_WITH_OPENCL

template<const uint16_t ctx_per_gpu>
const viennacl::ocl::device &gpu_handler<ctx_per_gpu>::device(const uint16_t idx) const
{
    const boost::shared_lock<boost::shared_mutex> rl(devices_mutex_);
    if (idx >= devices_.size()) LOG4_THROW("Wrong device index " << idx << " of available " << devices_.size());
    return devices_[idx];
}

#endif

}


namespace cl12 {

template<typename handler_type> kernel_helper::kernel_helper(handler_type const &handler) : ::cl::Kernel(handler)
{};

template<class... Args> kernel_helper &kernel_helper::set_args(Args... args)
{
    set_args(sizeof...(Args), args...);
    return *this;
}

template<class T> kernel_helper &kernel_helper::set_arg(const size_t position, T t)
{
    const auto code = setArg(position, t);
    if (code != CL_SUCCESS) LOG4_THROW("OpenCL adding argument failed. Error: " << svr::common::gpu_helper::get_error_string(code));
    return *this;
}

template<class T, class... Args> void kernel_helper::set_args(const size_t args_sz, T t, Args... args)
{
    const auto code = setArg(args_sz - sizeof...(Args) - 1, t);
    if (code != CL_SUCCESS) LOG4_THROW("OpenCL adding argument failed. Error: " << svr::common::gpu_helper::get_error_string(code));
    set_args(args_sz, args...);
}

template<class T> void kernel_helper::set_args(const size_t args_sz, T t)
{
    const auto code = setArg(args_sz - 1, t);
    if (code != CL_SUCCESS) LOG4_THROW("OpenCL adding argument failed. Error: " << svr::common::gpu_helper::get_error_string(code));
}

}

}
