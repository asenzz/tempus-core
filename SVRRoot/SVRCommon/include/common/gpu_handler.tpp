//
// Created by zarko on 5/5/24.
//
// TODO Refactor and clean up old logic
#pragma once

#include <cuda_runtime_api.h>
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
    const auto gpu_id = gpu_handler<ctx_per_gpu>::dev_id(context_id_);
    LOG4_TRACE("Returning GPU with ID " << gpu_id << " from context " << context_id_);
    return gpu_id;
}

template<const uint16_t ctx_per_gpu> uint16_t gpu_context_<ctx_per_gpu>::stream_id() const
{
    const auto stream_id = gpu_handler<ctx_per_gpu>::stream_id(context_id_);
    LOG4_TRACE("Returning stream with ID " << stream_id << " from context " << context_id_);
    return stream_id;
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
    if (context_id_ != gpu_handler<ctx_per_gpu>::C_no_gpu_id)
        gpu_handler<ctx_per_gpu>::get().return_gpu(context_id_);
    LOG4_TRACE("Exit ctx " << context_id_);
}


template<const uint16_t ctx_per_gpu>
gpu_handler<ctx_per_gpu>::gpu_handler()
{
#ifdef ENABLE_OPENCL
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
#endif // ENABLE_OPENCL

    max_running_gpu_threads_number_ = available_devices_.size() * ctx_per_gpu;
    LOG4_DEBUG("Max running GPU threads " << max_running_gpu_threads_number_);

#ifdef IPC_GPU
    p_gpu_sem_ = std::make_unique<typename DTYPE(p_gpu_sem_)::element_type>(
            boost::interprocess::open_or_create_t(), SVRWAVE_GPU_SEM, max_running_gpu_threads_number_, boost::interprocess::permissions(0x1FF));
    std::shared_ptr<FILE> pipe_gpu(popen("chmod a+rw /dev/shm/sem." SVRWAVE_GPU_SEM, "r"), pclose); // This hack is needed as the semaphore created with sudo privileges, does not have W rights.
#endif

}

template<const uint16_t ctx_per_gpu>
gpu_handler<ctx_per_gpu>::~gpu_handler()
{
#ifdef IPC_GPU
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

template<const uint16_t ctx_per_gpu> gpu_handler<ctx_per_gpu> &gpu_handler<ctx_per_gpu>::get()
{
    static gpu_handler handler;
    return handler;
}

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::init_devices(const cl_device_type device_type)
{
    available_devices_.clear();
    max_running_gpu_threads_number_ = 0;
    max_gpu_data_chunk_size_ = std::numeric_limits<uint32_t>::max();
    m_max_gpu_kernels_ = std::numeric_limits<uint32_t>::max();
#ifdef ENABLE_OPENCL
    auto ocl_platforms = viennacl::ocl::get_platforms();
    LOG4_INFO("Found " << ocl_platforms.size() << " platforms.");
    for (auto &pf: ocl_platforms) {
        LOG4_INFO("Platform " << pf.info());
        try {
            const auto new_devices = pf.devices(device_type);
            const auto prev_dev_size = available_devices_.size();
            for (const auto &nd: new_devices)
                available_devices_.emplace_back(device_info{
                        .id = (uint16_t) available_devices_.size(),
                        .p_mx = std::make_shared<tbb::mutex>(),
                        .p_ocl_info = std::make_shared<viennacl::ocl::device>(nd)});

            for (const auto &device: new_devices) {
                if (device.max_mem_alloc_size() && device.max_mem_alloc_size() < max_gpu_data_chunk_size_)
                    max_gpu_data_chunk_size_ = device.max_mem_alloc_size();
                uint32_t prod = 1;
                const auto item_sizes = device.max_work_item_sizes();
                for (const auto i: item_sizes) prod *= i;
                if (prod < m_max_gpu_kernels_) m_max_gpu_kernels_ = prod;
            }
            LOG4_INFO("Devices available " << available_devices_.size() * ctx_per_gpu << ", device max allocate size is " << max_gpu_data_chunk_size_ / ctx_per_gpu <<
                                           " bytes, max GPU kernels " << m_max_gpu_kernels_ / ctx_per_gpu);
            for (uint16_t i = prev_dev_size; i < available_devices_.size(); ++i) setup_context(i, *available_devices_[i].p_ocl_info);
        } catch (const std::exception &ex) {
            LOG4_ERROR("Failed enumerating platform " << pf.info() << " for devices, error " << ex.what());
        }
    }
#else
    max_running_gpu_threads_number_ = DEFAULT_GPU_NUM;
    m_max_gpu_kernels = 1;
    max_gpu_data_chunk_size_ = std::numeric_limits<uint32_t>::max();
#endif //ENABLE_OPENCL
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

template<const uint16_t ctx_per_gpu> uint16_t gpu_handler<ctx_per_gpu>::context_id(const DTYPE(available_devices_)::const_iterator &it) const
{
    assert(it->running_threads);
    assert(it->running_threads <= ctx_per_gpu);
    assert(size_t(it - available_devices_.cbegin()) < available_devices_.size());
    return it->running_threads - 1 + ctx_per_gpu * (it - available_devices_.cbegin());
}

template<const uint16_t ctx_per_gpu> uint16_t gpu_handler<ctx_per_gpu>::dev_id(const uint16_t context_id)
{
    return context_id / ctx_per_gpu;
}

template<const uint16_t ctx_per_gpu> uint16_t gpu_handler<ctx_per_gpu>::stream_id(const uint16_t context_id)
{
    return context_id % ctx_per_gpu;
}

// Use it to sort GPUs by fitness
constexpr auto comp_cuda_dev = [](const device_info &lhs, const device_info &rhs) -> bool {
    if (!lhs.p_ocl_info->available()) return false;
    if (lhs.running_threads < rhs.running_threads) return true;
    if (lhs.running_threads > rhs.running_threads) return false;

    cu_errchk(cudaSetDevice(lhs.id));
    size_t lhs_free_mem, lhs_total_mem;
    cu_errchk(cudaMemGetInfo(&lhs_free_mem, &lhs_total_mem));
    if (!lhs_free_mem) return false;

    cu_errchk(cudaSetDevice(rhs.id));
    size_t rhs_free_mem, rhs_total_mem;
    cu_errchk(cudaMemGetInfo(&rhs_free_mem, &rhs_total_mem));
    if (!rhs_free_mem) return true;

    if (lhs_free_mem > rhs_free_mem) return true;
    if (lhs_free_mem < rhs_free_mem) return false;
    if (lhs_total_mem > rhs_total_mem) return true;
    if (lhs_total_mem < rhs_total_mem) return false;

    if (lhs.p_ocl_info->max_mem_alloc_size() > rhs.p_ocl_info->max_mem_alloc_size()) return true;
    if (lhs.p_ocl_info->max_mem_alloc_size() < rhs.p_ocl_info->max_mem_alloc_size()) return false;

    return lhs.id < rhs.id;
};

template<const uint16_t ctx_per_gpu> uint16_t gpu_handler<ctx_per_gpu>::get_free_gpu()
{
#ifdef IPC_GPU
    (void) p_gpu_sem_->wait();
#endif

    std::unique_lock lck(cv_mx_);
    typename DTYPE(available_devices_)::iterator it_gpus;
    cv_.wait(lck, [&] {
        it_gpus = std::min_element(C_default_exec_policy, available_devices_.begin(), available_devices_.end(), comp_cuda_dev);
        if (it_gpus == available_devices_.cend()) return false;
        return it_gpus->running_threads < ctx_per_gpu;
    });
    ++it_gpus->running_threads;
    return context_id(it_gpus);
}

template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::try_free_gpu()
{
    auto it_gpus = std::min_element(C_default_exec_policy, available_devices_.begin(), available_devices_.end(),
                                    [&](const auto &lhs, const auto &rhs) { return lhs.running_threads < rhs.running_threads; });
    if (it_gpus == available_devices_.cend() || it_gpus->running_threads >= ctx_per_gpu) {
        LOG4_WARN("No free GPU found, running threads " << available_devices_.front().running_threads << ", back " << available_devices_.back().running_threads <<
                                                        ", it found " << (it_gpus == available_devices_.cend() ? "end" : "no end"));
        return C_no_gpu_id;
    }
    ++it_gpus->running_threads;
    return context_id(it_gpus);
}

template<const uint16_t ctx_per_gpu>
std::deque<uint16_t> gpu_handler<ctx_per_gpu>::get_free_gpus(const uint16_t gpu_ct)
{
    assert(gpu_ct < max_running_gpu_threads_number_);

#ifdef IPC_GPU
    for (uint16_t i = 0; i < gpu_ct; ++i)
        (void) p_gpu_sem_->wait();
#endif

    std::deque<uint16_t> res;
    while (res.size() < gpu_ct) res.emplace_back(get_free_gpu());
    return res;
}

template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::return_gpu(const uint16_t context_id)
{
    if (context_id == C_no_gpu_id || context_id >= max_running_gpu_threads_number_) {
        LOG4_ERROR("Wrong device index " << context_id << " of available " << available_devices_.size());
        return;
    }

#ifdef IPC_GPU
    p_gpu_sem_->post();
#endif

    auto &dev_ctx = available_devices_[dev_id(context_id)];
    const tbb::mutex::scoped_lock lck(*dev_ctx.p_mx);
    if (dev_ctx.running_threads) {
        --dev_ctx.running_threads;
        cv_.notify_one();
    } else
        LOG4_WARN("GPU " << context_id << " is already free.");
}


template<const uint16_t ctx_per_gpu>
void gpu_handler<ctx_per_gpu>::return_gpus(uint16_t gpu_ct)
{
    while (gpu_ct) {
        return_gpu();
        --gpu_ct;
    }
}


template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_max_gpu_threads() const
{
    return max_running_gpu_threads_number_;
}


template<const uint16_t ctx_per_gpu>
uint16_t gpu_handler<ctx_per_gpu>::get_gpu_devices_count() const
{
    return available_devices_.size();
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

#ifdef ENABLE_OPENCL

template<const uint16_t ctx_per_gpu>
const viennacl::ocl::device &gpu_handler<ctx_per_gpu>::device(const uint16_t idx) const
{
    if (idx >= max_running_gpu_threads_number_) LOG4_THROW("Wrong device index " << idx << " of available " << available_devices_.size());
    return *available_devices_[idx].p_ocl_info;
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
