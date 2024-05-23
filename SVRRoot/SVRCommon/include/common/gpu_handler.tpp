//
// Created by zarko on 5/5/24.
//

#pragma once

#include "gpu_handler.hpp"

namespace svr {
namespace common {

template<const unsigned context_per_gpu>
gpu_handler<context_per_gpu>::gpu_handler()
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
    p_gpu_sem_ = std::make_unique<typename dtype(p_gpu_sem_)::element_type>(
#ifdef IPC_SEMAPHORE
            boost::interprocess::open_or_create_t(), SVRWAVE_GPU_SEM, max_running_gpu_threads_number_, boost::interprocess::permissions(0x1FF));
    std::shared_ptr<FILE> pipe_gpu(popen("chmod a+rw /dev/shm/sem." SVRWAVE_GPU_SEM, "r"), pclose); // This hack is needed as the semaphore created with sudo privileges, does not have W rights.
#else
        max_running_gpu_threads_number_);
#endif
}

template<const unsigned context_per_gpu>
gpu_handler<context_per_gpu>::~gpu_handler()
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

template<const unsigned context_per_gpu>
gpu_handler<context_per_gpu> &gpu_handler<context_per_gpu>::get()
{
    static gpu_handler handler;
    return handler;
}

template<const unsigned context_per_gpu>
void gpu_handler<context_per_gpu>::init_devices(const int device_type)
{
    const std::scoped_lock wl(devices_mutex_);
    devices_.clear();
    max_running_gpu_threads_number_ = 0;
    max_gpu_data_chunk_size_ = std::numeric_limits<size_t>::max();
    m_max_gpu_kernels_ = std::numeric_limits<size_t>::max();
#ifdef VIENNACL_WITH_OPENCL
    auto ocl_platforms = viennacl::ocl::get_platforms();
    LOG4_INFO("Found " << ocl_platforms.size() << " platforms.");
    for (auto &pf: ocl_platforms) {
        LOG4_INFO("Platform " << pf.info());
        try {
            const auto new_devices = pf.devices(device_type);
            const auto prev_dev_size = devices_.size();
            for (unsigned mult = 0; mult < context_per_gpu; ++mult)
                devices_.insert(devices_.begin(), new_devices.cbegin(), new_devices.cend());

            max_running_gpu_threads_number_ += new_devices.size() * context_per_gpu;
            for (const auto &device: new_devices) {
                if (device.max_mem_alloc_size() && device.max_mem_alloc_size() < max_gpu_data_chunk_size_)
                    max_gpu_data_chunk_size_ = device.max_mem_alloc_size();
                size_t prod = 1;
                std::vector<size_t> item_sizes = device.max_work_item_sizes();
                for (const auto i: item_sizes) prod *= i;
                if (prod < m_max_gpu_kernels_) m_max_gpu_kernels_ = prod;
            }
            LOG4_INFO("Devices available " << max_running_gpu_threads_number_ << ", device max allocate size is " << max_gpu_data_chunk_size_ <<
                                           " bytes, max GPU kernels " << m_max_gpu_kernels_);
            for (size_t i = 0; i < devices_.size(); ++i) {
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
    max_gpu_data_chunk_size_ = std::numeric_limits<size_t>::max();
#endif //VIENNACL_WITH_OPENCL
}

template<const unsigned context_per_gpu>
void gpu_handler<context_per_gpu>::sort_free_gpus()
{
#if 0
    std::sort(std::execution::par_unseq, free_gpus_.begin(), free_gpus_.end(), [&](const auto lhs, const auto rhs) {
        size_t lhs_free, lhs_total, rhs_free, rhs_total;
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


template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_free_gpu()
{
    (void) p_gpu_sem_->wait();
    return ++next_device_ % max_running_gpu_threads_number_;
}

template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_free_gpus(const size_t gpu_ct)
{
    for (size_t i = 0; i < gpu_ct; ++i)
        (void) p_gpu_sem_->wait();
    return (next_device_ += gpu_ct) % max_running_gpu_threads_number_;
}

template<const unsigned context_per_gpu>
void gpu_handler<context_per_gpu>::return_gpu()
{
    p_gpu_sem_->post();
}

template<const unsigned context_per_gpu>
void gpu_handler<context_per_gpu>::return_gpus(const size_t gpu_ct)
{
    for (size_t i = 0; i < gpu_ct; ++i)
        (void) p_gpu_sem_->wait();
    p_gpu_sem_->post();
}


template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_max_running_gpu_threads_number() const
{
    return max_running_gpu_threads_number_;
}


template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_gpu_devices_count() const
{
    return max_running_gpu_threads_number_ / context_per_gpu;
}


template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_max_gpu_kernels() const
{
    return m_max_gpu_kernels_ / context_per_gpu;
}


template<const unsigned context_per_gpu>
size_t gpu_handler<context_per_gpu>::get_max_gpu_data_chunk_size() const
{
    return max_gpu_data_chunk_size_ / context_per_gpu;
}


#ifdef VIENNACL_WITH_OPENCL

template<const unsigned context_per_gpu>
const viennacl::ocl::device &gpu_handler<context_per_gpu>::device(const size_t idx) const
{
    const boost::shared_lock<boost::shared_mutex> rl(devices_mutex_);
    if (idx >= devices_.size()) LOG4_THROW("Wrong device index " << idx << " of available " << devices_.size());
    return devices_[idx];
}

#endif

}
}
