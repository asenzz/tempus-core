#pragma once
#include "kernel_gbm.hpp"

namespace svr {
namespace kernel {

template<typename T> boost::unordered_flat_map<datamodel::e_kernel_type, std::shared_ptr<kernel_factory<T>>> IKernel<T>::kernel_factories = [] {
    DTYPE(IKernel<T>::kernel_factories) r;
    for (auto k_type = static_cast<datamodel::e_kernel_type>(0); k_type < datamodel::e_kernel_type::end; ++k_type) r.emplace(k_type, ptr<kernel_factory<T>>(k_type));
    return r;
} ();

template<typename T> std::once_flag IKernel<T>::kernel_init_flag;

template<typename T> kernel_factory<T>::~kernel_factory()
{}

template<typename T> kernel_factory<T>::kernel_factory(const datamodel::e_kernel_type kernel_type): kernel_type_(kernel_type)
{}

template<typename T> kernel_factory<T>::bad_kernel_creation::bad_kernel_creation(const std::string &type)
{
    reason = "Cannot create type " + type;
}

template<typename T> const char *kernel_factory<T>::bad_kernel_creation::what() const noexcept
{
    return reason.c_str();
}

template<typename T> IKernel<T>::IKernel()
{
}

template<typename T> std::unique_ptr<kernel_base<T>> IKernel<T>::get(datamodel::SVRParameters &params)
{
    const auto search = kernel_factories.find(params.get_kernel_type());
    if (search == kernel_factories.cend()) THROW_EX_FS(std::invalid_argument, "Incorrect kernel type " << params.get_kernel_type());
    return search->second->create(params);
}

template<typename T> template<ckernel_base<T> K> std::unique_ptr<K> IKernel<T>::get(datamodel::SVRParameters &params)
{
    return dynamic_ptr_cast<K>(get(params));
}

template<typename T> std::unique_ptr<kernel_base<T>> IKernel<T>::new_f(datamodel::SVRParameters &params)
{
    kernel_factory<T> kf;
    return kf.create(params);
}

template<typename T> std::unique_ptr<kernel_base<T>> kernel_factory<T>::create(datamodel::SVRParameters &params)
{
    switch (kernel_type_) {

        case svr::datamodel::kernel_type::LINEAR:
            return std::make_unique<kernel_linear<T>>(params);

        case svr::datamodel::kernel_type::POLYNOMIAL:
            return std::make_unique<kernel_polynomial<T>>(params);

        case svr::datamodel::kernel_type::RBF:
            return std::make_unique<kernel_radial_basis<T>>(params);

        case svr::datamodel::kernel_type::RBF_GAUSSIAN:
            return std::make_unique<kernel_radial_basis_gaussian<T>>(params);

        case svr::datamodel::kernel_type::RBF_EXPONENTIAL:
            return std::make_unique<kernel_radial_basis_exponential<T>>(params);

        case svr::datamodel::kernel_type::MLP:
            return std::make_unique<kernel_sigmoidal<T>>(params);

        case svr::datamodel::kernel_type::GA:
            return std::make_unique<kernel_global_alignment<T>>(params);

        case svr::datamodel::kernel_type::PATH:
            return std::make_unique<kernel_path<T>>(params);

        case svr::datamodel::kernel_type::DEEP_PATH:
            return std::make_unique<kernel_deep_path<T>>(params);

        case svr::datamodel::kernel_type::DTW:
            return std::make_unique<kernel_dtw<T>>(params);

        case svr::datamodel::kernel_type::TFT:
            return std::make_unique<kernel_tft<T>>(params);

        case svr::datamodel::kernel_type::GBM:
            return std::make_unique<kernel_gbm<T>>(params);

        default:
            THROW_EX_FS(std::invalid_argument, "Incorrect kernel type " << params.get_kernel_type());
    }

    return std::make_unique<kernel_linear<T>>(params);
}

}
}
