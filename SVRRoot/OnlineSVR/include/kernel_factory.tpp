#pragma once
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* kernel factory template implementation to be included inside kernel_factory.h*/
namespace svr {
namespace kernel {

template<typename T> std::unordered_map<datamodel::e_kernel_type, std::shared_ptr<kernel_factory<T>>> IKernel<T>::kernel_factories;

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


template<typename T> void IKernel<T>::init()
{
    std::call_once(IKernel<T>::kernel_init_flag, [] {
        try {
            for (auto k_type = datamodel::e_kernel_type::begin; k_type < datamodel::e_kernel_type::end; ++k_type)
                IKernel<T>::kernel_factories.emplace(k_type, ptr<kernel_factory<T>>(k_type));
        } catch (const std::exception &ex) {
            LOG4_ERROR("Factory init failed, " << ex.what());
        }
    });
}

template<typename T> IKernel<T>::IKernel()
{
    IKernel<T>::init();
}

template<typename T> std::unique_ptr<kernel_base<T>> IKernel<T>::get(const datamodel::SVRParameters &params)
{
    const auto search = kernel_factories.find(params.get_kernel_type());
    if (search == kernel_factories.cend()) THROW_EX_FS(std::invalid_argument, "Incorrect kernel type " << params.get_kernel_type());
    return search->second->create(params);
}

template<typename T> std::unique_ptr<kernel_base<T>> IKernel<T>::newk(const datamodel::SVRParameters &params)
{
    kernel_factory<T> kf;
    return kf.create(params);
}

template<typename T> std::unique_ptr<kernel_base<T>> kernel_factory<T>::create(const datamodel::SVRParameters &params)
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

        case svr::datamodel::kernel_type::DTW:
            return std::make_unique<kernel_dtw<T>>(params);

        default:
            THROW_EX_FS(std::invalid_argument, "Incorrect kernel type " << params.get_kernel_type());
    }

    return std::make_unique<kernel_linear<T>>(params);
}

}
}