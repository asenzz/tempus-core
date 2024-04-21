#pragma once
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* kernel factory template implementation to be included inside kernel_factory.h*/
namespace svr {


using svr::datamodel::kernel_type_e;

template<typename scalar_type>
std::unordered_map<kernel_type_e, std::shared_ptr<kernel_factory<scalar_type>>> IKernel<scalar_type>::kernel_factories;

template<typename scalar_type>
std::once_flag IKernel<scalar_type>::kernel_init_flag;

template<typename scalar_type>
void IKernel<scalar_type>::IKernelInit()
{
    std::call_once(IKernel<scalar_type>::kernel_init_flag, [](){
        try {
            for (auto k_type = kernel_type_e(0); (int) k_type < (int) kernel_type_e::number_of_kernel_types; ++k_type)
                IKernel<scalar_type>::kernel_factories.emplace(k_type, ptr<kernel_factory<scalar_type>>(k_type));
        } catch (const std::exception &ex) {
            LOG4_ERROR("Something very bad happened during kernel factory init, " << ex.what());
        }
    });
}


template<typename scalar_type>
IKernel<scalar_type>::IKernel()
{
    IKernel<scalar_type>::IKernelInit();
}


template<typename scalar_type>
std::unique_ptr<kernel_base<scalar_type>>
IKernel<scalar_type>::get(const SVRParameters &params)
{
    svr::IKernel<scalar_type>::IKernelInit();
    const auto search = kernel_factories.find(params.get_kernel_type());
    if (search == kernel_factories.end())
        THROW_EX_FS(std::invalid_argument, "Incorrect kernel type " << tostring<const char *>(params.get_kernel_type()) << ", we don't know what to do yet.");
    return search->second->create(params);
}

template<typename scalar_type>
std::unique_ptr<kernel_base<scalar_type>>
kernel_factory<scalar_type>::create(const SVRParameters &params)
{
    switch (kernel_type_) {
        case svr::datamodel::kernel_type::LINEAR:
            return std::make_unique<kernel_linear<scalar_type> >(kernel_linear<scalar_type>(params));
        case svr::datamodel::kernel_type::POLYNOMIAL:
            return std::make_unique<kernel_polynomial<scalar_type> >(kernel_polynomial<scalar_type>(params));
        case svr::datamodel::kernel_type::RBF:
            return std::make_unique<kernel_radial_basis<scalar_type> >(kernel_radial_basis<scalar_type>(params));
        case svr::datamodel::kernel_type::RBF_GAUSSIAN:
            return std::make_unique<kernel_radial_basis_gaussian<scalar_type> >(
                    kernel_radial_basis_gaussian<scalar_type>(params));
        case svr::datamodel::kernel_type::RBF_EXPONENTIAL:
            return std::make_unique<kernel_radial_basis_exponential<scalar_type> >(
                    kernel_radial_basis_exponential<scalar_type>(params));
        case svr::datamodel::kernel_type::MLP:
            return std::make_unique<kernel_sigmoidal<scalar_type>>(kernel_sigmoidal<scalar_type>(params));
        case svr::datamodel::kernel_type::GA:
            return std::make_unique<kernel_global_alignment<scalar_type>>(kernel_global_alignment<scalar_type>(params));
        case svr::datamodel::kernel_type::PATH:
            return std::make_unique<kernel_path<scalar_type> >(kernel_path<scalar_type>(params));
        default:
            throw std::invalid_argument("Incorrect kernel type.");
    }
}
// Static member definition and initializer on compile time - a C++ strong feature!
//KernelFactoryInizializer KernelFactoryInizializer::KernelsInitializer;
}
