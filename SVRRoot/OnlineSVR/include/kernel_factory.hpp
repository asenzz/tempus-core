/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   kernel_factory.h
 * Author: mitko
 *
 * Kernel factory needs to be integrated in as single C++ translation unit.
 * Currently the project configuration generates some errors while compiled with the factory as single unit.
 * Thus kernel factory lives in OnlineSVR.h/.cpp now.
 * 
 * Created on March 5, 2018, 12:45 PM
 */

#ifndef KERNEL_FACTORY_H
#define KERNEL_FACTORY_H


#include <vector>
#include <map>
#include <string>
#include <exception>


#include <kernel_base.hpp>
#include <kernel_linear.hpp>
#include <kernel_polynomial.hpp>
#include <kernel_radial_basis.hpp>
#include <kernel_radial_basis_gaussian.hpp>
#include <kernel_radial_basis_exponential.hpp>
#include <kernel_sigmoidal.hpp>
#include <kernel_global_alignment.hpp>
#include <kernel_path.hpp>

using svr::datamodel::kernel_type_e;

namespace svr{


template<typename scalar_type>
class kernel_factory {
public:

    virtual std::unique_ptr<kernel_base<scalar_type>> create(const SVRParameters & params);


    virtual ~kernel_factory() {}
    explicit kernel_factory(kernel_type_e kernel_type): kernel_type_(kernel_type){};

    class bad_kernel_creation : public std::exception {
        std::string reason;
    public:
        bad_kernel_creation(const std::string &type) {
            reason = "Cannot create type " + type;
        }
        const char *what() const noexcept override {
            return reason.c_str();
        }
    };

    kernel_type_e kernel_type_;


};

//class IKernal provides access to kernel factory
template<typename scalar_type>
class IKernel {
    static std::map<kernel_type_e, kernel_factory<scalar_type>*> kernel_factories;
    static std::once_flag kernel_init_flag;

public:
    //must call this before use IKernel
    static void IKernelInit();
    IKernel();
    ~IKernel() = default;
    static std::unique_ptr<kernel_base<scalar_type>> get_kernel(const kernel_type_e& ktype, const SVRParameters & params);
};
}

#include <kernel_factory.tcc>


#endif /* KERNEL_FACTORY_H */
