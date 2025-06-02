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
#include <string>
#include <exception>
#include <boost/unordered/unordered_flat_map.hpp>
#include "kernel_base.hpp"
#include "kernel_linear.hpp"
#include "kernel_polynomial.hpp"
#include "kernel_radial_basis.hpp"
#include "kernel_radial_basis_gaussian.hpp"
#include "kernel_radial_basis_exponential.hpp"
#include "kernel_sigmoidal.hpp"
#include "kernel_global_alignment.hpp"
#include "kernel_path.hpp"
#include "kernel_dtw.hpp"
#include "kernel_deep_path.hpp"

namespace svr {
namespace kernel {

template<typename T>
class kernel_factory final {
public:
    std::unique_ptr<kernel_base<T>> create(const datamodel::SVRParameters &params);

    ~kernel_factory();

    explicit kernel_factory(const datamodel::e_kernel_type kernel_type);

    class bad_kernel_creation : public std::exception {
        std::string reason;
    public:
        explicit bad_kernel_creation(const std::string &type);

        const char *what() const noexcept override;
    };

    datamodel::e_kernel_type kernel_type_;
};

template<typename Derived, typename T> concept ckernel_base = std::derived_from<Derived, kernel::kernel_base<T>>;

template<typename T> class IKernel final {
    static boost::unordered_flat_map<datamodel::e_kernel_type, std::shared_ptr<kernel_factory<T>>> kernel_factories;
    static std::once_flag kernel_init_flag;

public:
    IKernel();

    ~IKernel() = default;

    template<ckernel_base<T> K> static std::unique_ptr<K> get(const datamodel::SVRParameters &params);

    static std::unique_ptr<kernel_base<T>> get(const datamodel::SVRParameters &params);

    std::unique_ptr<kernel_base<T>> new_f(const datamodel::SVRParameters &params);
};

}
}

#include "kernel_factory.tpp"


#endif /* KERNEL_FACTORY_H */
