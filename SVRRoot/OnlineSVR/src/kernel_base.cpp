//
// Created by jarko on 11.09.17.
//

#include "kernel_base.hpp"
#include "SVRParametersService.hpp"
#include <string>

namespace svr {

using namespace datamodel;

e_kernel_type get_kernel_type_from_string(const std::string &kernel_type_str)
{
    e_kernel_type kernel_type;
    if (kernel_type_str == "LINEAR")
        kernel_type = e_kernel_type::LINEAR;
    else if (kernel_type_str == "POLYNOMIAL")
        kernel_type = e_kernel_type::POLYNOMIAL;
    else if (kernel_type_str == "RBF")
        kernel_type = e_kernel_type::RBF;
    else if (kernel_type_str == "RBF_GAUSSIAN")
        kernel_type = e_kernel_type::RBF_GAUSSIAN;
    else if (kernel_type_str == "RBF_EXPONENTIAL")
        kernel_type = e_kernel_type::RBF_EXPONENTIAL;
    else if (kernel_type_str == "MLP")
        kernel_type = e_kernel_type::MLP;
    else if (kernel_type_str == "GA")
        kernel_type = e_kernel_type::GA;
    else if (kernel_type_str == "PATH")
        kernel_type = e_kernel_type::PATH;

    else
        throw std::invalid_argument("Incorrect kernel type.");

    return kernel_type;
}

std::string kernel_type_to_string(const e_kernel_type kernel_type)
{
    std::string kernel_type_str;
    if (kernel_type == e_kernel_type::LINEAR)
        kernel_type_str = "LINEAR";
    else if (kernel_type == e_kernel_type::POLYNOMIAL)
        kernel_type_str = "POLYNOMIAL";
    else if (kernel_type == e_kernel_type::RBF)
        kernel_type_str = "RBF";
    else if (kernel_type == e_kernel_type::RBF_GAUSSIAN)
        kernel_type_str = "RBF_GAUSSIAN";
    else if (kernel_type == e_kernel_type::RBF_EXPONENTIAL)
        kernel_type_str = "RBF_EXPONENTIAL";
    else if (kernel_type == e_kernel_type::MLP)
        kernel_type_str = "MLP";
    else if (kernel_type == e_kernel_type::GA)
        kernel_type_str = "GA";
    else if (kernel_type == e_kernel_type::PATH)
        kernel_type_str = "PATH";
    else
        throw std::invalid_argument("Incorrect kernel type.");

    return kernel_type_str;
}
}