//
// Created by jarko on 11.09.17.
//

#include "kernel_base.hpp"
#include "SVRParametersService.hpp"
#include <string>

namespace svr {

using namespace datamodel;

kernel_type_e get_kernel_type_from_string(const std::string &kernel_type_str)
{
    kernel_type_e kernel_type;
    if (kernel_type_str == "LINEAR")
        kernel_type = kernel_type_e::LINEAR;
    else if (kernel_type_str == "POLYNOMIAL")
        kernel_type = kernel_type_e::POLYNOMIAL;
    else if (kernel_type_str == "RBF")
        kernel_type = kernel_type_e::RBF;
    else if (kernel_type_str == "RBF_GAUSSIAN")
        kernel_type = kernel_type_e::RBF_GAUSSIAN;
    else if (kernel_type_str == "RBF_EXPONENTIAL")
        kernel_type = kernel_type_e::RBF_EXPONENTIAL;
    else if (kernel_type_str == "MLP")
        kernel_type = kernel_type_e::MLP;
    else if (kernel_type_str == "GA")
        kernel_type = kernel_type_e::GA;
    else if (kernel_type_str == "PATH")
        kernel_type = kernel_type_e::PATH;

    else
        throw std::invalid_argument("Incorrect kernel type.");

    return kernel_type;
}

std::string kernel_type_to_string(const kernel_type_e kernel_type)
{
    std::string kernel_type_str;
    if (kernel_type == kernel_type_e::LINEAR)
        kernel_type_str = "LINEAR";
    else if (kernel_type == kernel_type_e::POLYNOMIAL)
        kernel_type_str = "POLYNOMIAL";
    else if (kernel_type == kernel_type_e::RBF)
        kernel_type_str = "RBF";
    else if (kernel_type == kernel_type_e::RBF_GAUSSIAN)
        kernel_type_str = "RBF_GAUSSIAN";
    else if (kernel_type == kernel_type_e::RBF_EXPONENTIAL)
        kernel_type_str = "RBF_EXPONENTIAL";
    else if (kernel_type == kernel_type_e::MLP)
        kernel_type_str = "MLP";
    else if (kernel_type == kernel_type_e::GA)
        kernel_type_str = "GA";
    else if (kernel_type == kernel_type_e::PATH)
        kernel_type_str = "PATH";
    else
        throw std::invalid_argument("Incorrect kernel type.");

    return kernel_type_str;
}
}