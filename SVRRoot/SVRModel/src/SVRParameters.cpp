//
// Created by zarko on 1/13/24.
//

#include "model/SVRParameters.hpp"

namespace svr {
namespace datamodel {

bool less_SVRParameters_ptr::operator()(const datamodel::SVRParameters_ptr &lhs, const datamodel::SVRParameters_ptr &rhs) const
{
    return lhs->operator < (*rhs);
}


}
}