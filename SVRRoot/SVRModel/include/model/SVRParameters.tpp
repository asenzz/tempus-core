//
// Created by zarko on 31/07/2025.
//

#ifndef SVR_SVRPARAMETERS_TPP
#define SVR_SVRPARAMETERS_TPP

#include "SVRParameters.hpp"

namespace svr {
namespace datamodel {

template<typename S> void t_feature_mechanics::save(const t_feature_mechanics &feature_mechanics, S &output_stream) const
{
    boost::archive::binary_oarchive oa(output_stream);
    oa << feature_mechanics;
}

template<class A> void t_feature_mechanics::serialize(A &ar, const uint32_t version)
{
    ar & quantization;
    ar & stretches;
    ar & trims;
    ar & shifts;
    ar & steps;
}

}
}

#endif //SVR_SVRPARAMETERS_TPP