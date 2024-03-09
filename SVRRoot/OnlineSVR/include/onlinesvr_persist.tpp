#pragma once

#include "onlinesvr.hpp"
#include "common/serialization.hpp"
#include <boost/archive/binary_oarchive.hpp>

namespace svr {
namespace datamodel {

template<class A> void
OnlineMIMOSVR::serialize(A &ar, const unsigned version)
{
    ar & tmp_ixs;
    ar & samples_trained;
    ar & p_features;
    ar & p_labels;
    ar & p_manifold;
    ar & labels_scaling_factor;
    ar & p_kernel_matrices;
    ar & chunk_weights;
    ar & chunks_weight;
    ar & chunk_bias;
    ar & chunk_mae;
    ar & ixs;
    ar & multistep_len;
    ar & max_chunk_size;
    ar & gradient;
    ar & decon_level;
}

// TODO Rewrite using boost serialization or similar library
template<typename S> void
OnlineMIMOSVR::save(const OnlineMIMOSVR &osvr, S &output_stream)
{
    boost::archive::binary_oarchive oa(output_stream);
    oa << osvr;
}

template<typename S> OnlineMIMOSVR_ptr
OnlineMIMOSVR::load(S &input_stream)
{
    return ptr<OnlineMIMOSVR>(input_stream);
}

} // datamodel
} // svr
