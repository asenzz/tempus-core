#pragma once

#include "onlinesvr.hpp"
#include "common/serialization.hpp"
#include <boost/archive/binary_oarchive.hpp>

namespace svr {

template<class A> void MimoBase::serialize(A &ar, const unsigned version)
{
    ar & chunk_weights;
    ar & chunks_weight;
    ar & total_weights;
    ar & epsilon;
    ar & mae_chunk_values;
}

template<class archive> void
OnlineMIMOSVR::serialize(archive &ar, const unsigned version)
{
    ar & tmp_ixs;
    ar & samples_trained;
    ar & p_features;
    ar & p_labels;
//    ar & p_param_set; // Parameters are loaded from a table
    ar & p_manifold;
    ar & labels_scaling_factor;
    ar & p_kernel_matrices;
    ar & model_type;
    ar & main_components;
    ar & ixs;
    ar & multistep_len;
    ar & max_chunk_size;
    ar & gradient;
    ar & decon_level;
}

// TODO Rewrite using boost serialization or similar library
template<typename S> bool
OnlineMIMOSVR::save(const OnlineMIMOSVR &osvr, S &output_stream)
{
    boost::archive::binary_oarchive oa(output_stream);
    oa << osvr;
    return true;
}

template<typename S> OnlineMIMOSVR_ptr
OnlineMIMOSVR::load(S &input_stream)
{
    return std::make_shared<OnlineMIMOSVR>(input_stream);
}


} // svr
