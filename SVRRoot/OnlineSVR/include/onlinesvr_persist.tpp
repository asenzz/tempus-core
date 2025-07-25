#pragma once

#include "onlinesvr.hpp"
#include "common/serialization.hpp"
#include <boost/archive/binary_oarchive.hpp>

namespace svr {
namespace datamodel {

template<class A> void
OnlineSVR::serialize(A &ar, const unsigned version)
{
    // TODO Decide upon weight scaling factors serialization
    ar & last_trained_time;
    // TODO Solve dataset id saving
    ar & column_name;
    ar & tmp_ixs;
    ar & samples_trained;
    ar & p_features;
    ar & p_labels;
    ar & p_kernel_matrices;
    ar & p_input_weights;
    ar & train_label_chunks;
    ar & train_feature_chunks_t;
    ar & weight_chunks;
    ar & chunks_score;
    ar & ixs;
    ar & multiout;
    ar & max_chunk_size;
    ar & gradient;
    ar & level;
    ar & step;
    ar & projection;
}

template<typename S> void
OnlineSVR::save(const OnlineSVR &osvr, S &output_stream)
{
    boost::archive::binary_oarchive oa(output_stream);
    oa << osvr;
}

template<typename S> OnlineSVR_ptr
OnlineSVR::load(S &input_stream)
{
    return ptr<OnlineSVR>(0, 0, input_stream);
}

} // datamodel
} // svr
