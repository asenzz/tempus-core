//
// Created by zarko on 2/12/24.
//

#include "common/serialization.hpp"
#include "appcontext.hpp"
#include "onlinesvr_persist.tpp"

namespace svr {
namespace datamodel {

bigint OnlineSVR::get_model_id() const
{
    return model_id;
}

OnlineSVR::OnlineSVR(const bigint id, const bigint model_id, std::stringstream &input_stream) : Entity(id), model_id(model_id), chunk_offlap(1 - PROPS.get_chunk_overlap())
{
    boost::archive::binary_iarchive ia(input_stream);
    ia >> *this;
}

std::stringstream OnlineSVR::save() const
{
    std::stringstream res;
    save(*this, res);
    return res;
}

void OnlineSVR::set_model_id(const size_t new_model_id)
{
    model_id = new_model_id;
}


void OnlineSVR::init_id()
{
    if (!id) {
        boost::hash_combine(id, column_name);
        boost::hash_combine(id, level);
        boost::hash_combine(id, gradient);
        SVRParameters_ptr p_head_params;
        if ((p_head_params = get_params_ptr())) {
            boost::hash_combine(id, p_head_params->get_input_queue_table_name());
            boost::hash_combine(id, p_head_params->get_input_queue_column_name());
        }
    }
}


std::string OnlineSVR::to_string() const
{
    std::stringstream s;
    s << " Samples trained " << samples_trained <<
      ", features " << p_features->rows(0, 10) <<
      ", labels " << p_labels->rows(0, 10) <<
      ", kernel matrices " << (p_kernel_matrices ? p_kernel_matrices->size() : 0) <<
      ", indexes " << ixs.size() <<
      ", multiout " << multiout <<
      ", max chunk size " << max_chunk_size <<
      ", gradient level " << gradient <<
      ", decon level " << level;
    return s.str();
}

} // datamodel
} // svr