//
// Created by zarko on 16/11/2024.
//

#include <sstream>
#include "model/WScalingFactor.hpp"

namespace svr {
namespace datamodel {

WScalingFactor::WScalingFactor(const bigint id, const bigint dataset_id, const uint16_t step, const double scaling_factor, const double dc_offset) :
        Entity(id),
        dataset_id(dataset_id),
        step(step),
        scaling_factor(scaling_factor),
        dc_offset(dc_offset)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void WScalingFactor::init_id()
{
    if (!id) {
        boost::hash_combine(id, dataset_id);
        boost::hash_combine(id, step);
    }
}

bool WScalingFactor::operator==(const WScalingFactor &o) const
{
    return o.id == id &&
           o.dataset_id == dataset_id &&
           o.step == step &&
           o.scaling_factor == scaling_factor &&
           o.dc_offset == dc_offset;
}

std::string WScalingFactor::to_string() const
{
    std::stringstream s;
    s << "Scaling task id " << id <<
       ", dataset id " << dataset_id <<
       ", step " << step <<
       ", scaling factor " << scaling_factor <<
       ", dc offset " << dc_offset;
    return s.str();
}


} // namespace datamodel
} // namespace svr
