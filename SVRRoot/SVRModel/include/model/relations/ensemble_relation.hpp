#ifndef ENSEMBLE_RELATION_HPP
#define ENSEMBLE_RELATION_HPP

#include "relation.hpp"

namespace svr {
namespace datamodel {

class Ensemble;

struct ensemble_storage_adapter
{
    static bigint get_next_id();
    static std::shared_ptr<Ensemble> load(bigint id);
};

typedef relation<Ensemble, ensemble_storage_adapter> ensemble_relation;

}}


#endif /* ENSEMBLE_RELATION_HPP */
