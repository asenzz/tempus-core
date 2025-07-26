#include "appcontext.hpp"
#include "InputQueueService.hpp"
#include "EnsembleService.hpp"
#include "model/relations/iq_relation.hpp"
#include "model/relations/ensemble_relation.hpp"

namespace svr {
namespace datamodel {


std::shared_ptr<InputQueue> iq_storage_adapter::load(std::string const & table_name)
{
    return APP.input_queue_service.get_queue_metadata(table_name);
}


bigint ensemble_storage_adapter::get_next_id()
{
    return APP.ensemble_service.get_next_id();
}


std::shared_ptr<Ensemble> ensemble_storage_adapter::load(const bigint id)
{
    return APP.ensemble_service.get(id);
}

}
}
