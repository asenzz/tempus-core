#include "TsEnsembleDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsEnsembleDAO, EnsembleDAO)
{}


bigint TsEnsembleDAO::get_next_id()
{
    return ts_call<bigint>(&EnsembleDAO::get_next_id);
}


Ensemble_ptr TsEnsembleDAO::get_by_id(bigint id)
{
    return ts_call<Ensemble_ptr>(&EnsembleDAO::get_by_id, id);
}


bool TsEnsembleDAO::exists(bigint ensemble_id)
{
    std::scoped_lock<std::recursive_mutex> scope_guar(mutex);
    return dao->exists(ensemble_id);
}


bool TsEnsembleDAO::exists(const Ensemble_ptr &ensemble)
{
    std::scoped_lock<std::recursive_mutex> scope_guar(mutex);
    return dao->exists(ensemble);
}


Ensemble_ptr TsEnsembleDAO::get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue)
{
    return ts_call<Ensemble_ptr>(&EnsembleDAO::get_by_dataset_and_decon_queue, dataset, decon_queue);
}


std::vector<Ensemble_ptr> TsEnsembleDAO::find_all_ensembles_by_dataset_id(bigint dataset_id)
{
    return ts_call<std::vector<Ensemble_ptr>>(&EnsembleDAO::find_all_ensembles_by_dataset_id, dataset_id);
}


int TsEnsembleDAO::save(const Ensemble_ptr &ensemble)
{
    return ts_call<int>(&EnsembleDAO::save, ensemble);
}


int TsEnsembleDAO::remove(const Ensemble_ptr &ensemble)
{
    return ts_call<int>(&EnsembleDAO::remove, ensemble);
}


}}
