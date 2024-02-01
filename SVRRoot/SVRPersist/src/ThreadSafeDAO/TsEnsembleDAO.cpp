#include "TsEnsembleDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsEnsembleDAO, EnsembleDAO)
{}


bigint TsEnsembleDAO::get_next_id()
{
    return ts_call<bigint>(&EnsembleDAO::get_next_id);
}


datamodel::Ensemble_ptr TsEnsembleDAO::get_by_id(const bigint id)
{
    return ts_call<datamodel::Ensemble_ptr>(&EnsembleDAO::get_by_id, id);
}


bool TsEnsembleDAO::exists(const bigint ensemble_id)
{
    std::scoped_lock<std::recursive_mutex> scope_guar(mutex);
    return dao->exists(ensemble_id);
}


bool TsEnsembleDAO::exists(const datamodel::Ensemble_ptr &ensemble)
{
    std::scoped_lock<std::recursive_mutex> scope_guar(mutex);
    return dao->exists(ensemble);
}


datamodel::Ensemble_ptr TsEnsembleDAO::get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset, const datamodel::DeconQueue_ptr& decon_queue)
{
    return ts_call<datamodel::Ensemble_ptr>(&EnsembleDAO::get_by_dataset_and_decon_queue, dataset, decon_queue);
}


std::deque<datamodel::Ensemble_ptr> TsEnsembleDAO::find_all_ensembles_by_dataset_id(const bigint dataset_id)
{
    return ts_call<std::deque<datamodel::Ensemble_ptr>>(&EnsembleDAO::find_all_ensembles_by_dataset_id, dataset_id);
}


int TsEnsembleDAO::save(const datamodel::Ensemble_ptr &ensemble)
{
    return ts_call<int>(&EnsembleDAO::save, ensemble);
}


int TsEnsembleDAO::remove(const datamodel::Ensemble_ptr &ensemble)
{
    return ts_call<int>(&EnsembleDAO::remove, ensemble);
}


}}
