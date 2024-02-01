#include "PgEnsembleDAO.hpp"
#include <DAO/DataSource.hpp>
#include <DAO/EnsembleRowMapper.hpp>

namespace svr { namespace dao {

PgEnsembleDAO::PgEnsembleDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:EnsembleDAO(sqlProperties, dataSource)
{}

datamodel::Ensemble_ptr PgEnsembleDAO::get_by_id(const bigint id) {
    EnsembleRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_id"), id);
}

bigint PgEnsembleDAO::get_next_id() {
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bool PgEnsembleDAO::exists(const bigint ensembleId) {
    return data_source.query_for_type<int>(get_sql("exists_by_id"), ensembleId) == 1;
}

bool PgEnsembleDAO::exists(const datamodel::Ensemble_ptr &ensemble) {
    if(ensemble.get() == nullptr || ensemble->get_id() == 0){
        return false;
    }
    return exists(ensemble->get_id());
}

int PgEnsembleDAO::save(const datamodel::Ensemble_ptr &ensemble) {
    if(!exists(ensemble))
    {
        if (ensemble->get_decon_queue() == nullptr) {
            return data_source.update(get_sql("save"),
                    ensemble->get_id(),
                    ensemble->get_dataset_id(),
                    nullptr,
                    ensemble->get_aux_decon_table_names()
            );
        }
        return data_source.update(get_sql("save"),
                ensemble->get_id(),
                ensemble->get_dataset_id(),
                ensemble->get_decon_queue()->get_table_name(),
                ensemble->get_aux_decon_table_names()
        );
    } else {
        return data_source.update(get_sql("update"),
                                  ensemble->get_dataset_id(),
                                  ensemble->get_decon_queue() != nullptr
                                      ? ensemble->get_decon_queue()->get_table_name()
                                      : nullptr,
                                  ensemble->get_aux_decon_table_names(),
                                  ensemble->get_id()
        );
    }
}

int PgEnsembleDAO::remove(const datamodel::Ensemble_ptr &ensemble) {
    return data_source.update(get_sql("remove"), ensemble->get_id());
}

datamodel::Ensemble_ptr PgEnsembleDAO::get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset,
                                                         const datamodel::DeconQueue_ptr &decon_queue) {
    EnsembleRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_dataset_and_decon_queue"), dataset->get_id(), decon_queue->get_table_name());
}

std::deque<datamodel::Ensemble_ptr> PgEnsembleDAO::find_all_ensembles_by_dataset_id(const bigint dataset_id)
{
    EnsembleRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, get_sql("find_all_ensembles_by_dataset"), dataset_id);
}

} }