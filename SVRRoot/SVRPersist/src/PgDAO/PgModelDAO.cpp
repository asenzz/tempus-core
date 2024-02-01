#include "PgModelDAO.hpp"
#include <DAO/ModelRowMapper.hpp>
#include <DAO/DataSource.hpp>
#include <appcontext.hpp>

namespace svr {
namespace dao {

PgModelDAO::PgModelDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: ModelDAO(sqlProperties, dataSource)
{}

bigint PgModelDAO::get_next_id() {
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bool PgModelDAO::exists(const bigint model_id) {
    return data_source.query_for_type<int>(get_sql("exists_by_id"), model_id) == 1;
}

int PgModelDAO::save(const datamodel::Model_ptr &model) {
    if(!exists(model->get_id()))
        return data_source.update(
                get_sql("save"),
                model->get_id(),
                model->get_ensemble_id(),
                model->get_decon_level(),
                model->get_learning_levels(),
                model->get_gradients(),
                model->get_last_modified(),
                model->get_last_modeled_value_time()
        );
    else
        return data_source.update(
                get_sql("update"),
                model->get_learning_levels(),
                model->get_gradients(),
                model->get_last_modified(),
                model->get_last_modeled_value_time(),
                model->get_id()
        );
}

int PgModelDAO::remove(const datamodel::Model_ptr &model) {
    return data_source.update(get_sql("remove"), model->get_id());
}

int PgModelDAO::remove_by_ensemble_id(const bigint ensemble_id)
{
    return data_source.update(get_sql("remove_by_ensemble_id"), ensemble_id);
}

datamodel::Model_ptr PgModelDAO::get_by_id(const bigint model_id) {
    ModelRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_id"), model_id);
}

datamodel::Model_ptr PgModelDAO::get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level)
{
    ModelRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_ensemble_id_and_decon_level"), ensemble_id, decon_level);
}

std::deque<datamodel::Model_ptr> PgModelDAO::get_all_ensemble_models(const bigint ensemble_id)
{
    ModelRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, get_sql(/* "get_all_ensemble_models_empty" */ "get_all_ensemble_models"), ensemble_id);
}

} // namespace dao
} // namespace svr
