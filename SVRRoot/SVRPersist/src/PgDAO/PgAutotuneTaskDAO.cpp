#include "PgAutotuneTaskDAO.hpp"

#include <DAO/AutotuneTaskRowMapper.hpp>
#include <DAO/DataSource.hpp>

namespace svr {
namespace dao {

PgAutotuneTaskDAO::PgAutotuneTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AutotuneTaskDAO(sqlProperties, dataSource)
{}

bigint PgAutotuneTaskDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgAutotuneTaskDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"),
                                           id) == 1;
}

int PgAutotuneTaskDAO::save(const AutotuneTask_ptr &autotune_task)
{

    if(!exists(autotune_task->get_id()))
        return data_source.update(AbstractDAO::get_sql("save"),
                                  autotune_task->get_id(),
                                  autotune_task->get_dataset_id(),
                                  autotune_task->get_result_dataset_id(),
                                  autotune_task->get_creation_time(),
                                  autotune_task->get_done_time(),
                                  autotune_task->get_parameters_in_string(),
                                  autotune_task->get_start_train_time(),
                                  autotune_task->get_end_train_time(),
                                  autotune_task->get_start_validation_time(),
                                  autotune_task->get_end_validation_time(),
                                  autotune_task->get_vp_sliding_direction(),
                                  autotune_task->get_vp_slide_count(),
                                  autotune_task->get_vp_slide_period_sec().total_seconds(),
                                  autotune_task->get_pso_best_points_counter(),
                                  autotune_task->get_pso_iteration_number(),
                                  autotune_task->get_pso_particles_number(),
                                  autotune_task->get_pso_topology(),
                                  autotune_task->get_nm_max_iteration_number(),
                                  autotune_task->get_nm_tolerance(),
                                  autotune_task->get_status(),
                                  autotune_task->get_mse());

    return data_source.update(AbstractDAO::get_sql("update"),
                              autotune_task->get_dataset_id(),
                              autotune_task->get_result_dataset_id(),
                              autotune_task->get_creation_time(),
                              autotune_task->get_done_time(),
                              autotune_task->get_parameters_in_string(),
                              autotune_task->get_start_train_time(),
                              autotune_task->get_end_train_time(),
                              autotune_task->get_start_validation_time(),
                              autotune_task->get_end_validation_time(),
                              autotune_task->get_vp_sliding_direction(),
                              autotune_task->get_vp_slide_count(),
                              autotune_task->get_vp_slide_period_sec().total_seconds(),
                              autotune_task->get_pso_best_points_counter(),
                              autotune_task->get_pso_iteration_number(),
                              autotune_task->get_pso_particles_number(),
                              autotune_task->get_pso_topology(),
                              autotune_task->get_nm_max_iteration_number(),
                              autotune_task->get_nm_tolerance(),
                              autotune_task->get_status(),
                              autotune_task->get_mse(),
                              autotune_task->get_id());
}


int PgAutotuneTaskDAO::remove(const AutotuneTask_ptr &autotuneTask)
{
    if (autotuneTask->get_id() == 0) {
        return 0;
    }
    return data_source.update(AbstractDAO::get_sql("remove"), autotuneTask->get_id());
}

AutotuneTask_ptr PgAutotuneTaskDAO::get_by_id(const bigint id)
{
    AutotuneTaskRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, AbstractDAO::get_sql("get_by_id"), id);
}

std::vector<AutotuneTask_ptr> PgAutotuneTaskDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    AutotuneTaskRowMapper rowMapper;
    return data_source.query_for_array(rowMapper, AbstractDAO::get_sql("get_by_dataset_id"), dataset_id);
}

}
}
