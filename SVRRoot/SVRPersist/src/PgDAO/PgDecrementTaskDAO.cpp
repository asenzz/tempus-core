#include "PgDecrementTaskDAO.hpp"
#include <DAO/DataSource.hpp>
#include <DAO/DecrementTaskRowMapper.hpp>

namespace svr {
namespace dao {

PgDecrementTaskDAO::PgDecrementTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: DecrementTaskDAO(sqlProperties, dataSource)
{}

bigint PgDecrementTaskDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgDecrementTaskDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), id) == 1;
}

int PgDecrementTaskDAO::save(const DecrementTask_ptr& decrementTask)
{
    if(!decrementTask->get_id())
    {
        decrementTask->set_id(get_next_id());
        return data_source.update(AbstractDAO::get_sql("save"),
                                  decrementTask->get_id(),
                                  decrementTask->get_dataset_id(),
                                  decrementTask->get_start_task_time(),
                                  decrementTask->get_end_task_time(),
                                  decrementTask->get_start_train_time(),
                                  decrementTask->get_end_train_time(),
                                  decrementTask->get_start_validation_time(),
                                  decrementTask->get_end_validation_time(),
                                  decrementTask->get_parameters(),
                                  decrementTask->get_status(),
                                  decrementTask->get_decrement_step(),
                                  decrementTask->get_vp_sliding_direction(),
                                  decrementTask->get_vp_slide_count(),
                                  decrementTask->get_vp_slide_period_sec().total_seconds(),
                                  decrementTask->get_values(),
                                  decrementTask->get_suggested_value()
                                  );
    }
    return data_source.update(AbstractDAO::get_sql("update"),
                              decrementTask->get_dataset_id(),
                              decrementTask->get_start_task_time(),
                              decrementTask->get_end_task_time(),
                              decrementTask->get_start_train_time(),
                              decrementTask->get_end_train_time(),
                              decrementTask->get_start_validation_time(),
                              decrementTask->get_end_validation_time(),
                              decrementTask->get_parameters(),
                              decrementTask->get_status(),
                              decrementTask->get_decrement_step(),
                              decrementTask->get_vp_sliding_direction(),
                              decrementTask->get_vp_slide_count(),
                              decrementTask->get_vp_slide_period_sec().total_seconds(),
                              decrementTask->get_values(),
                              decrementTask->get_suggested_value(),
                              decrementTask->get_id()
                              );

}

int PgDecrementTaskDAO::remove(const DecrementTask_ptr& decrementTask)
{
    if(decrementTask->get_id() == 0)
        return 0;

    return data_source.update(AbstractDAO::get_sql("remove"), decrementTask->get_id());
}

DecrementTask_ptr PgDecrementTaskDAO::get_by_id(const bigint id)
{
    DecrementTaskRowMapper rowMaper;
    return data_source.query_for_object(&rowMaper, AbstractDAO::get_sql("get_by_id"), id);
}

}
}
