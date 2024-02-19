#include "PgPredictionTaskDAO.hpp"

#include <DAO/DataSource.hpp>
#include <DAO/PredictionTaskRowMapper.hpp>

namespace svr {
namespace dao {

PgPredictionTaskDAO::PgPredictionTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: PredictionTaskDAO(tempus_config, data_source)
{}


bigint PgPredictionTaskDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgPredictionTaskDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"),
                                           id) == 1;
}

int PgPredictionTaskDAO::save(const PredictionTask_ptr &predictionTask)
{
    if(!predictionTask->get_id())
    {
        predictionTask->set_id(get_next_id());

        return data_source.update(AbstractDAO::get_sql("save"),
                                  predictionTask->get_id(),
                                  predictionTask->get_dataset_id(),
                                  predictionTask->get_start_train_time(),
                                  predictionTask->get_end_train_time(),
                                  predictionTask->get_start_prediction_time(),
                                  predictionTask->get_end_prediction_time(),
                                  predictionTask->get_status(),
                                  predictionTask->get_mse());
    }
    return data_source.update(AbstractDAO::get_sql("update"),
                              predictionTask->get_dataset_id(),
                              predictionTask->get_start_train_time(),
                              predictionTask->get_end_train_time(),
                              predictionTask->get_start_prediction_time(),
                              predictionTask->get_end_prediction_time(),
                              predictionTask->get_status(),
                              predictionTask->get_mse(),
                              predictionTask->get_id());
}

int PgPredictionTaskDAO::remove(const PredictionTask_ptr &predictionTask)
{
    if (predictionTask->get_id() == 0) {
        return 0;
    }
    return data_source.update(AbstractDAO::get_sql("remove"), predictionTask->get_id());
}

PredictionTask_ptr PgPredictionTaskDAO::get_by_id(const bigint id)
{
    PredictionTaskRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, AbstractDAO::get_sql("get_by_id"), id);
}

} /* namespace dao */
} /* namespace svr */
