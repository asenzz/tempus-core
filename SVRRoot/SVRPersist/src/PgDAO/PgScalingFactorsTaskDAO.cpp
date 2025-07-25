#include "DAO/DataSource.hpp"
#include "DAO/ScalingFactorsTaskRowMapper.hpp"
#include "PgScalingFactorsTaskDAO.hpp"

namespace svr {
namespace dao {

PgScalingFactorsTaskDAO::PgScalingFactorsTaskDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : ScalingFactorsTaskDAO(tempus_config, data_source)
{}


bigint PgScalingFactorsTaskDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgScalingFactorsTaskDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), id) == 1;
}

int PgScalingFactorsTaskDAO::save(const ScalingFactorsTask_ptr &scalingFactorsTask)
{
    if (!scalingFactorsTask->get_id()) {
        scalingFactorsTask->set_id(get_next_id());

        return data_source.update(AbstractDAO::get_sql("save"),
                                  scalingFactorsTask->get_id(),
                                  scalingFactorsTask->get_dataset_id(),
                                  scalingFactorsTask->get_status(),
                                  scalingFactorsTask->get_mse());
    }
    return data_source.update(AbstractDAO::get_sql("update"),
                              scalingFactorsTask->get_dataset_id(),
                              scalingFactorsTask->get_status(),
                              scalingFactorsTask->get_mse(),
                              scalingFactorsTask->get_id());
}

int PgScalingFactorsTaskDAO::remove(const ScalingFactorsTask_ptr &scalingFactorsTask)
{
    if (scalingFactorsTask->get_id() == 0) return 0;
    return data_source.update(AbstractDAO::get_sql("remove"), scalingFactorsTask->get_id());
}

ScalingFactorsTask_ptr PgScalingFactorsTaskDAO::get_by_id(const bigint id)
{
    ScalingFactorsTaskRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, AbstractDAO::get_sql("get_by_id"), id);
}

} /* namespace dao */
} /* namespace svr */
