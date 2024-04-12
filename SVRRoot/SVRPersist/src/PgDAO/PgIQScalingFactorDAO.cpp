#include "PgIQScalingFactorDAO.hpp"
#include <DAO/DataSource.hpp>
#include <DAO/IQScalingFactorRowMapper.hpp>

namespace svr {
namespace dao {

PgIQScalingFactorDAO::PgIQScalingFactorDAO(common::PropertiesFileReader &tempus_config, DataSource &data_source) :
        IQScalingFactorDAO(tempus_config, data_source)
{}

bigint PgIQScalingFactorDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgIQScalingFactorDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), id) == 1;
}

int PgIQScalingFactorDAO::save(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    if (!p_iq_scaling_factor->get_id()) {
        p_iq_scaling_factor->set_id(get_next_id());
        return data_source.update(AbstractDAO::get_sql("save"),
                                  p_iq_scaling_factor->get_id(),
                                  p_iq_scaling_factor->get_dataset_id(),
                                  p_iq_scaling_factor->get_input_queue_table_name(),
                                  p_iq_scaling_factor->get_input_queue_column_name(),
                                  p_iq_scaling_factor->get_scaling_factor(),
                                  p_iq_scaling_factor->get_dc_offset());
    } else {
        return data_source.update(AbstractDAO::get_sql("update"),
                                  p_iq_scaling_factor->get_id(),
                                  p_iq_scaling_factor->get_dataset_id(),
                                  p_iq_scaling_factor->get_input_queue_table_name(),
                                  p_iq_scaling_factor->get_input_queue_column_name(),
                                  p_iq_scaling_factor->get_scaling_factor(),
                                  p_iq_scaling_factor->get_dc_offset());
    }
}

int PgIQScalingFactorDAO::remove(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    if (!p_iq_scaling_factor->get_id())  return 0;
    return data_source.update(AbstractDAO::get_sql("remove"), p_iq_scaling_factor->get_id());
}

std::deque<datamodel::IQScalingFactor_ptr> PgIQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    IQScalingFactorRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, AbstractDAO::get_sql("find_all_by_model_id"), dataset_id);
}

}
}
