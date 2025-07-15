#include "PgWScalingFactorDAO.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/WScalingFactorRowMapper.hpp"

namespace svr {
namespace dao {

PgWScalingFactorDAO::PgWScalingFactorDAO(common::PropertiesReader &tempus_config, DataSource &data_source) :
        WScalingFactorDAO(tempus_config, data_source)
{}

bigint PgWScalingFactorDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgWScalingFactorDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), id) == 1;
}

int PgWScalingFactorDAO::save(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    if (!p_w_scaling_factor->get_id()) {
        p_w_scaling_factor->set_id(get_next_id());
        return data_source.update(AbstractDAO::get_sql("save"),
                                  p_w_scaling_factor->get_id(),
                                  p_w_scaling_factor->get_dataset_id(),
                                  p_w_scaling_factor->get_step(),
                                  p_w_scaling_factor->get_scaling_factor(),
                                  p_w_scaling_factor->get_dc_offset());
    } else {
        return data_source.update(AbstractDAO::get_sql("update"),
                                  p_w_scaling_factor->get_id(),
                                  p_w_scaling_factor->get_dataset_id(),
                                  p_w_scaling_factor->get_step(),
                                  p_w_scaling_factor->get_scaling_factor(),
                                  p_w_scaling_factor->get_dc_offset());
    }
}

int PgWScalingFactorDAO::remove(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    if (!p_w_scaling_factor->get_id())  return 0;
    return data_source.update(AbstractDAO::get_sql("remove"), p_w_scaling_factor->get_id());
}

std::deque<datamodel::WScalingFactor_ptr> PgWScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    WScalingFactorRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, AbstractDAO::get_sql("find_all_by_model_id"), dataset_id);
}

}
}
