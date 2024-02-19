#include "PgDQScalingFactorDAO.hpp"
#include <DAO/DataSource.hpp>
#include <DAO/DQScalingFactorRowMapper.hpp>

namespace svr {
namespace dao {

PgDQScalingFactorDAO::PgDQScalingFactorDAO(svr::common::PropertiesFileReader& tempus_config,
                                           svr::dao::DataSource& data_source) :
    DQScalingFactorDAO(tempus_config, data_source)
{}

bigint PgDQScalingFactorDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgDQScalingFactorDAO::exists(const datamodel::DQScalingFactor_ptr& p_dq_scaling_factor)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_pk")
            , p_dq_scaling_factor->get_dataset_id()
            , p_dq_scaling_factor->get_input_queue_table_name()
            , p_dq_scaling_factor->get_input_queue_column_name()
            , p_dq_scaling_factor->get_decon_level()
        ) == 1;
}

int PgDQScalingFactorDAO::save(const datamodel::DQScalingFactor_ptr& p_dq_scaling_factor)
{
    return data_source.update(
            AbstractDAO::get_sql("save"),
            p_dq_scaling_factor->get_id(),
            p_dq_scaling_factor->get_dataset_id(),
            p_dq_scaling_factor->get_input_queue_table_name(),
            p_dq_scaling_factor->get_input_queue_column_name(),
            p_dq_scaling_factor->get_decon_level(),
            p_dq_scaling_factor->get_features_factor(),
            p_dq_scaling_factor->get_labels_factor(),
            p_dq_scaling_factor->get_dc_offset_features(),
            p_dq_scaling_factor->get_dc_offset_labels());
}

int PgDQScalingFactorDAO::remove(const datamodel::DQScalingFactor_ptr& p_dqscaling_factor)
{
    if(p_dqscaling_factor->get_id() == 0) return 0;

    return data_source.update(AbstractDAO::get_sql("remove"), p_dqscaling_factor->get_id());
}

svr::datamodel::dq_scaling_factor_container_t PgDQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    DQScalingFactorRowMapper row_mapper;
    auto sfs = data_source.query_for_array(row_mapper, AbstractDAO::get_sql("find_all_by_dataset_id"), dataset_id);

    svr::datamodel::dq_scaling_factor_container_t result;
    for (auto &sf: sfs) result.insert(sf);
    return result;
}

}
}
