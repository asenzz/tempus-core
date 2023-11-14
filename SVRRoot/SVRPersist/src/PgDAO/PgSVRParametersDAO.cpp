#include "PgSVRParametersDAO.hpp"

#include <DAO/SVRParametersRowMapper.hpp>
#include <DAO/DataSource.hpp>
#include <model/SVRParameters.hpp>

namespace svr { namespace dao {

PgSVRParametersDAO::PgSVRParametersDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: SVRParametersDAO(sqlProperties, dataSource)
{}

bigint PgSVRParametersDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bool PgSVRParametersDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(get_sql("exists_by_id"), id);
}

int PgSVRParametersDAO::save(const SVRParameters_ptr &svr_parameters)
{
    if (svr_parameters->get_id() == 0)
        svr_parameters->set_id(get_next_id());

    if(!exists(svr_parameters->get_id()))
        return data_source.update(get_sql("save"),
                              svr_parameters->get_id(),
                              svr_parameters->get_dataset_id(),
                              svr_parameters->get_input_queue_table_name(),
                              svr_parameters->get_input_queue_column_name(),
                              svr_parameters->get_decon_level(),
                              svr_parameters->get_chunk_ix(),
                              svr_parameters->get_grad_level(),
                              svr_parameters->get_svr_C(),
                              svr_parameters->get_svr_epsilon(),
                              svr_parameters->get_svr_kernel_param(),
                              svr_parameters->get_svr_kernel_param2(),
                              svr_parameters->get_svr_decremental_distance(),
                              svr_parameters->get_svr_adjacent_levels_ratio(),
                              static_cast<int>(svr_parameters->get_kernel_type()),
                              svr_parameters->get_lag_count());
    else
        return data_source.update(get_sql("update"),
                              svr_parameters->get_dataset_id(),
                              svr_parameters->get_input_queue_table_name(),
                              svr_parameters->get_input_queue_column_name(),
                              svr_parameters->get_decon_level(),
                              svr_parameters->get_chunk_ix(),
                              svr_parameters->get_grad_level(),
                              svr_parameters->get_svr_C(),
                              svr_parameters->get_svr_epsilon(),
                              svr_parameters->get_svr_kernel_param(),
                              svr_parameters->get_svr_kernel_param2(),
                              svr_parameters->get_svr_decremental_distance(),
                              svr_parameters->get_svr_adjacent_levels_ratio(),
                              static_cast<int>(svr_parameters->get_kernel_type()),
                              svr_parameters->get_lag_count(),
                              svr_parameters->get_id());
}

int PgSVRParametersDAO::remove(const SVRParameters_ptr& svr_parameters)
{
    return data_source.update(get_sql("remove_by_unique"), svr_parameters->get_dataset_id(), svr_parameters->get_input_queue_table_name(), svr_parameters->get_input_queue_column_name(), svr_parameters->get_decon_level(), svr_parameters->get_chunk_ix(), svr_parameters->get_grad_level());
}

int PgSVRParametersDAO::remove_by_dataset_id(const bigint dataset_id)
{
    return data_source.update(get_sql("remove_by_dataset_id"), dataset_id);
}
std::vector<SVRParameters_ptr> PgSVRParametersDAO::get_all_svrparams_by_dataset_id(const bigint dataset_id)
{
    SVRParametersRowMapper row_mapper;
    return data_source.query_for_array(row_mapper, get_sql("get_all_by_dataset_id"), dataset_id);
}

size_t PgSVRParametersDAO::get_dataset_levels(const bigint dataset_id)
{
    return data_source.query_for_type<int>(get_sql("dataset_levels"), dataset_id);
}

} }
