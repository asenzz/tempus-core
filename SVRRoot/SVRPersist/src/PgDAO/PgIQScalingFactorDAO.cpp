#include "PgIQScalingFactorDAO.hpp"
#include <DAO/DataSource.hpp>
#include <DAO/IQScalingFactorRowMapper.hpp>

namespace svr {
namespace dao {

PgIQScalingFactorDAO::PgIQScalingFactorDAO(common::PropertiesFileReader &sqlProperties,
                                           DataSource &dataSource) :
    IQScalingFactorDAO(sqlProperties, dataSource)
{}

bigint PgIQScalingFactorDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

bool PgIQScalingFactorDAO::exists(const bigint id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), id) == 1;
}

int PgIQScalingFactorDAO::save(const IQScalingFactor_ptr& iQscalingFactor)
{
    if(!iQscalingFactor->get_id())
    {
        iQscalingFactor->set_id(get_next_id());

        return data_source.update(AbstractDAO::get_sql("save"),
                                  iQscalingFactor->get_id(),
                                  iQscalingFactor->get_dataset_id(),
                                  iQscalingFactor->get_input_queue_table_name(),
                                  iQscalingFactor->get_scaling_factor()
                                  );
    }
    else
    {
        return data_source.update(AbstractDAO::get_sql("update"),
                                  iQscalingFactor->get_id(),
                                  iQscalingFactor->get_dataset_id(),
                                  iQscalingFactor->get_input_queue_table_name(),
                                  iQscalingFactor->get_scaling_factor()
                                  );
    }
}

int PgIQScalingFactorDAO::remove(const IQScalingFactor_ptr& iQscalingFactor)
{
    if(iQscalingFactor->get_id() == 0)
        return 0;

    return data_source.update(AbstractDAO::get_sql("remove"), iQscalingFactor->get_id());
}

std::deque<IQScalingFactor_ptr> PgIQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    IQScalingFactorRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, AbstractDAO::get_sql("find_all_by_dataset_id"), dataset_id);
}

}
}
