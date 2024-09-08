#include "PgDatasetDAO.hpp"
#include <DAO/DatasetRowMapper.hpp>
#include <DAO/DataSource.hpp>

namespace svr {
namespace dao {

PgDatasetDAO::PgDatasetDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
        : DatasetDAO(tempus_config, data_source)
{}

bool PgDatasetDAO::exists(const bigint dataset_id)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_id"), dataset_id) == 1;
}

bool PgDatasetDAO::exists(const std::string &user_name, const std::string &dataset_name)
{
    return data_source.query_for_type<int>(AbstractDAO::get_sql("exists_by_user_name_and_dataset_name"),
                                           user_name, dataset_name) == 1;
}

int PgDatasetDAO::save(const datamodel::Dataset_ptr &dataset)
{
    if (dataset->get_id() == 0)
        dataset->set_id(get_next_id());

    if (!exists(dataset->get_id()))
        return data_source.update(
                AbstractDAO::get_sql("save"),
                dataset->get_id(),
                dataset->get_dataset_name(),
                dataset->get_user_name(),
                dataset->get_input_queue()->get_table_name(),
                dataset->get_aux_input_table_names(),
                dataset->get_priority(),
                dataset->get_description(),
                dataset->get_gradient_count(),
                dataset->get_max_chunk_size(),
                dataset->get_multistep(),
                dataset->get_spectral_levels(),
                dataset->get_transformation_name(),
                dataset->get_max_lookback_time_gap(),
                dataset->get_is_active()
        );
    else
        return data_source.update(
                AbstractDAO::get_sql("update"),
                dataset->get_dataset_name(),
                dataset->get_user_name(),
                dataset->get_input_queue()->get_table_name(),
                dataset->get_aux_input_table_names(),
                dataset->get_priority(),
                dataset->get_description(),
                dataset->get_gradient_count(),
                dataset->get_max_chunk_size(),
                dataset->get_multistep(),
                dataset->get_spectral_levels(),
                dataset->get_transformation_name(),
                dataset->get_max_lookback_time_gap(),
                dataset->get_is_active(),
                dataset->get_id()
        );
}

int PgDatasetDAO::remove(const datamodel::Dataset_ptr &dataset)
{

    if (dataset->get_id() == 0) return 0;
    return data_source.update(AbstractDAO::get_sql("remove"), dataset->get_id());
}

datamodel::Dataset_ptr PgDatasetDAO::get_by_id(const bigint dataset_id)
{
    DatasetRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, AbstractDAO::get_sql("get_by_id"), dataset_id);
}

datamodel::Dataset_ptr PgDatasetDAO::get_by_name(const std::string &user_name, const std::string &dataset_name)
{
    DatasetRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_name"), dataset_name, user_name);
}

bigint PgDatasetDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(AbstractDAO::get_sql("get_next_id"));
}

std::deque<datamodel::Dataset_ptr> PgDatasetDAO::find_all_user_datasets(const std::string &user_name)
{
    DatasetRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, AbstractDAO::get_sql("find_all_user_datasets"), user_name);
}

bool PgDatasetDAO::link_user_to_dataset(const std::string &user_name, const datamodel::Dataset_ptr &dataset)
{
    return data_source.update(AbstractDAO::get_sql("link_user_to_dataset"), user_name, dataset->get_id()) == 1;
}

bool PgDatasetDAO::unlink_user_from_dataset(const std::string &user_name, const datamodel::Dataset_ptr &dataset)
{
    return data_source.update(AbstractDAO::get_sql("unlink_user_from_dataset"), user_name, dataset->get_id()) == 1;
}

// TODO Return a set of bigints
PgDatasetDAO::UserDatasetPairs PgDatasetDAO::get_active_datasets()
{
    UserDatasetRowMapper rowMapper;

    PgDatasetDAO::UserDatasetPairs result;

    for (auto p: data_source.query_for_array(rowMapper, AbstractDAO::get_sql("get_active_datasets")))
        result.emplace_back(p->first, p->second);

    return result;
}

size_t PgDatasetDAO::get_level_count(const bigint dataset_id)
{
    return data_source.query_for_type<size_t>(AbstractDAO::get_sql("get_level_count"), dataset_id);
}

}
}
