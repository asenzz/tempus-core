#include "DAO/AbstractDAO.hpp"
#include "DAO/DataSource.hpp"


namespace svr {
namespace dao {

AbstractDAO::AbstractDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source, const std::string &properties_file_name)
        : sql_properties(tempus_config), properties_file_name(properties_file_name), data_source(data_source)
{}

AbstractDAO::~AbstractDAO()
{}

std::string AbstractDAO::get_sql(const std::string &sql_key)
{
    return sql_properties.get_property<std::string>(properties_file_name, sql_key);
}

DataSource &AbstractDAO::get_data_source()
{
    return data_source;
}

} /* namespace dao */
} /* namespace svr */
