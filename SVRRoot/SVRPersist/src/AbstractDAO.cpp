#include "DAO/AbstractDAO.hpp"
#include "DAO/DataSource.hpp"


using namespace svr::common;

namespace svr {
namespace dao {

AbstractDAO::AbstractDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, std::string properties_file_name)
:sql_properties(tempus_config), properties_file_name(std::move(properties_file_name)), data_source(data_source)
{
}

AbstractDAO::~AbstractDAO() {}

std::string AbstractDAO::get_sql(const std::string& sql_key)
{
    return sql_properties.get_property<std::string>(properties_file_name, sql_key);
}

} /* namespace dao */
} /* namespace svr */
