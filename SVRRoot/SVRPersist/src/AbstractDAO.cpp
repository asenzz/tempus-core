#include "DAO/AbstractDAO.hpp"
#include "DAO/DataSource.hpp"


using namespace svr::common;

namespace svr {
namespace dao {

AbstractDAO::AbstractDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, std::string propsFileName)
:sql_properties(tempus_config), propsFileName(std::move(propsFileName)), data_source(data_source)
{
}

AbstractDAO::~AbstractDAO() {}

std::string AbstractDAO::get_sql(const std::string& sqlKey) {
    return sql_properties.get_property<std::string>(propsFileName, sqlKey);
}

} /* namespace dao */
} /* namespace svr */
