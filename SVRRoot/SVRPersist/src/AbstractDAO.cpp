#include "DAO/AbstractDAO.hpp"
#include "DAO/DataSource.hpp"


using namespace svr::common;

namespace svr {
namespace dao {

AbstractDAO::AbstractDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, std::string propsFileName)
:sql_properties(sqlProperties), propsFileName(std::move(propsFileName)), data_source(dataSource)
{
}

AbstractDAO::~AbstractDAO() {}

std::string AbstractDAO::get_sql(const std::string& sqlKey) {
    return sql_properties.get_property<std::string>(propsFileName, sqlKey);
}

} /* namespace dao */
} /* namespace svr */
