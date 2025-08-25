#include <DAO/UserRowMapper.hpp>
#include <util/string_utils.hpp>
#include <common/logging.hpp>

namespace svr {
namespace dao {

datamodel::User_ptr UserRowMapper::map_row(const pqxx_tuple& row_set) const
{
    return ptr<datamodel::User>(
            row_set["user_id"].as<bigint>(0),
            row_set["username"].as<std::string>(),
            row_set["email"].as<std::string>(),
            row_set["password"].as<std::string>(),
            row_set["name"].as<std::string>(),
            common::ignore_case_equals(row_set["role"].as<std::string>(""), "admin") ? datamodel::ROLE::ADMIN : datamodel::ROLE::USER, 
            datamodel::Priority(row_set["priority"].as<int>((int)datamodel::Priority::Normal)));
}

datamodel::User_ptr UserRowMapper::map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const
{
    return ptr<datamodel::User>(
            common::dd_get_value(row_set, row, col_count, "user_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "username", std::string()),
            common::dd_get_value(row_set, row, col_count, "email", std::string()),
            common::dd_get_value(row_set, row, col_count, "password", std::string()),
            common::dd_get_value(row_set, row, col_count, "name", std::string()),
            common::ignore_case_equals("admin", common::dd_get_value(row_set, row, col_count, "admin", std::string())) ? datamodel::ROLE::ADMIN : datamodel::ROLE::USER, 
			datamodel::Priority(common::dd_get_value(row_set, row, col_count, "priority", int(datamodel::Priority::Normal))));
}

} /* namespace dao */
} /* namespace svr */

