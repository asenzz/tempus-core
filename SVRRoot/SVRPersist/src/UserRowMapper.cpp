#include <DAO/UserRowMapper.hpp>
#include <util/string_utils.hpp>
#include <common/logging.hpp>

namespace svr {
namespace dao {

User_ptr UserRowMapper::map_row(const pqxx_tuple& row_set) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	if (row_set.empty()){
		LOG4_ERROR("No row returned from database!");
		return nullptr;
	}
#pragma GCC diagnostic pop
	datamodel::ROLE role;
	datamodel::Priority priority = static_cast<datamodel::Priority>(row_set["priority"].as<int>((int)datamodel::Priority::Normal));

	if(common::ignore_case_equals(row_set["role"].as<std::string>(""), "admin"))
		role = datamodel::ROLE::ADMIN;
	else
	    role = datamodel::ROLE::USER;

    return ptr<datamodel::User>(
            row_set["user_id"].as<bigint>(0),
            row_set["username"].as<std::string>(),
            row_set["email"].as<std::string>(),
            row_set["password"].as<std::string>(),
            row_set["name"].as<std::string>(),
            role, priority);
}

} /* namespace dao */
} /* namespace svr */

