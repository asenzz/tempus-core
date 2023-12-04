#include <DAO/UserRowMapper.hpp>
#include <util/string_utils.hpp>
#include <common/Logging.hpp>

using namespace svr::datamodel;
using namespace svr::common;

namespace svr {
namespace dao {

User_ptr UserRowMapper::mapRow(const pqxx_tuple& rowSet) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	if (rowSet.empty()){
		LOG4_ERROR("No row returned from database!");
		return nullptr;
	}
#pragma GCC diagnostic pop
	svr::datamodel::ROLE role;
	svr::datamodel::Priority priority = static_cast<svr::datamodel::Priority>(rowSet["priority"].as<int>((int)svr::datamodel::Priority::Normal));

	if(ignoreCaseEquals(rowSet["role"].as<std::string>(""), "admin"))
		role = svr::datamodel::ROLE::ADMIN;
	else
	    role = svr::datamodel::ROLE::USER;

	return std::make_shared<User>(
			rowSet["user_id"].as<bigint>(std::numeric_limits<bigint>::quiet_NaN()),
			rowSet["username"].as<std::string>(""),
			rowSet["email"].as<std::string>(""),
			rowSet["password"].as<std::string>(""),
			rowSet["name"].as<std::string>(""),
			role,
			priority
	);
}

} /* namespace dao */
} /* namespace svr */

