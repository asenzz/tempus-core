#pragma once

#[[#include]]# "config/common.hpp"
#[[#include]]# "DAO/${NAME}DAO.hpp"

using svr::datamodel::${NAME};

namespace svr {
namespace business {

class ${NAME}Service {

	${NAME}DAO_ptr ${NAME}Dao;

public:

	${NAME}Service(const ${NAME}DAO_ptr& ${NAME}Dao):${NAME}Dao(${NAME}Dao){}

	${NAME}_ptr     get_${NAME}_by_id(const bigint id);
	int             save(const ${NAME}_ptr&);
	bool            exists(const ${NAME}_ptr&);
	int             remove(const ${NAME}_ptr&);
};

} /* namespace business */
} /* namespace svr */

using ${NAME}Service_ptr = std::shared_ptr<svr::business::${NAME}Service>;