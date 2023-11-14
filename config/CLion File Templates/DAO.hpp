#pragma once

// external
#[[#include]]# "common.hpp"

#[[#include]]# "model/${NAME}.hpp"
// internal
#[[#include]]# "DAO/AbstractDAO.hpp"
#[[#include]]# "DAO/DataSource.hpp"

using svr::datamodel::${NAME};
using svr::common::MessageSource;

namespace svr {
namespace dao {

class ${NAME}DAO : public AbstractDAO {
private:
	const std::string PropertiesFile;
    virtual const std::string & get_properties_file_name() const override { return PropertiesFile; }

public:
	explicit ${NAME}DAO(MessageSource_ptr sqlProperties, DataSource_ptr dataSource):
		AbstractDAO(sqlProperties, dataSource), PropertiesFile("${NAME}DAO.properties"){}

    ${NAME}_ptr     get_by_id(bigint id);
	bigint          get_next_id();

	bool        exists(bigint ${NAME}Id);	
	bool        exists(const ${NAME}_ptr& ${NAME});

	int         save(const ${NAME}_ptr& ${NAME});
	int         update(const ${NAME}_ptr& ${NAME});
	int         remove(const ${NAME}_ptr& ${NAME});
};

}
}

using ${NAME}DAO_ptr = std::shared_ptr<svr::dao::${NAME}DAO>;
