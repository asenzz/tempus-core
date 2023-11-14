#pragma once

#[[#include]]# "DAO/IRowMapper.hpp"
#[[#include]]# "model/${NAME}.hpp"

namespace svr{
namespace dao{

class ${NAME}RowMapper : public IRowMapper<svr::datamodel::${NAME}>{
public:
    ${NAME}_ptr mapRow(const pqxx::tuple& rowSet) const override {
        return std::make_shared<svr::datamodel::${NAME}>();
    }
};
}
}