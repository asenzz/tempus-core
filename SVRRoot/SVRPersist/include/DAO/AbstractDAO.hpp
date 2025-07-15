#pragma once

#include "common/types.hpp"
#include "util/PropertiesFileReader.hpp"


namespace svr {
namespace dao {

class DataSource;

class AbstractDAO {
    common::PropertiesReader &sql_properties;
    const std::string properties_file_name;

protected:
    DataSource &data_source;

    virtual ~AbstractDAO();

    explicit AbstractDAO(common::PropertiesReader &sql_properties, DataSource &data_source, const std::string &properties_file_name);

    template<class BasicDao, class PostgresDao, class AsyncDao, class ThreadSafeDao>
    static BasicDao *build(common::PropertiesReader &sql_properties, DataSource &data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao);

public:
    DataSource &get_data_source();

    virtual std::string get_sql(const std::string &sql_key);
};


template<class BasicDao, class PostgresDao, class AsyncDao, class ThreadSafeDao> BasicDao *
AbstractDAO::build(
        common::PropertiesReader &sql_properties,
        DataSource &data_source,
        const common::ConcreteDaoType dao_type,
        const bool use_threadsafe_dao)
{
    const auto basic_dao = dao_type == common::ConcreteDaoType::AsyncDao ? (BasicDao *) new AsyncDao(sql_properties, data_source) : (BasicDao *) new PostgresDao(sql_properties, data_source);
    return use_threadsafe_dao ? new ThreadSafeDao(sql_properties, data_source, std::unique_ptr<BasicDao>(basic_dao)) : basic_dao;
}

using AbstractDAO_ptr = std::shared_ptr<dao::AbstractDAO>;

} /* namespace dao */
} /* namespace svr */