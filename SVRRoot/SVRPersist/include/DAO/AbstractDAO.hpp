#pragma once

#include <common/types.hpp>
#include <util/PropertiesFileReader.hpp>


namespace svr {
namespace dao {

class DataSource;

class AbstractDAO {
    svr::common::PropertiesFileReader& sql_properties;
    const std::string propsFileName;
protected:
    DataSource& data_source;

    virtual ~AbstractDAO();
    explicit AbstractDAO(svr::common::PropertiesFileReader& sql_properties, DataSource& data_source, std::string propsFileName);

    template<class BasicDao, class PostgresDao, class AsyncDao, class ThreadSafeDao>
    static BasicDao * build (svr::common::PropertiesFileReader& sql_properties, DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

public:
    DataSource& get_data_source(){ return data_source; }
    virtual std::string get_sql(const std::string& sqlKey);
};


template<class BasicDao, class PostgresDao, class AsyncDao, class ThreadSafeDao>
BasicDao * AbstractDAO::build (svr::common::PropertiesFileReader& sql_properties, DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    BasicDao * basic_dao = nullptr;

    if(dao_type == svr::common::ConcreteDaoType::AsyncDao)
        basic_dao = new AsyncDao(sql_properties, data_source);
    else
        basic_dao = new PostgresDao(sql_properties, data_source);


    if(use_threadsafe_dao)
        return new ThreadSafeDao(sql_properties, data_source, std::unique_ptr<BasicDao>(basic_dao));

    return basic_dao;
}


} /* namespace dao */
} /* namespace svr */

using AbstractDAO_ptr = std::shared_ptr<svr::dao::AbstractDAO>;
