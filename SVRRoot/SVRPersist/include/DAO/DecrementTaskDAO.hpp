#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class DecrementTask; } }
using DecrementTask_ptr = std::shared_ptr<svr::datamodel::DecrementTask>;

namespace svr {
namespace dao {

class DecrementTaskDAO : public AbstractDAO
{
public:
    static DecrementTaskDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit DecrementTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const DecrementTask_ptr& decrementTask) = 0;
    virtual int remove(const DecrementTask_ptr& decrementTask) = 0;
    virtual DecrementTask_ptr get_by_id(const bigint id) = 0;
};

} // namespace dao
} // namespace svr

using DecrementTaskDAO_ptr = std::shared_ptr<svr::dao::DecrementTaskDAO>;
