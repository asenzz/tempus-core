#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr {

namespace datamodel { class DecrementTask; }

namespace dao {

using DecrementTask_ptr = std::shared_ptr<datamodel::DecrementTask>;

class DecrementTaskDAO : public AbstractDAO
{
public:
    static DecrementTaskDAO * build(common::PropertiesReader& sql_properties, dao::DataSource& data_source, const common::ConcreteDaoType dao_type, const bool use_threadsafe_dao);

    explicit DecrementTaskDAO(common::PropertiesReader& sql_properties, dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;
    virtual bool exists(const bigint id) = 0;
    virtual int save(const DecrementTask_ptr& decrementTask) = 0;
    virtual int remove(const DecrementTask_ptr& decrementTask) = 0;
    virtual DecrementTask_ptr get_by_id(const bigint id) = 0;
};

using DecrementTaskDAO_ptr = std::shared_ptr<dao::DecrementTaskDAO>;

} // namespace dao
} // namespace svr