#ifndef PGDECREMENTTASKDAO_HPP
#define PGDECREMENTTASKDAO_HPP

#include "DAO/DecrementTaskDAO.hpp"

namespace svr {
namespace dao {

class PgDecrementTaskDAO : public DecrementTaskDAO
{
public:
    explicit PgDecrementTaskDAO(common::PropertiesReader& sql_properties, dao::DataSource& data_source);

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const DecrementTask_ptr& decrementTask);
    virtual int remove(const DecrementTask_ptr& decrementTask);
    virtual DecrementTask_ptr get_by_id(const bigint id);
};

} }

#endif /* PGDECREMENTTASKDAO_HPP */

