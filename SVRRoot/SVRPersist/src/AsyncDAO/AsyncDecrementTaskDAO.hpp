#ifndef ASYNCDECREMENTTASKDAO_HPP
#define ASYNCDECREMENTTASKDAO_HPP

#include <DAO/DecrementTaskDAO.hpp>

namespace svr {
namespace dao {

class AsyncDecrementTaskDAO : public DecrementTaskDAO
{
public:
    explicit AsyncDecrementTaskDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncDecrementTaskDAO();

    virtual bigint get_next_id();
    virtual bool exists(const bigint id);
    virtual int save(const DecrementTask_ptr& decrementTask);
    virtual int remove(const DecrementTask_ptr& decrementTask);
    virtual DecrementTask_ptr get_by_id(const bigint id);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

} }

#endif /* ASYNCDECREMENTTASKDAO_HPP */
