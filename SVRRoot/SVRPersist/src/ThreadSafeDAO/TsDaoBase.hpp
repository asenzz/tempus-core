#pragma once

#include <functional>
#include <memory>
#include <mutex>

namespace svr {
namespace dao {

std::mutex &common_mutex_instance();

template<class ConcreteDao> class TsDaoBase {
protected:
    std::mutex &mutex;
    std::unique_ptr<ConcreteDao> dao;
public:
    explicit TsDaoBase(std::unique_ptr<ConcreteDao> dao) : mutex(common_mutex_instance()), dao(std::move(dao))
    {}

    template<class Result, class Method, class ...Args>
    Result ts_call(Method method,
                   Args ...args) // TODO Replace with a different kind of wrapper, because this way, binding to template parameter we effectively disable function overloading
    {
        const std::scoped_lock lock(mutex);
        auto fun = std::bind(method, dao.get(), args...);
        return fun();
    }
};

}
}

#define THREADSAFE_DAO_CLASS_DECLARATION_HEADER(TS_DAO, BASIC_DAO) class TS_DAO : public BASIC_DAO, public TsDaoBase<BASIC_DAO> { \
public: \
    TS_DAO(svr::common::PropertiesReader& sql_properties, svr::dao::DataSource& data_source, std::unique_ptr<BASIC_DAO> dao);

#define DEFINE_THREADSAFE_DAO_CONSTRUCTOR(TS_DAO, BASIC_DAO) TS_DAO::TS_DAO(svr::common::PropertiesReader& sql_properties, svr::dao::DataSource& data_source, std::unique_ptr<BASIC_DAO> dao) \
: BASIC_DAO(sql_properties, data_source) \
, TsDaoBase<BASIC_DAO>(std::move(dao))
