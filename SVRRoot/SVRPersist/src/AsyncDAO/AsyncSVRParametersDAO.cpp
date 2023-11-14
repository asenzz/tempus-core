#include "AsyncSVRParametersDAO.hpp"
#include "../PgDAO/PgSVRParametersDAO.hpp"
#include "AsyncImplBase.hpp"
#include "model/SVRParameters.hpp"

namespace svr { namespace dao {

using svr::datamodel::SVRParameters;

namespace {
    static bool cmp_primary_key(SVRParameters_ptr const & lhs, SVRParameters_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(SVRParameters_ptr const & lhs, SVRParameters_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncSVRParametersDAO::AsyncImpl
    : AsyncImplBase<SVRParameters_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgSVRParametersDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};


AsyncSVRParametersDAO::AsyncSVRParametersDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: SVRParametersDAO(sqlProperties, dataSource), pImpl( *new AsyncSVRParametersDAO::AsyncImpl(sqlProperties, dataSource))
{}

AsyncSVRParametersDAO::~AsyncSVRParametersDAO()
{
    delete &pImpl;
}

bigint AsyncSVRParametersDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncSVRParametersDAO::exists(const bigint id)
{
    SVRParameters_ptr svrParams {new SVRParameters()};
    svrParams->set_id(id);

    if (pImpl.cached(svrParams)) return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncSVRParametersDAO::save(const SVRParameters_ptr &svr_parameters)
{
    pImpl.cache(svr_parameters);
    return 1;
}

int AsyncSVRParametersDAO::remove(const SVRParameters_ptr& svr_parameters)
{
    return pImpl.remove(svr_parameters);
}

int AsyncSVRParametersDAO::remove_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    std::vector<SVRParameters_ptr> params;
    {  // Important to unlock before calling remove()
        const std::scoped_lock lg(pImpl.pgMutex);
        params = pImpl.pgDao.get_all_svrparams_by_dataset_id(dataset_id);
    }

    for (const auto &p: params) pImpl.remove(p);

    return params.size();
}

std::vector<SVRParameters_ptr> AsyncSVRParametersDAO::get_all_svrparams_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    std::vector<SVRParameters_ptr> params = pImpl.pgDao.get_all_svrparams_by_dataset_id(dataset_id);

    for(const auto &p: params) pImpl.cache_no_store(p);

    return params;
}

size_t AsyncSVRParametersDAO::get_dataset_levels(const bigint dataset_id)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_dataset_levels(dataset_id);;
}

} }
