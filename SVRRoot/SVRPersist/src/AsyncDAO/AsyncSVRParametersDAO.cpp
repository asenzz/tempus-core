#include "AsyncSVRParametersDAO.hpp"
#include "../PgDAO/PgSVRParametersDAO.hpp"
#include "AsyncImplBase.hpp"
#include "model/SVRParameters.hpp"

namespace svr {
namespace dao {

using datamodel::SVRParameters;

namespace {
static const auto cmp_primary_key = [] (datamodel::SVRParameters_ptr const &lhs, datamodel::SVRParameters_ptr const &rhs)
{
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_id() == rhs->get_id();
};

static const auto cmp_whole_value = [] (datamodel::SVRParameters_ptr const &lhs, datamodel::SVRParameters_ptr const &rhs)
{
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncSVRParametersDAO::AsyncImpl
        : AsyncImplBase<datamodel::SVRParameters_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgSVRParametersDAO>
{
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};


AsyncSVRParametersDAO::AsyncSVRParametersDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : SVRParametersDAO(tempus_config, data_source), pImpl(*new AsyncSVRParametersDAO::AsyncImpl(tempus_config, data_source))
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
    auto svrParams = ptr<SVRParameters>();
    svrParams->set_id(id);

    if (pImpl.cached(svrParams)) return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncSVRParametersDAO::save(const datamodel::SVRParameters_ptr &p_svr_parameters)
{
    pImpl.cache(p_svr_parameters); // TODO Hack, review!
    return 1;
}

int AsyncSVRParametersDAO::remove(const datamodel::SVRParameters_ptr &svr_parameters)
{
    return pImpl.remove(svr_parameters);
}

int AsyncSVRParametersDAO::remove_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    std::deque<datamodel::SVRParameters_ptr> params;
    {  // Important to unlock before calling remove()
        const std::scoped_lock lg(pImpl.pgMutex);
        params = pImpl.pgDao.get_all_svrparams_by_dataset_id(dataset_id);
    }

    for (const auto &p: params) pImpl.remove(p);

    return params.size();
}

std::deque<datamodel::SVRParameters_ptr> AsyncSVRParametersDAO::get_all_svrparams_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    std::deque<datamodel::SVRParameters_ptr> params = pImpl.pgDao.get_all_svrparams_by_dataset_id(dataset_id);

    for (const auto &p: params) pImpl.cache_no_store(p);

    return params;
}

std::deque<datamodel::SVRParameters_ptr>
AsyncSVRParametersDAO::get_svrparams(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level, const size_t step)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    std::deque<datamodel::SVRParameters_ptr> params = pImpl.pgDao.get_svrparams(dataset_id, input_queue_column_name, decon_level, step);
    for (const auto &p: params) pImpl.cache_no_store(p);
    return params;
}

size_t AsyncSVRParametersDAO::get_dataset_levels(const bigint dataset_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_dataset_levels(dataset_id);;
}

}
}
