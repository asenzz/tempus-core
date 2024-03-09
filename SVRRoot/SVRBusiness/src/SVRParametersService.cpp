#include "util/math_utils.hpp"
#include "common/parallelism.hpp"
#include "onlinesvr.hpp"
#include <vector>
#include <armadillo>
#include "appcontext.hpp"
#include "SVRParametersService.hpp"
#include "util/ValidationUtils.hpp"
#include "DAO/SVRParametersDAO.hpp"
#include "model/SVRParameters.hpp"
#include "DAO/DatasetDAO.hpp"

namespace svr {
namespace business {

SVRParametersService::SVRParametersService(dao::SVRParametersDAO &svr_parameters_dao):
    svr_parameters_dao(svr_parameters_dao)
{}


datamodel::SVRParameters_ptr SVRParametersService::is_manifold(const datamodel::t_param_set &param_set)
{
    auto res = std::find_if(std::execution::par_unseq, param_set.cbegin(), param_set.cend(), [](const auto &p) { return p->is_manifold(); });
    return res == param_set.end() ? nullptr : *res;
}

bool SVRParametersService::exists(const datamodel::SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }
    return exists(svr_parameters->get_id());
}


bool SVRParametersService::exists(const bigint svr_parameters_id)
{
    return svr_parameters_dao.exists(svr_parameters_id);
}


int SVRParametersService::save(const datamodel::SVRParameters &svr_parameters)
{
    return svr_parameters_dao.save(ptr<datamodel::SVRParameters>(svr_parameters));
}


int SVRParametersService::save(const datamodel::SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }

    auto p_saved_svr_parameters = ptr<datamodel::SVRParameters>(*svr_parameters);
    if (!p_saved_svr_parameters->get_id()) p_saved_svr_parameters->set_id(svr_parameters_dao.get_next_id());

    return svr_parameters_dao.save(p_saved_svr_parameters);
}


int SVRParametersService::remove(const datamodel::SVRParameters &svr_parameters)
{
    return svr_parameters_dao.remove(ptr<datamodel::SVRParameters>(svr_parameters));
}

int SVRParametersService::remove(const datamodel::SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }
    return svr_parameters_dao.remove(svr_parameters);
}


int SVRParametersService::remove_by_dataset(const bigint dataset_id)
{
    return svr_parameters_dao.remove_by_dataset_id(dataset_id);
}


std::deque<datamodel::SVRParameters_ptr> SVRParametersService::get_all_by_dataset_id(const bigint dataset_id)
{
    return svr_parameters_dao.get_all_svrparams_by_dataset_id(dataset_id);
}

std::deque<datamodel::SVRParameters_ptr>
SVRParametersService::get_by_dataset_column_level(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level)
{
    return svr_parameters_dao.get_svrparams(dataset_id, input_queue_column_name, decon_level);
}

datamodel::t_param_set
SVRParametersService::slice(const std::deque<datamodel::SVRParameters_ptr> &params, const size_t chunk_ix, const size_t grad_ix)
{
    datamodel::t_param_set r;
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
            && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix))
            r.emplace(p);
    return r;
}

datamodel::t_param_set
SVRParametersService::slice(const datamodel::t_param_set &params, const size_t chunk_ix, const size_t grad_ix)
{
    datamodel::t_param_set r;
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
            && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix))
            r.emplace(p);
    return r;
}

datamodel::SVRParameters_ptr SVRParametersService::find(const datamodel::t_param_set &params, const size_t chunk_ix, const size_t grad_ix)
{
    const auto res = std::find_if(std::execution::par_unseq, params.cbegin(), params.cend(), [chunk_ix, grad_ix](const auto &p) {
        return (chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
               && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix);
    });
    return res == params.end() ? nullptr : *res;
}

bool SVRParametersService::check(const datamodel::t_param_set &params, const size_t num_chunks)
{
    std::deque<bool> present(num_chunks, false);
    for (const auto &p: params) present[p->get_chunk_ix()] = true;
    return std::all_of(std::execution::par_unseq, present.cbegin(), present.cend(), [](const auto p) { return p; });
}


}
}
