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

bool svr::business::SVRParametersService::is_manifold(const datamodel::t_param_set &param_set, datamodel::SVRParameters_ptr &p_out)
{
    return std::any_of(std::execution::par_unseq, param_set.begin(), param_set.end(), [](const auto &p) { return p->is_manifold(); });
}

using namespace svr::common;

namespace svr {
namespace business {

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


int SVRParametersService::save(const svr::datamodel::SVRParameters &svr_parameters)
{
   return svr_parameters_dao.save(std::make_shared<svr::datamodel::SVRParameters>(svr_parameters));
}


int SVRParametersService::save(const datamodel::SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }

    datamodel::SVRParameters_ptr p_saved_svr_parameters = std::make_shared<svr::datamodel::SVRParameters>(*svr_parameters);
    if (!p_saved_svr_parameters->get_id()) p_saved_svr_parameters->set_id(svr_parameters_dao.get_next_id());

    return svr_parameters_dao.save(p_saved_svr_parameters);
}


int SVRParametersService::remove(const svr::datamodel::SVRParameters &svr_parameters)
{
    return svr_parameters_dao.remove(std::make_shared<svr::datamodel::SVRParameters>(svr_parameters));
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

std::deque<datamodel::SVRParameters_ptr> SVRParametersService::get_by_dataset_column_level(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level)
{
    return svr_parameters_dao.get_svrparams(dataset_id, input_queue_column_name, decon_level);
}

datamodel::t_param_set_ptr
SVRParametersService::slice(const std::deque<datamodel::SVRParameters_ptr> &params, const size_t chunk_ix, const size_t grad_ix)
{
    auto r = std::make_shared<datamodel::t_param_set>();
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
            && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix))
            r->emplace(p);
    return r;
}

datamodel::t_param_set_ptr
SVRParametersService::slice(const datamodel::t_param_set &params, const size_t chunk_ix, const size_t grad_ix)
{
    auto r = std::make_shared<datamodel::t_param_set>();
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
            && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix))
            r->emplace(p);
    return r;
}

datamodel::SVRParameters_ptr SVRParametersService::find(const datamodel::t_param_set &params, const size_t chunk_ix, const size_t grad_ix)
{
    const auto res = std::find_if(std::execution::par_unseq, params.begin(), params.end(), [chunk_ix, grad_ix](const auto &p) {
        return (chunk_ix == std::numeric_limits<size_t>::max() || p->get_chunk_ix() == chunk_ix)
               && (grad_ix == std::numeric_limits<size_t>::max() || p->get_grad_level() == grad_ix);
    });
    if (res == params.end()) return nullptr;
    return *res;
}


datamodel::t_param_set_ptr
SVRParametersService::get_best_params(const t_gradient_tuned_parameters &tune_results)
{
    auto r = std::make_shared<datamodel::t_param_set>();
    for (const auto &tr: tune_results)
        r->emplace(tr.begin()->get()->p_params);
    return r;
}

}
}
