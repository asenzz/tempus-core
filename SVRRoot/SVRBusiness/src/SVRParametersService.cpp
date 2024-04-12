#include "util/math_utils.hpp"
#include "common/parallelism.hpp"
#include "onlinesvr.hpp"
#include <vector>
#include <armadillo>
#include "appcontext.hpp"
#include "SVRParametersService.hpp"
#include "util/validation_utils.hpp"
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


int SVRParametersService::save(datamodel::SVRParameters &svr_parameters)
{
    return svr_parameters_dao.save(ptr(svr_parameters));
}


int SVRParametersService::save(const datamodel::SVRParameters_ptr &p_svr_parameters)
{
    if (!p_svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }

    return svr_parameters_dao.save(p_svr_parameters);
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

std::set<size_t> SVRParametersService::get_adjacent_indexes(const size_t level, const double ratio, const size_t level_count)
{
    if (level_count < MIN_LEVEL_COUNT or ratio == 0) return {level};
    //const size_t full_count = level_count * ratio - 1;
    const size_t half_count = (double(level_count) * ratio - 1.) / 2.;
    ssize_t min_index;
    ssize_t max_index;
    if (ratio == 1) { // TODO Hack, fix!
        min_index = 0;
        max_index = level_count - 1;
    } else {
        min_index = level - half_count;
        max_index = level + half_count;
    }
    if (min_index < 0) {
        max_index -= min_index;
        min_index -= min_index;
    }
    if (max_index >= ssize_t(level_count)) {
        min_index -= max_index - level_count + 1;
        max_index -= max_index - level_count + 1;
    }

    std::set<size_t> res;
    OMP_LOCK(level_indexes_l)
#pragma omp parallel for num_threads(adj_threads(max_index - min_index)) schedule(static, 1 + (max_index - min_index) / std::thread::hardware_concurrency())
    for (ssize_t level_index = min_index; level_index <= max_index; ++level_index) {
        if (level_index >= 0
            && level_index < ssize_t(level_count)
            && (level_count < MIN_LEVEL_COUNT || (level_index != ssize_t(level_count) / 2 && level_index % 2 == 0)
                    /*|| level_index > ssize_t(level_count) / 2*/))
        {
            omp_set_lock(&level_indexes_l);
            res.emplace(level_index);
            omp_unset_lock(&level_indexes_l);
        } else
            LOG4_TRACE("Skipping level " << level_index << " adjacent ratio " << ratio << " for level " << level);
    }

    LOG4_TRACE("Adjacent ratio " << ratio << " for level " << level << " includes levels " << res);

    return res;
}

}
}
