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

SVRParametersService::SVRParametersService(dao::SVRParametersDAO &svr_parameters_dao) :
        svr_parameters_dao(svr_parameters_dao)
{}


datamodel::SVRParameters_ptr SVRParametersService::is_manifold(const datamodel::t_param_set &param_set)
{
    auto res = std::find_if(C_default_exec_policy, param_set.cbegin(), param_set.cend(), [](const auto &p) { return p->is_manifold(); });
    return res == param_set.end() ? nullptr : *res;
}

datamodel::SVRParameters_ptr SVRParametersService::is_tft(const datamodel::t_param_set &param_set)
{
    auto res = std::find_if(C_default_exec_policy, param_set.cbegin(), param_set.cend(), [](const auto &p) {
        return p->get_kernel_type() == datamodel::e_kernel_type::TFT || p->get_kernel_type() == datamodel::e_kernel_type::GBM;
    });
    return res == param_set.end() ? nullptr : *res;
}

bool SVRParametersService::exists(const datamodel::SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters || !svr_parameters->get_id()) {
        LOG4_ERROR("Parameters not initialized or id is zero!");
        return false;
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
SVRParametersService::get_by_dataset_column_level(const bigint dataset_id, const std::string &input_queue_column_name, const uint16_t decon_level, const uint16_t step)
{
    return svr_parameters_dao.get_svrparams(dataset_id, input_queue_column_name, decon_level, step);
}

datamodel::t_param_set
SVRParametersService::slice(const std::deque<datamodel::SVRParameters_ptr> &params, const uint16_t chunk_ix, const uint16_t grad_ix)
{
    datamodel::t_param_set r;
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<DTYPE(chunk_ix)>::max() || p->get_chunk_index() == chunk_ix)
            && (grad_ix == std::numeric_limits<DTYPE(grad_ix)>::max() || p->get_grad_level() == grad_ix))
            r.emplace(p);
    return r;
}

datamodel::t_param_set
SVRParametersService::slice(const datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix)
{
    datamodel::t_param_set r;
    for (const auto &p: params)
        if ((chunk_ix == std::numeric_limits<DTYPE(chunk_ix)>::max() || p->get_chunk_index() == chunk_ix)
            && (grad_ix == std::numeric_limits<DTYPE(grad_ix)>::max() || p->get_grad_level() == grad_ix))
            r.emplace(p);
    return r;
}


datamodel::t_param_set::iterator SVRParametersService::find(datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix)
{
    return std::find_if(C_default_exec_policy, params.begin(), params.end(), [chunk_ix, grad_ix](const auto &p) {
        return (chunk_ix == std::numeric_limits<DTYPE(chunk_ix)>::max() || p->get_chunk_index() == chunk_ix)
               && (grad_ix == std::numeric_limits<DTYPE(grad_ix)>::max() || p->get_grad_level() == grad_ix);
    });
}

datamodel::t_param_set::const_iterator SVRParametersService::find(const datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix)
{
    return std::find_if(C_default_exec_policy, params.cbegin(), params.cend(), [chunk_ix, grad_ix](const auto &p) {
        return (chunk_ix == std::numeric_limits<DTYPE(chunk_ix)>::max() || p->get_chunk_index() == chunk_ix)
               && (grad_ix == std::numeric_limits<DTYPE(grad_ix)>::max() || p->get_grad_level() == grad_ix);
    });
}

datamodel::SVRParameters_ptr SVRParametersService::find_ptr(const datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix)
{
    const auto res = find(params, chunk_ix, grad_ix);
    return res == params.cend() ? nullptr : *res;
}

bool SVRParametersService::check(const datamodel::t_param_set &params, const uint16_t num_chunks)
{
    std::deque<bool> present(num_chunks, false);
    for (const auto &p: params) present[p->get_chunk_index()] = true;
    return std::all_of(C_default_exec_policy, present.cbegin(), present.cend(), std::identity());
}

uint16_t SVRParametersService::get_trans_levix(const uint16_t levels)
{
#if defined(VMD_ONLY) || defined(EMD_ONLY)
    return std::numeric_limits<DTYPE(levels)>::max();
#else
    return levels < MIN_LEVEL_COUNT ? std::numeric_limits<DTYPE(levels)>::max() : levels / 2;
#endif
}

std::set<uint16_t> SVRParametersService::get_adjacent_indexes(const uint16_t level, const double ratio, const uint16_t level_count)
{
    if (level_count < MIN_LEVEL_COUNT or ratio == 0) return {level};
    //const uint16_t full_count = level_count * ratio - 1;
    const uint16_t half_count = (level_count * ratio - 1.) / 2.;
    int16_t min_index;
    int16_t max_index;
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
    if (max_index >= level_count) {
        min_index -= max_index - level_count + 1;
        max_index -= max_index - level_count + 1;
    }

    std::set<uint16_t> res;
    tbb::mutex level_indexes_l;
    const auto n_indexes = 1 + max_index - min_index;
    LOG4_DEBUG("Getting adjacent indexes for level " << level << " with ratio " << ratio << " and level count " << level_count << " min " << min_index << " max " << max_index <<
        ", n_indexes " << n_indexes);
#if !defined(VMD_ONLY) && !defined(EMD_ONLY)
    const int16_t trans_levix = get_trans_levix(level_count);
#endif
    OMP_FOR_(n_indexes, simd firstprivate(level_count, ratio, level))
    for (int16_t i = min_index; i <= max_index; ++i) {
        if (i >= 0
            && i < level_count
#ifdef VMD_ONLY
            && i % 2 == 0
#elif defined(EMD_ONLY)
#else
            && (level_count < MIN_LEVEL_COUNT || (i != trans_levix && i % 2 == 0) /* || level > trans_levix */ )
#endif
            ) {
            const tbb::mutex::scoped_lock lk(level_indexes_l);
            res.emplace(i);
        } else
            LOG4_TRACE("Skipping level " << i << " adjacent ratio " << ratio << " for level " << level);
    }

    LOG4_TRACE("Adjacent ratio " << ratio << " for level " << level << " includes levels " << res);

    return res;
}

}
}
