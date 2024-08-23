#include "PgModelDAO.hpp"
#include "PgDQScalingFactorDAO.hpp"
#include <DAO/ModelRowMapper.hpp>
#include <DAO/DataSource.hpp>
#include <appcontext.hpp>

namespace svr {
namespace dao {

PgModelDAO::PgModelDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
        : ModelDAO(tempus_config, data_source)
{}

bigint PgModelDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bool PgModelDAO::exists(const bigint model_id)
{
    return data_source.query_for_type<int>(get_sql("exists_by_id"), model_id) == 1;
}

bool PgModelDAO::svr_exists(const bigint svr_id)
{
    return data_source.query_for_type<int>(get_sql("svr_exists_by_id"), svr_id) == 1;
}

int PgModelDAO::save(const datamodel::Model_ptr &model)
{
    int res = 0;
    if (!exists(model->get_id()))
        res += data_source.update(
                get_sql("save"),
                model->get_id(),
                model->get_ensemble_id(),
                model->get_decon_level(),
                model->get_last_modified(),
                model->get_last_modeled_value_time()
        );
    else
        res += data_source.update(
                get_sql("update"),
                model->get_last_modified(),
                model->get_last_modeled_value_time(),
                model->get_id()
        );

#pragma omp parallel for num_threads(adj_threads(model->get_gradient_count())) schedule(static, 1)
    for (const auto &p_svr: model->get_gradients()) {
        if (!svr_exists(model->get_id()))
            res += data_source.update(
                    get_sql("save_svr"),
                    p_svr->get_id(),
                    p_svr->get_model_id(),
                    p_svr->save());
        else
            res += data_source.update(
                    get_sql("update_svr"),
                    p_svr->get_id(),
                    p_svr->get_model_id(),
                    p_svr->save());
        const auto p_saved_svr = p_svr->is_manifold() ? p_svr->get_manifold() : p_svr;
        if (!p_saved_svr) continue;
        std::for_each(C_default_exec_policy, p_saved_svr->get_scaling_factors().cbegin(), p_saved_svr->get_scaling_factors().cend(),
                 [&](const auto &s) { if (APP.dq_scaling_factor_service.exists(s)) APP.dq_scaling_factor_service.remove(s); APP.dq_scaling_factor_service.save(s); });
        std::for_each(C_default_exec_policy, p_saved_svr->get_param_set().cbegin(), p_saved_svr->get_param_set().cend(),
              [&](const auto &s) { if (APP.svr_parameters_service.exists(s)) APP.svr_parameters_service.remove(s); APP.svr_parameters_service.save(s); });
    }

    return res;
}

int PgModelDAO::remove(const datamodel::Model_ptr &model)
{
    return data_source.update(get_sql("remove"), model->get_id());
}

int PgModelDAO::remove_by_ensemble_id(const bigint ensemble_id)
{
    return data_source.update(get_sql("remove_by_ensemble_id"), ensemble_id);
}

datamodel::Model_ptr PgModelDAO::get_by_id(const bigint model_id)
{
    ModelRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_id"), model_id);
}

datamodel::Model_ptr PgModelDAO::get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level)
{
    ModelRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_by_ensemble_id_and_decon_level"), ensemble_id, decon_level);
}

std::deque<datamodel::Model_ptr> PgModelDAO::get_all_ensemble_models(const bigint ensemble_id)
{
    ModelRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, get_sql(/* "get_all_ensemble_models_empty" */ "get_all_ensemble_models"), ensemble_id);
}

std::deque<datamodel::OnlineMIMOSVR_ptr> PgModelDAO::get_svr_by_model_id(const bigint model_id)
{
    SVRModelRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, get_sql("get_svr_by_model_id"), model_id);
}

} // namespace dao
} // namespace svr
