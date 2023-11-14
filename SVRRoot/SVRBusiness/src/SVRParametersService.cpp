#include "appcontext.hpp"
#include "SVRParametersService.hpp"
#include "util/ValidationUtils.hpp"
#include "DAO/SVRParametersDAO.hpp"
#include "model/SVRParameters.hpp"
#include "DAO/DatasetDAO.hpp"

using namespace svr::common;

namespace svr {
namespace business {

bool SVRParametersService::exists(const SVRParameters_ptr &svr_parameters)
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


int SVRParametersService::remove(const svr::datamodel::SVRParameters &svr_parameters)
{
    return svr_parameters_dao.remove(std::make_shared<svr::datamodel::SVRParameters>(svr_parameters));
}

int SVRParametersService::save(const SVRParameters_ptr &svr_parameters)
{
    if (!svr_parameters) {
        LOG4_ERROR("Parameters not initialized!");
        return 0;
    }

    SVRParameters_ptr p_saved_svr_parameters = std::make_shared<svr::datamodel::SVRParameters>(*svr_parameters);
    if (!p_saved_svr_parameters->get_id()) p_saved_svr_parameters->set_id(svr_parameters_dao.get_next_id());

    return svr_parameters_dao.save(p_saved_svr_parameters);
}


int SVRParametersService::remove(const SVRParameters_ptr &svr_parameters)
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


svr::datamodel::ensemble_svr_parameters_t SVRParametersService::get_all_by_dataset_id(const bigint dataset_id)
{
    const auto decon_levels = svr_parameters_dao.get_dataset_levels(dataset_id);
    std::vector<SVRParameters_ptr> vec_svr_parameters = svr_parameters_dao.get_all_svrparams_by_dataset_id(dataset_id);
    LOG4_DEBUG("Loaded " << vec_svr_parameters.size() << " parameter sets.");
    std::set<std::string> input_columns, input_tables;
    for (const auto &svr_param: vec_svr_parameters) {
        input_columns.insert(svr_param->get_input_queue_column_name());
        input_tables.insert(svr_param->get_input_queue_table_name());
    }

    svr::datamodel::ensemble_svr_parameters_t map_svr_parameters;
    for (const SVRParameters_ptr &svr_parameters: vec_svr_parameters)
    {
        LOG4_TRACE("Adding " << svr_parameters->get_input_queue_table_name() << " " << svr_parameters->get_input_queue_column_name() << ": " << svr_parameters->to_sql_string());
        if (map_svr_parameters[{svr_parameters->get_input_queue_table_name(), svr_parameters->get_input_queue_column_name()}].empty())
            map_svr_parameters[{svr_parameters->get_input_queue_table_name(), svr_parameters->get_input_queue_column_name()}].resize(decon_levels);
        map_svr_parameters[{svr_parameters->get_input_queue_table_name(), svr_parameters->get_input_queue_column_name()}][svr_parameters->get_decon_level()] = svr_parameters;
    }

    return map_svr_parameters;
}

}
}
