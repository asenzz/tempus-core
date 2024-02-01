#pragma once

#include <memory>
#include "common/types.hpp"
#include "model/SVRParameters.hpp"

namespace svr { namespace dao { class  SVRParametersDAO; } }

namespace svr {
namespace business {

class SVRParametersService {
    svr::dao::SVRParametersDAO &svr_parameters_dao;

public:
    SVRParametersService(svr::dao::SVRParametersDAO &svr_parameters_dao):
        svr_parameters_dao(svr_parameters_dao)
    {}

    bool exists(const datamodel::SVRParameters_ptr &svr_parameters);
    bool exists(const bigint svr_parameters_id);
    int save(const datamodel::SVRParameters &svr_parameters);
    int save(const datamodel::SVRParameters_ptr &svr_parameters);
    int remove(const datamodel::SVRParameters_ptr &svr_parameters);
    int remove(const datamodel::SVRParameters &svr_parameters);
    int remove_by_dataset(const bigint dataset_id);

    std::deque<datamodel::SVRParameters_ptr> get_all_by_dataset_id(const bigint dataset_id);
    std::deque<datamodel::SVRParameters_ptr> get_by_dataset_column_level(const bigint dataset_id, const std::string &input_queue_column_name, const size_t decon_level);

    static datamodel::t_param_set_ptr slice(const datamodel::t_param_set &params,
                                            const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max());
    static datamodel::t_param_set_ptr slice(const std::deque<datamodel::SVRParameters_ptr> &params,
                                            const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max());
    static datamodel::SVRParameters_ptr find(const datamodel::t_param_set &params,
                                             const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_level = std::numeric_limits<size_t>::max());
    static datamodel::SVRParameters_ptr &find(datamodel::t_param_set &params,
                                              const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max());
};

}
}
