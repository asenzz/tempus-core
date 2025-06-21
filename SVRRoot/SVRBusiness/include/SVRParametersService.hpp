#pragma once

#include <memory>
#include <limits>
#include <set>
#include <tuple>
#include <armadillo>
#include <deque>
#include <oneapi/tbb/concurrent_map.h>
#include <oneapi/tbb/concurrent_set.h>
#include "common/types.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"

namespace svr { namespace dao { class  SVRParametersDAO; } }

namespace svr {
namespace business {

class SVRParametersService {
    svr::dao::SVRParametersDAO &svr_parameters_dao;

public:
    explicit SVRParametersService(dao::SVRParametersDAO &svr_parameters_dao);
    bool exists(const datamodel::SVRParameters_ptr &svr_parameters);
    bool exists(const bigint svr_parameters_id);
    int save(datamodel::SVRParameters &svr_parameters);
    int save(const datamodel::SVRParameters_ptr &p_svr_parameters);
    int remove(const datamodel::SVRParameters_ptr &svr_parameters);
    int remove(const datamodel::SVRParameters &svr_parameters);
    int remove_by_dataset(const bigint dataset_id);

    std::deque<datamodel::SVRParameters_ptr> get_all_by_dataset_id(const bigint dataset_id);
    std::deque<datamodel::SVRParameters_ptr> get_by_dataset_column_level(const bigint dataset_id, const std::string &input_queue_column_name, const uint16_t decon_level, const uint16_t step);

    static datamodel::t_param_set slice(
            const datamodel::t_param_set &params,
            const uint16_t chunk_ix = std::numeric_limits<uint16_t>::max(),
            const uint16_t grad_ix = std::numeric_limits<uint16_t>::max());
    static datamodel::t_param_set slice(
            const std::deque<datamodel::SVRParameters_ptr> &params,
            const uint16_t chunk_ix = std::numeric_limits<uint16_t>::max(),
            const uint16_t grad_ix = std::numeric_limits<uint16_t>::max());

    static datamodel::t_param_set::iterator find(datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix);
    static datamodel::t_param_set::const_iterator find(const datamodel::t_param_set &params, const uint16_t chunk_ix, const uint16_t grad_ix);
    static datamodel::SVRParameters_ptr find_ptr(
            const datamodel::t_param_set &params,
            const uint16_t chunk_ix = std::numeric_limits<uint16_t>::max(),
            const uint16_t grad_ix = std::numeric_limits<uint16_t>::max());

    static bool check(const datamodel::t_param_set &params, const uint16_t num_chunks);

    static datamodel::SVRParameters_ptr is_manifold(const datamodel::t_param_set &param_set);

    static datamodel::SVRParameters_ptr is_tft(const datamodel::t_param_set &param_set);

    static uint16_t get_trans_levix(const uint16_t levels);

    static std::set<uint16_t> get_adjacent_indexes(const uint16_t level, const double ratio, const uint16_t level_count);
};

}
}
