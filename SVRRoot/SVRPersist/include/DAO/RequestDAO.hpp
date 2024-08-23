#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr {
namespace datamodel {
struct MultivalRequest;
struct MultivalResponse;
struct ValueRequest;
class User;
class Dataset;
using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<ValueRequest>;
using User_ptr = std::shared_ptr<User>;
using Dataset_ptr = std::shared_ptr<Dataset>;
}
}

namespace svr {
namespace dao {

class RequestDAO : public AbstractDAO
{
public:
    static RequestDAO *build(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source,
                             svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit RequestDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bigint get_next_result_id() = 0;

    virtual int save(const datamodel::MultivalRequest_ptr &request) = 0;

    virtual datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end) = 0;

    virtual datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, size_t resolution, std::string const &value_columns) = 0;

    virtual datamodel::MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, const bigint dataset_id) = 0;

    virtual std::deque<datamodel::MultivalRequest_ptr>
    get_active_multival_requests(const std::string &user_name, const bigint dataset_id) = 0;

    virtual std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, const size_t resolution) = 0;

    virtual std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results_column(
            const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
            const bpt::ptime &value_time_end, const size_t resolution) = 0;

    virtual int save(const datamodel::MultivalResponse_ptr &response) = 0;

    virtual int force_finalize(const datamodel::MultivalRequest_ptr &request) = 0;

    virtual void prune_finalized_requests(bpt::ptime const &olderThan) = 0;
};

}
}
