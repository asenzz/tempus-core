#pragma once

#include <memory>
#include <common/types.hpp>

namespace svr { namespace dao { class RequestDAO; }}
namespace svr {
namespace datamodel {
struct MultivalRequest;
struct MultivalResponse;
struct ValueRequest;

class User;

class Dataset;

class InputQueue;

using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<ValueRequest>;
using User_ptr = std::shared_ptr<User>;
using Dataset_ptr = std::shared_ptr<Dataset>;
}
}


namespace svr {
namespace business {

class RequestService {

private:
    svr::dao::RequestDAO &request_dao;

public:
    RequestService(svr::dao::RequestDAO &request_dao);

    int save(const datamodel::MultivalRequest_ptr &request);

    datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end);

    datamodel::MultivalRequest_ptr get_multival_request(
            const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                                                        const bpt::time_duration &resolution, const std::string &value_columns);

    datamodel::MultivalRequest_ptr get_latest_multival_request(const svr::datamodel::User &user, const svr::datamodel::Dataset &dataset);

    std::deque<datamodel::MultivalRequest_ptr> get_active_multival_requests(svr::datamodel::User const &user, svr::datamodel::Dataset const &dataset);

    std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution);

    std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results(const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, const bpt::time_duration &resolution);

    std::deque<datamodel::MultivalResponse_ptr> get_multival_results(
            const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end, const std::string &resolution);

    int force_finalize(const datamodel::MultivalRequest_ptr &request);

    int save(const datamodel::MultivalResponse_ptr &response);

    void prune_finalized_requests(const bpt::ptime &older_than);

    bool
    exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
           const bpt::time_duration &resolution, const std::string &value_columns);

    bigint make_request(
            const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
            const std::string &resolution_str, const std::string &value_columns);
};

} /* namespace business */ } /* namespace svr */

using RequestService_ptr = std::shared_ptr<svr::business::RequestService>;

