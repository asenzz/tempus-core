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

class RequestDAO : public AbstractDAO {
public:
    static RequestDAO *build(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source,
                             svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    explicit RequestDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bigint get_next_result_id() = 0;

    virtual int save(const datamodel::MultivalRequest_ptr &request) = 0;

    virtual bool exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
                        const bpt::time_duration &resolution, const std::string &value_columns) = 0;

    virtual bigint make_request(const std::string &user, const std::string &dataset_id, const std::string &start_time, const std::string &end_time,
                                const std::string &resolution, const std::string &value_columns) = 0;

    virtual datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end) = 0;

    virtual datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, const bpt::time_duration &resolution, const std::string &value_columns) = 0;

    virtual datamodel::MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, const bigint dataset_id) = 0;

    virtual std::deque<datamodel::MultivalRequest_ptr>
    get_active_multival_requests(const std::string &user_name, const bigint dataset_id) = 0;

    virtual std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, const bpt::time_duration &resolution) = 0;

    virtual std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results_str(const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start,
                             const std::string &value_time_end, const std::string &resolution) = 0;

    virtual std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results_column(
            const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
            const bpt::ptime &value_time_end, const bpt::time_duration &resolution) = 0;

    virtual int save(const datamodel::MultivalResponse_ptr &response) = 0;

    virtual int force_finalize(const datamodel::MultivalRequest_ptr &request) = 0;

    virtual void prune_finalized_requests(bpt::ptime const &before) = 0;
};

}
}
