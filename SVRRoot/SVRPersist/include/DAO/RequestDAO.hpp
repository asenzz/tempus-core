#pragma once

#include "DAO/AbstractDAO.hpp"

namespace svr {
namespace datamodel {
class MultivalRequest;

class MultivalResponse;

class ValueRequest;

class User;

class Dataset;
}
}
using MultivalRequest_ptr = std::shared_ptr<svr::datamodel::MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<svr::datamodel::MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<svr::datamodel::ValueRequest>;
using User_ptr = std::shared_ptr<svr::datamodel::User>;
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;

namespace svr {
namespace dao {

class RequestDAO : public AbstractDAO
{
public:
    static RequestDAO *build(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source,
                             svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit RequestDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    virtual bigint get_next_id() = 0;

    virtual bigint get_next_result_id() = 0;

    virtual int save(const MultivalRequest_ptr &request) = 0;

    virtual MultivalRequest_ptr
    get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end) = 0;

    virtual MultivalRequest_ptr
    get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, size_t resolution, std::string const &value_columns) = 0;

    virtual MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, bigint dataset_id) = 0;

    virtual std::vector<MultivalRequest_ptr>
    get_active_multival_requests(const std::string &user_name, bigint dataset_id,
                                 std::string const &inputQueueName) = 0;

    virtual std::vector<MultivalResponse_ptr>
    get_multival_results(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end, const size_t resolution) = 0;

    virtual std::vector<MultivalResponse_ptr>
    get_multival_results_column(
            const std::string &user_name, const std::string &column_name, bigint dataset_id, const bpt::ptime &value_time_start,
            const bpt::ptime &value_time_end, const size_t resolution) = 0;

    virtual int save(const MultivalResponse_ptr &response) = 0;

    virtual int force_finalize(const MultivalRequest_ptr &request) = 0;

    virtual void prune_finalized_requests(bpt::ptime const &olderThan) = 0;
};

}
}
