#pragma once

#include "TsDaoBase.hpp"
#include <DAO/RequestDAO.hpp>


namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsRequestDAO, RequestDAO)

    virtual bigint get_next_id();
    virtual bigint get_next_result_id();

    virtual int save(const MultivalRequest_ptr& request);

    virtual MultivalRequest_ptr get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end);

    virtual MultivalRequest_ptr get_multival_request(const std::string &user_name, bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns);

    virtual MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, bigint dataset_id);

    virtual std::vector<MultivalRequest_ptr> get_active_multival_requests(const std::string &user_name, bigint dataset_id, std::string const & inputQueueName);

    virtual std::vector<MultivalResponse_ptr> get_multival_results(const std::string &user_name, bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution);

    virtual std::vector<MultivalResponse_ptr> get_multival_results_column(const std::string &user_name, const std::string &column_name,
            bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution);

    virtual int save(const MultivalResponse_ptr &response);

    virtual int force_finalize(const MultivalRequest_ptr &request);

    virtual void prune_finalized_requests(bpt::ptime const & olderThan);
};

}
}
