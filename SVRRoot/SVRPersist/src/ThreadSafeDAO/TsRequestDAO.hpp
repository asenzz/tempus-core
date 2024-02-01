#pragma once

#include "TsDaoBase.hpp"
#include <DAO/RequestDAO.hpp>


namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsRequestDAO, RequestDAO)

    virtual bigint get_next_id();
    virtual bigint get_next_result_id();

    virtual int save(const datamodel::MultivalRequest_ptr& request);

    virtual datamodel::MultivalRequest_ptr get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end);

    virtual datamodel::MultivalRequest_ptr get_multival_request(const std::string &user_name, const bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns);

    virtual datamodel::MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, const bigint dataset_id);

    virtual std::deque<datamodel::MultivalRequest_ptr> get_active_multival_requests(const std::string &user_name, const bigint dataset_id);

    virtual std::deque<datamodel::MultivalResponse_ptr> get_multival_results(const std::string &user_name, const bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution);

    virtual std::deque<datamodel::MultivalResponse_ptr> get_multival_results_column(const std::string &user_name, const std::string &column_name,
            bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution);

    virtual int save(const datamodel::MultivalResponse_ptr &response);

    virtual int force_finalize(const datamodel::MultivalRequest_ptr &request);

    virtual void prune_finalized_requests(bpt::ptime const & olderThan);
};

}
}
