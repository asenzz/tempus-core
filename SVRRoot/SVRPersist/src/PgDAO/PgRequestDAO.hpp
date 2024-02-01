#ifndef PGREQUESTDAO_HPP
#define PGREQUESTDAO_HPP

#include <DAO/RequestDAO.hpp>

namespace svr {
namespace dao {

class PgRequestDAO : public RequestDAO
{
public:
    explicit PgRequestDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    bigint get_next_id();

    bigint get_next_result_id();

    int save(const datamodel::MultivalRequest_ptr &request);

    datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                         const bpt::ptime &value_time_end);

    datamodel::MultivalRequest_ptr get_multival_request(const std::string &user_name, const bigint dataset_id,
                                             const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                             size_t resolution, std::string const &value_columns);

    datamodel::MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, const bigint dataset_id);

    std::deque<datamodel::MultivalRequest_ptr>
    get_active_multival_requests(const std::string &user_name, const bigint dataset_id);

    std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results(
            const std::string &user_name, const bigint dataset_id,
            const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution);

    std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results_column(const std::string &user_name, const std::string &column_name, const bigint dataset_id,
                                const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                const size_t resolution);

    int save(const datamodel::MultivalResponse_ptr &response);

    int force_finalize(const datamodel::MultivalRequest_ptr &request);

    void prune_finalized_requests(bpt::ptime const &olderThan);
};


}
}

#endif /* PGREQUESTDAO_HPP */

