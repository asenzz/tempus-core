#ifndef ASYNCREQUESTDAO_HPP
#define ASYNCREQUESTDAO_HPP

#include <DAO/RequestDAO.hpp>

namespace svr{
namespace dao{

class AsyncRequestDAO : public RequestDAO
{
public:
    explicit AsyncRequestDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncRequestDAO();

    bigint get_next_id();
    bigint get_next_result_id();

    int remove(const MultivalRequest_ptr& request);
    int save(const MultivalRequest_ptr& request);

    MultivalRequest_ptr get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end);

    MultivalRequest_ptr get_multival_request(const std::string &user_name, bigint dataset_id,
                                             const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns);

    MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, bigint dataset_id);

    virtual std::vector<MultivalRequest_ptr> get_active_multival_requests(const std::string &user_name, bigint dataset_id, std::string const & inputQueueName);

    std::vector<MultivalResponse_ptr> get_multival_results(const std::string &user_name, bigint dataset_id,
                                const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution);

    std::vector<MultivalResponse_ptr> get_multival_results_column(const std::string &user_name, const std::string &column_name,
            bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution);

    int save(const MultivalResponse_ptr &p_response);

    int force_finalize(const MultivalRequest_ptr &p_request);

    void prune_finalized_requests(bpt::ptime const &older_than);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

} }
#endif /* ASYNCREQUESTDAO_HPP */

