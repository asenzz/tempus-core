#ifndef ASYNCREQUESTDAO_HPP
#define ASYNCREQUESTDAO_HPP

#include <DAO/RequestDAO.hpp>

namespace svr {
namespace dao {

class AsyncRequestDAO : public RequestDAO {
public:
    explicit AsyncRequestDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    ~AsyncRequestDAO() override;

    bigint get_next_id() override;

    bigint get_next_result_id() override;

    int save(const datamodel::MultivalRequest_ptr &request) override;

    bool exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
                                    const bpt::time_duration &resolution, const std::string &value_columns) override;

    bigint make_request(const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
                      const std::string &resolution_str, const std::string &value_columns) override;

    datamodel::MultivalRequest_ptr
    get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end) override;

    datamodel::MultivalRequest_ptr get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution, const std::string &value_columns) override;

    datamodel::MultivalRequest_ptr get_latest_multival_request(const std::string &user_name, const bigint dataset_id) override;

    virtual std::deque<datamodel::MultivalRequest_ptr> get_active_multival_requests(const std::string &user_name, const bigint dataset_id) override;

    std::deque<datamodel::MultivalResponse_ptr> get_multival_results(const std::string &user_name, const bigint dataset_id,
                                                                     const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution) override;

    std::deque<datamodel::MultivalResponse_ptr>
    get_multival_results_str(const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end,
                         const std::string &resolution) override;

    std::deque<datamodel::MultivalResponse_ptr> get_multival_results_column(
            const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                                                            const bpt::time_duration &resolution) override;

    int save(const datamodel::MultivalResponse_ptr &p_response) override;

    int force_finalize(const datamodel::MultivalRequest_ptr &p_request) override;

    void prune_finalized_requests(bpt::ptime const &older_than) override;

private:
    struct AsyncImpl;
    AsyncImpl &pImpl;
};

}
}
#endif /* ASYNCREQUESTDAO_HPP */

