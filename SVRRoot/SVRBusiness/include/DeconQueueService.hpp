#pragma once

#include <memory>
#include <vector>

#include "model/DataRow.hpp"
#include "fast_cvmd.hpp"


#define PART_CVMD_COUNT 1000000 // Last N samples of the input queue used for calculating frequencies


namespace svr { namespace dao { class DeconQueueDAO; }}

namespace svr {
namespace datamodel {
class DeconQueue;

class InputQueue;

class Dataset;
}
}
using DeconQueue_ptr = std::shared_ptr<svr::datamodel::DeconQueue>;
using InputQueue_ptr = std::shared_ptr<svr::datamodel::InputQueue>;
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;

namespace svr { namespace business { class InputQueueService; }}

namespace svr {
namespace business {

class DeconQueueService
{
    svr::dao::DeconQueueDAO &decon_queue_dao;
    svr::business::InputQueueService &input_queue_service;

public:
    DeconQueueService(svr::dao::DeconQueueDAO &deconQueueDao,
                      svr::business::InputQueueService &input_queue_service)
            :
            decon_queue_dao(deconQueueDao),
            input_queue_service(input_queue_service)
    {}

    DeconQueue_ptr get_by_table_name(const std::string &table_name);

    DeconQueue_ptr get_queue_metadata(const std::string &table_name);

    size_t load_latest_decon_data(const DeconQueue_ptr &p_decon_queue, const bpt::ptime &time_to, size_t limit);

    size_t load_decon_data(
            const DeconQueue_ptr &decon_queue, const bpt::ptime &time_from = bpt::min_date_time,
            const bpt::ptime &time_to = bpt::max_date_time, const size_t limit = 0);

    static std::vector<DeconQueue_ptr>
    extract_copy_data(const Dataset_ptr &p_dataset, const boost::posix_time::time_period &period);

    void save(const DeconQueue_ptr &p_decon_queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    bool exists(const DeconQueue_ptr &p_decon_queue);

    bool exists(const std::string &decon_queue_table_name);

    int remove(const DeconQueue_ptr &p_decon_queue);

    int clear(const DeconQueue_ptr &p_decon_queue);

    long count(const DeconQueue_ptr &decon_queue);

    static const DeconQueue_ptr &find_decon_queue(
            const std::vector<DeconQueue_ptr> &decon_queues,
            const std::string &input_queue_table_name,
            const std::string &input_queue_column_name);

    static const DeconQueue_ptr &find_decon_queue(const std::vector<DeconQueue_ptr> &decon_queues, const std::string &decon_queue_table_name);

    std::vector<DeconQueue_ptr>
    deconstruct(
            const InputQueue_ptr &p_input_queue,
            const Dataset_ptr &p_dataset,
            const bool get_data_from_database = false);

    DeconQueue_ptr
    deconstruct(
            const InputQueue_ptr &p_input_queue,
            const Dataset_ptr &p_dataset,
            const std::string &column_name,
            const bool get_data_from_database = false);

    static void deconstruct(
            const InputQueue_ptr &p_input_queue,
            const Dataset_ptr &p_dataset,
            const std::string &column_name,
            svr::datamodel::DataRow::container &decon_out,
            const bool get_data_from_database = true);

    static void deconstruct_ticks(
            const InputQueue_ptr &p_input_queue,
            const Dataset_ptr &p_dataset,
            const std::string &column_name,
            svr::datamodel::DataRow::container &decon_data);

#if 0
    void
    deconstruct_batch(
            const InputQueue_ptr &p_input_queue,
            const Dataset_ptr &p_dataset,
            const std::string &column_name,
            svr::datamodel::DataRow::container &decon_data);
#endif

    static data_row_container
    reconstruct(
            const svr::datamodel::datarow_range &data,
            const std::string &transformation_name,
            const size_t n_decomposition_levels);

    static void
    reconstruct(
            const svr::datamodel::datarow_range &decon_queue,
            const std::string &transformation_name,
            const size_t n_decomposition_levels,
            data_row_container &output);

    static std::vector<double>
    get_actual_values(
            const data_row_container &data,
            const data_row_container::const_iterator &target_iter);

private:
    static void
    copy_decon_data_to_container(
            datamodel::DataRow::container &decon_out,
            const datamodel::DataRow::container &input_data,
            std::vector<std::vector<double>> &decon,
            const boost::posix_time::time_duration &resolution);

    static void
    copy_recon_frame_to_container(
            const svr::datamodel::DataRow::container &decon_data,
            const std::set<boost::posix_time::ptime> &times,
            svr::datamodel::DataRow::container &recon_data,
            const std::vector<double> &recon_out,
            size_t limit = std::numeric_limits<size_t>::max());

    void
    deconstruct_cvmd(
            const Dataset_ptr &p_dataset, const InputQueue_ptr &p_input_queue, const std::string &column_name,
            const std::vector<double> &column_values,
            std::vector<std::vector<double>> &raw_deconstructed_data,
            const bool do_batch = false) const;

public: size_t test_start_cvmd_pos = 0;
};

} /* namespace business */
} /* namespace svr */

using DeconQueueService_ptr = std::shared_ptr<svr::business::DeconQueueService>;

