#pragma once

#include <memory>
#include <vector>
#include <csignal>
#include <error.h>
#include "appcontext.hpp"
#include "model/DataRow.hpp"
#include "fast_cvmd.hpp"


namespace svr {
namespace dao {
class DeconQueueDAO;
}

namespace datamodel {
class DeconQueue;

class InputQueue;

class Dataset;

using DeconQueue_Ptr = std::shared_ptr<DeconQueue>;
using InputQueue_ptr = std::shared_ptr<InputQueue>;
using Dataset_ptr = std::shared_ptr<Dataset>;
}

namespace business {
constexpr unsigned C_mirror_fade_in = 100;

typedef enum class recon_type : uint
{
    ADDITIVE = 0,
    MULTIPLICATIVE = 1,
    WAVELET = 2,
    FFT = 3,
    number_of_recon_types = 4 // end of enum = invalid type
} recon_type_e, *recon_type_e_ptr;


class DeconQueueService
{
    svr::dao::DeconQueueDAO &decon_queue_dao;

public:
    explicit DeconQueueService(svr::dao::DeconQueueDAO &decon_queue_dao);

    datamodel::DeconQueue_ptr get_by_table_name(const std::string &table_name) const;

    datamodel::DeconQueue_ptr get_by_table_name(const std::string &input_queue_table_name, const bigint dataset_id, const std::string &input_queue_column_name) const;

    void load_latest(
        datamodel::DeconQueue &decon_queue,
        const bpt::ptime &time_to = bpt::max_date_time,
        const size_t limit = 0);

    void load(
        datamodel::DeconQueue &decon_queue,
        const bpt::ptime &time_from = bpt::min_date_time,
        const bpt::ptime &time_to = bpt::max_date_time,
        const size_t limit = 0);

    static std::deque<datamodel::DeconQueue_ptr>
    extract_copy_data(const datamodel::Dataset_ptr &p_dataset, const boost::posix_time::time_period &period);

    void save(const datamodel::DeconQueue_ptr &p_decon_queue, boost::posix_time::ptime start_time = boost::posix_time::min_date_time) const;

    bool exists(const datamodel::DeconQueue_ptr &p_decon_queue) const;

    bool exists(const std::string &decon_queue_table_name) const;

    int32_t remove(const datamodel::DeconQueue_ptr &p_decon_queue) const;

    int32_t clear(const datamodel::DeconQueue_ptr &p_decon_queue);

    int64_t count(const datamodel::DeconQueue_ptr &decon_queue);

    static datamodel::DeconQueue_ptr find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues, const std::string &input_queue_table_name, const std::string &input_queue_column_name);

    static const datamodel::DeconQueue_ptr &find_decon_queue(const std::deque<datamodel::DeconQueue_ptr> &decon_queues, const std::string &decon_queue_table_name);

    datamodel::DeconQueue_ptr deconstruct(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, const std::string &column_name);

    static void deconstruct(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue);

    static void reconstruct(const datamodel::datarow_range &decon, const recon_type_e type, datamodel::data_row_container &recon, const datamodel::t_iqscaler &iq_unscaler);

    static datamodel::data_row_container reconstruct(const svr::datamodel::datarow_range &data, const recon_type_e type, const datamodel::t_iqscaler &unscaler);

    static std::vector<double> get_actual_values(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &target_iter);

    static void prepare_decon(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue);

    static void dummy_decon(const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue, const uint16_t levix, const uint16_t levct, const datamodel::t_iqscaler &iq_scaler);

    static std::string make_queue_table_name(const std::string &input_queue_table_name_, const bigint dataset_id_, const std::string &input_queue_column_name_);

    static void mirror_tail(const datamodel::datarow_crange &input, const size_t needed_data_ct, std::vector<double> &tail, const unsigned in_colix);
};
} /* namespace business */
} /* namespace svr */
