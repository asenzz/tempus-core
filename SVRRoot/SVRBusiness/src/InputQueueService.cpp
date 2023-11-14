#include <boost/interprocess/exceptions.hpp>

#include "appcontext.hpp"

#include "DAO/InputQueueDAO.hpp"
#include "DAO/ScopedTransaction.hpp"

#include "InterprocessReader.hpp"
#include "util/TimeUtils.hpp"

using namespace std;
using namespace bpt;
using namespace svr::common;
using namespace svr::datamodel;

namespace svr {
namespace business {

InputQueueService::InputQueueService(svr::dao::InputQueueDAO &input_queue_dao)
        : input_queue_dao(input_queue_dao)
{}

InputQueueService::~InputQueueService()
{}

InputQueue_ptr
InputQueueService::get_queue_metadata(
        const string &user_name,
        const string &logical_name,
        const time_duration &resolution)
{
    return input_queue_dao.get_queue_metadata(user_name, logical_name, resolution);
}

InputQueue_ptr
InputQueueService::get_queue_metadata(const string &input_queue_table_name)
{
    return input_queue_dao.get_queue_metadata(input_queue_table_name);
}

svr::datamodel::DataRow::container InputQueueService::get_queue_data(
        const string &table_name,
        const ptime &time_from,
        const ptime &time_to,
        size_t limit)
{
    PROFILE_EXEC_TIME(
            return input_queue_dao.get_queue_data_by_table_name(table_name, time_from, time_to, limit),
            "Getting queue data from DAO");
}

size_t
InputQueueService::get_count_from_start(const InputQueue_ptr &p_input_queue, const boost::posix_time::ptime &time)
{
    return input_queue_dao.get_count_from_start(p_input_queue->get_table_name(), time);
}

data_row_container
InputQueueService::get_latest_queue_data_from_mmf(
        InputQueue_ptr const &input_queue,
        const bpt::ptime &last_time)
{
    data_row_container result;
#ifndef BUILD_WITHOUT_SVR_FIX
    if (not input_queue->get_uses_fix_connection()) return result;

    svr::fix::mm_file_reader::bid_ask_spread_container values;

    try {
        svr::fix::mm_file_reader reader(input_queue->get_logical_name());
        values = reader.read_new(last_time);
    }
    catch (std::runtime_error const &) {
        return result;
    }
    catch (boost::interprocess::interprocess_exception &) {
        return result;
    }

    if (last_time < values.front().time - input_queue->get_resolution())
        return result;

    for (const auto &bas :values)
        result.push_back(std::make_shared<svr::datamodel::DataRow>(bas.time, bas.time, 0,
                                                                   std::vector<double>{bas.ask_px, double(bas.ask_qty), bas.bid_px,
                                                                    double(bas.bid_qty)}));
#endif //BUILD_WITHOUT_SVR_FIX
    return result;
}


data_row_container
InputQueueService::get_latest_queue_data(
        InputQueue_ptr const &input_queue,
        const size_t limit,
        const bpt::ptime &last_time)
{
    // TODO: This method takes last_time inclusively, and the specification for that is not clear.
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Getting " << limit << " rows until " << last_time << " from " << input_queue->get_table_name());

    auto result = get_latest_queue_data_from_mmf(input_queue, last_time);

    if (!result.empty()) goto __bail;

    result = input_queue_dao.get_latest_queue_data_by_table_name(input_queue->get_table_name(), limit, last_time);

    __bail:
    LOG4_DEBUG("Got " << result.size() << " rows successfully.");
    return result;
}

DataRow_ptr
InputQueueService::get_nth_last_row(const InputQueue_ptr &input_queue, const size_t position, const bpt::ptime target_time)
{
    LOG4_DEBUG("Getting " << position << "th row before " << target_time << " from " << input_queue->get_table_name());
    return input_queue_dao.get_nth_last_row(input_queue->get_table_name(), position, target_time);
}

data_row_container
InputQueueService::get_queue_data(
        const InputQueue_ptr &p_input_queue,
        const size_t tail_length,
        const bpt::time_period &range)
{
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Getting up to " << tail_length << " rows during " << range << " from " << p_input_queue->get_table_name());

    data_row_container data, result = get_latest_queue_data_from_mmf(p_input_queue, range.begin());
// TODO update for MMF
    if (!result.empty()) goto __bail;

    result = input_queue_dao.get_latest_queue_data_by_table_name(p_input_queue->get_table_name(), tail_length, range.begin());
    if (!result.empty() and result.back()->get_value_time() == range.begin()) result.pop_back();
    // Get range between last train and most current data
    data = input_queue_dao.get_queue_data_by_table_name(p_input_queue->get_table_name(), range.begin(), range.end());
    result.insert(result.end(), data.begin(), data.end());

__bail:
    if (result.empty())
        LOG4_DEBUG("No data found.");
    else
        LOG4_DEBUG("Got " << result.size() << " rows successfully, starting " << result.front()->get_value_time() << " until " << result.back()->get_value_time());
    return result;
}

long InputQueueService::save(const InputQueue_ptr &p_input_queue)
{
    return input_queue_dao.save(p_input_queue);
}


bool InputQueueService::exists(
        const std::string &user_name,
        const std::string &logical_name,
        const bpt::time_duration &resolution)
{
    return input_queue_dao.exists(user_name, logical_name, resolution);
}


int InputQueueService::remove(const InputQueue_ptr &p_input_queue)
{
    reject_nullptr(p_input_queue);
    return input_queue_dao.remove(p_input_queue);
}

int InputQueueService::clear(const InputQueue_ptr &p_input_queue)
{
    reject_nullptr(p_input_queue);
    return input_queue_dao.clear(p_input_queue);
}


/* TODO Optimize! */
bpt::ptime InputQueueService::adjust_time_on_grid(
        const InputQueue_ptr &p_input_queue,
        const bpt::ptime &value_time)
{
    ptime epoch = from_time_t(0);
    time_duration duration_since_epoch = value_time - epoch;

    // check whether value_time may be present in two different timegrid slots
    // Hint: this can(should) be caught on object construction time.
    if (p_input_queue->get_resolution().total_seconds() / 2
        < p_input_queue->get_legal_time_deviation().total_seconds()) {
        throw invalid_argument(
                "Invalid Resolution/Legal time deviation arguments!"
                "Cannot calculate time on grid because value_time may be valid on two different timegrid slots!");
    }

    long seconds_on_grid_since_epoch = 0;

    // check if the value_time can fit on the previous time grid slot
    if ((duration_since_epoch.total_seconds() % p_input_queue->get_resolution().total_seconds())
        <= p_input_queue->get_legal_time_deviation().total_seconds()) {
        seconds_on_grid_since_epoch = duration_since_epoch.total_seconds()
                                      - (duration_since_epoch.total_seconds() %
                                         p_input_queue->get_resolution().total_seconds());
        // else check if the value_time can fit on the next timegrid slot
    } else if (((
                        seconds_on_grid_since_epoch =
                                duration_since_epoch.total_seconds() +
                                p_input_queue->get_legal_time_deviation().total_seconds()
                ) % p_input_queue->get_resolution().total_seconds())
               < p_input_queue->get_legal_time_deviation().total_seconds()) {
        seconds_on_grid_since_epoch -= seconds_on_grid_since_epoch
                                       % p_input_queue->get_resolution().total_seconds();
    } else
        return not_a_date_time;

    return from_time_t(0) + seconds(seconds_on_grid_since_epoch);
}


bool
InputQueueService::add_row(
        InputQueue_ptr &p_input_queue,
        DataRow_ptr p_row,
        bool concatenate)
{
    reject_nullptr(p_input_queue);
    reject_nullptr(p_row);
    LOG4_DEBUG("Adding " << p_row->to_string());

    const bpt::ptime adjusted_value_time = adjust_time_on_grid(p_input_queue, p_row->get_value_time());
    if (adjusted_value_time == not_a_date_time) {
        LOG4_DEBUG("The following row doesn't fit in the queue: " << p_row->to_string());
        return false;
    }

    if (p_row->get_value_time() != adjusted_value_time) p_row->set_value_time(adjusted_value_time);

    auto &data = p_input_queue->get_data();
    if (!data.empty() && data.back()->get_value_time() <= p_row->get_value_time())
        data.erase(lower_bound(data, p_row->get_value_time()), data.end());
    data.emplace_back(p_row);
    return true;
}


InputQueue_ptr
InputQueueService::clone_with_data(
        const InputQueue_ptr &p_input_queue,
        const time_period &time_range,
        const size_t minimum_rows_count)
{
    auto result_queue = p_input_queue->clone_empty(); // TODO Why do we clone empty when we clone with data?
    result_queue->set_data(input_queue_dao.get_queue_data_by_table_name(
            p_input_queue->get_table_name(), time_range.begin(), time_range.end()));

    if (result_queue->get_data().size() < minimum_rows_count)
        THROW_EX_FS(insufficient_data,
                "Not enough data in inputQueue in time range " << time_range << " number of observations is lesser than " << minimum_rows_count);

    return result_queue;
}


data_row_container
InputQueueService::get_column_data(
        const InputQueue_ptr &queue,
        const string &column_name)
{
    LOG4_BEGIN();

    DataRow::container &column_data = queue->get_data();
    const size_t column_index = get_value_column_index(queue, column_name);
    DataRow::container result;

    if (column_data.empty()) {
        LOG4_ERROR("Column data is empty. Aborting.");
        return result;
    }

    if (column_data.front()->get_values().size() <= column_index || column_index < 0) {
        LOG4_ERROR("Column index out of bounds " << column_index << ", columns in input queue " <<
                                                 column_data.front()->get_values().size() << ". Aborting.");
        return result;
    }

    for (auto &row: column_data) {
        if (!row) {
            LOG4_ERROR("Invalid row. Skipping.");
            continue;
        }
        result.push_back(std::make_shared<DataRow>(
                DataRow(row->get_value_time(),
                        row->get_update_time(),
                        row->get_tick_volume(),
                        {row->get_values()[column_index]})));
    }

    LOG4_END();

    return result;
}

std::vector<std::string>
InputQueueService::get_db_table_column_names(const InputQueue_ptr &queue)
{
    const auto db_all_columns = input_queue_dao.get_db_table_column_names(queue);

    std::vector<std::string> result;

    auto iut = std::find_if(db_all_columns.begin(), db_all_columns.end(), [](std::shared_ptr<std::string> const &col)
    { return *col == "tick_volume"; });
    if (iut == db_all_columns.end())
        return result;

    iut += 1;

    result.reserve(std::distance(iut, db_all_columns.end()));

    for (auto ir = result.begin(); iut != db_all_columns.end(); ++iut, ++ir)
        result.push_back(**iut);

    return result;
}


size_t
InputQueueService::get_value_column_index(
        const InputQueue_ptr &p_input_queue,
        const std::string &column_name)
{
    const auto &cols = p_input_queue->get_value_columns();
    const auto pos = find(cols.begin(), cols.end(), column_name);
    if (pos == cols.end())
        THROW_EX_FS(std::invalid_argument, "Column " << column_name << " is not part of input queue " << p_input_queue->get_table_name());

    return (size_t) std::abs(std::distance(cols.begin(), pos));
}


DataRow_ptr
InputQueueService::find_oldest_record(const InputQueue_ptr &queue)
{
    reject_nullptr(queue);

    return input_queue_dao.find_oldest_record(queue);
}

DataRow_ptr
InputQueueService::find_newest_record(const InputQueue_ptr &queue)
{
    reject_nullptr(queue);

    return input_queue_dao.find_newest_record(queue);
}


std::vector<InputQueue_ptr>
InputQueueService::get_all_user_queues(const std::string &user_name)
{
    return input_queue_dao.get_all_user_queues(user_name);
}


std::vector<InputQueue_ptr> InputQueueService::get_all_queues_with_sign(const bool uses_fix_connector)
{
    return input_queue_dao.get_all_queues_with_sign(uses_fix_connector);
}


size_t InputQueueService::save_data(const InputQueue_ptr &queue, const bpt::ptime &start_time)
{
    reject_nullptr(queue);
    reject_empty(queue->get_data());

    return input_queue_dao.save_data(queue, start_time);
}


svr::dao::OptionalTimeRange
InputQueueService::get_missing_hours(const InputQueue_ptr &queue, svr::dao::TimeRange const &fromRange) const
{
    return input_queue_dao.get_missing_hours(queue, fromRange);
}


void InputQueueService::purge_missing_hours(InputQueue_ptr const &queue)
{
    input_queue_dao.purge_missing_hours(queue);
}


boost::posix_time::ptime
InputQueueService::compare_to_decon_queue(
        const InputQueue_ptr &p_input_queue,
        const DeconQueue_ptr &p_decon_queue)
{
    // max_date_time means don't decompose anything, min_date_time means decompose all data from the input queue
    LOG4_BEGIN();
    if (p_input_queue->get_data().empty())
        return boost::posix_time::max_date_time;

    if (p_decon_queue->get_data().empty())
        return boost::posix_time::min_date_time;

    if (p_input_queue->is_tick_queue())
        return p_input_queue->get_data().front()->get_value_time() > p_decon_queue->get_data().front()->get_value_time() or p_input_queue->get_data().rbegin()->get()->get_value_time() < p_decon_queue->get_data().rbegin()->get()->get_value_time() ? boost::posix_time::min_date_time : boost::posix_time::max_date_time;

    if (std::find(p_input_queue->get_value_columns().begin(), p_input_queue->get_value_columns().end(), p_decon_queue->get_input_queue_column_name()) == p_input_queue->get_value_columns().end()) {
        LOG4_ERROR("Input queue column " << p_decon_queue->get_input_queue_column_name().c_str() << " is missing in " << deep_to_string(p_input_queue->get_value_columns()));
        return boost::posix_time::max_date_time;
    }
    auto range_iter = lower_bound(p_input_queue->get_data(), p_decon_queue->get_data().back()->get_value_time());
    const auto end_range_iter = p_input_queue->get_data().end();
    boost::posix_time::ptime missing_time = boost::posix_time::max_date_time;
    for (auto prev_decon_found = p_decon_queue->get_data().begin(); range_iter != end_range_iter; ++range_iter) {
        const auto find_res = std::find_if(prev_decon_found, p_decon_queue->get_data().end(), [&](const auto &item) {
            return item->get_value_time() == range_iter->get()->get_value_time();});
        if (find_res == p_decon_queue->get_data().end()) {
            LOG4_DEBUG("Input row at " << range_iter->get()->get_value_time() << " not found in decon queue " << p_decon_queue->get_input_queue_table_name() <<
                                       " " << p_decon_queue->get_input_queue_column_name());
            missing_time = range_iter->get()->get_value_time();
            break;
        } else {
            prev_decon_found = find_res;
        }
    }

    LOG4_DEBUG("Compared input queue " << p_input_queue->get_table_name() << " with data from " << p_input_queue->get_data().front()->get_value_time() << " to " <<
            p_input_queue->get_data().back()->get_value_time() << " with decon queue with data from " << p_decon_queue->get_data().front()->get_value_time() << " to " << p_decon_queue->get_data().back()->get_value_time());

    return missing_time;
}


void
InputQueueService::prepare_queues(
        Dataset_ptr &p_dataset,
        const bool trim /* false */)
{
    LOG4_DEBUG("Preparing input queue " << p_dataset->get_input_queue()->get_table_name());

    // First prepare input data
    prepare_input_data(p_dataset);
    bpt::ptime first_new_row_time = bpt::max_date_time;
#pragma omp parallel for
    for (size_t ix = 0; ix < p_dataset->get_ensembles().size(); ++ix) {
        auto res = compare_to_decon_queue(p_dataset->get_input_queue(), p_dataset->get_ensemble(ix)->get_decon_queue());
        for (size_t q_ix = 0; q_ix < p_dataset->get_aux_input_queues().size(); ++q_ix) {
            auto res_aux = compare_to_decon_queue(p_dataset->get_aux_input_queue(0), p_dataset->get_ensemble(ix)->get_aux_decon_queue(0));
            if (res_aux < res) res = res_aux;
        }
#pragma omp critical(prepare_queues)
        {
            if (res < first_new_row_time)
                first_new_row_time = res;
        }
    }
    if (first_new_row_time == boost::posix_time::max_date_time) {
        LOG4_DEBUG("Decon queues already have the latest data, bailing.");
        return;
    }

#ifdef DEBUG_PREPARE
    {
        static size_t call_counter;
        std::stringstream ss;
        using namespace svr::common;
        ss <<
           "input_queue_" << p_dataset->get_input_queue()->get_table_name() <<
           "_call_" << call_counter++ << ".out";
        std::ofstream of(ss.str());
        of.precision(std::numeric_limits<double>::max_digits10);
        for (const auto &row: p_dataset->get_input_queue()->get_data()) {
            of << row->get_value_time() << ", ";
            for (const auto value: row.get()->get_values()) of << value << ", ";
            of << "\n";
        }
        of.flush();
    }
#endif
    {
        std::deque<std::thread> th;
        th.emplace_back([&]() { APP.decon_queue_service.deconstruct(p_dataset->get_input_queue(), p_dataset);});
        const auto aux_input_queues = p_dataset->get_aux_input_queues();
        for (const auto &aux_input_queue: aux_input_queues)
            th.emplace_back([&](const InputQueue_ptr &p_input_queue_) { APP.decon_queue_service.deconstruct(p_input_queue_, p_dataset); }, aux_input_queue);
        for (auto &t: th) t.join();
    }

    if (false && getenv("BACKTEST")) {
        std::deque<std::thread> th;
        const auto decon_queues = p_dataset->get_decon_queues();
        for (const auto &p_ensemble: p_dataset->get_ensembles()) {
            th.emplace_back([&](const Ensemble_ptr &p_ensemble_){APP.decon_queue_service.save(p_ensemble_->get_decon_queue(), first_new_row_time);}, p_ensemble);
            const auto aux_decon_queues = p_ensemble->get_aux_decon_queues();
            for (const auto &p_aux_decon_decon_queue: aux_decon_queues)
                th.emplace_back([&](const DeconQueue_ptr &p_aux_decon_decon_queue_){APP.decon_queue_service.save(p_aux_decon_decon_queue_, first_new_row_time);}, p_aux_decon_decon_queue);
        }

        for (auto &t: th) t.join();
    }
    LOG4_END();
}

void InputQueueService::prepare_input_data(Dataset_ptr &p_dataset, const InputQueue_ptr &p_input_queue)
{
    LOG4_BEGIN();

    if (!p_input_queue) LOG4_THROW("Something is wrong, input queue nil!");

    data_row_container latest_data = APP.input_queue_service.get_queue_data(
            p_input_queue->get_table_name(),
            p_input_queue->get_data().empty() ? bpt::min_date_time : p_input_queue->get_data().back()->get_value_time() + p_input_queue->get_resolution());

    p_input_queue->update_data(latest_data, true);

    if (!latest_data.empty() && !p_input_queue->get_data().empty())
        LOG4_DEBUG("Input queue " << p_input_queue->get_table_name() << ", begin time " << p_input_queue->get_data().front()->get_value_time() <<
            ", end time " << p_input_queue->get_data().back()->get_value_time() << ", latest data size " << latest_data.size() <<
            ", value columns count " << p_input_queue->get_data().front()->get_values().size() << ", input queue data new size " << p_input_queue->get_data().size());
    else
        LOG4_ERROR("Returned data is empty.");
}

void InputQueueService::prepare_input_data(Dataset_ptr &p_dataset)
{
    LOG4_BEGIN();
    std::vector<InputQueue_ptr> input_queues = p_dataset->get_aux_input_queues();
    input_queues.emplace_back(p_dataset->get_input_queue());
    __pxt_pfor_i(0, input_queues.size(), prepare_input_data(p_dataset, input_queues[i]))
    LOG4_END();
}

#if 0
data_row_container
InputQueueService::shift_times_forward(const data_row_container &data, const bpt::time_duration &resolution)
{
    data_row_container new_data_cont;
    for (size_t ix = 0; ix < data.size() - 1; ++ix) {
        if (ix == 0) {
            auto new_data_row = std::make_shared<DataRow>(*data.begin()->get());
            new_data_cont.insert({new_data_row->get_value_time() - resolution, new_data_row});
        } else {
            auto data_row = std::next(data.begin(), ix);
            auto data_row_next = std::next(data_row);
            auto new_data_row = std::make_shared<DataRow>(*data_row_next->get());
            new_data_row->set_value_time(data_row->get()->get_value_time());
            new_data_cont.insert({new_data_row->get_value_time(), new_data_row});
        }
    }
    return new_data_cont;
}
#endif

} /* namespace business */
} /* namespace svr */
