#include "InputQueueService.hpp"
#include <model/InputQueue.hpp>
#include <boost/interprocess/exceptions.hpp>

#include "appcontext.hpp"

#include "DAO/InputQueueDAO.hpp"
#include "DAO/ScopedTransaction.hpp"

#include "InterprocessReader.hpp"
#include "util/time_utils.hpp"


std::string svr::business::InputQueueService::make_queue_table_name(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution)
{
    std::string result = common::sanitize_db_table_name(common::formatter() <<
                                                                            common::C_input_queue_table_name_prefix << "_" << user_name << "_" << logical_name << "_"
                                                                            << resolution.total_seconds());
    std::transform(C_default_exec_policy, result.begin(), result.end(), result.begin(), ::tolower);
    LOG4_DEBUG("Returning " << result);
    return result;
}

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

datamodel::InputQueue_ptr InputQueueService::get_queue_metadata(const std::string &user_name, const std::string &logical_name, const time_duration &resolution)
{
    return input_queue_dao.get_queue_metadata(user_name, logical_name, resolution);
}

datamodel::InputQueue_ptr InputQueueService::get_queue_metadata(const std::string &input_queue_table_name)
{
    return input_queue_dao.get_queue_metadata(input_queue_table_name);
}


size_t
InputQueueService::get_count_from_start(const datamodel::InputQueue_ptr &p_input_queue, const boost::posix_time::ptime &time)
{
    return input_queue_dao.get_count_from_start(p_input_queue->get_table_name(), time);
}


svr::datamodel::DataRow::container
InputQueueService::load(
        const std::string &table_name,
        const bpt::ptime &time_from,
        const bpt::ptime &timeTo,
        size_t limit)
{
    return input_queue_dao.get_queue_data_by_table_name(table_name, time_from, timeTo, limit);
}


void
InputQueueService::load(datamodel::InputQueue &input_queue)
{
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_TRACE("Loading " << input_queue);

    data_row_container &data = input_queue.get_data();
    if (data.empty()) {
        if (input_queue.get_uses_fix_connection())
            input_queue.set_data(load_latest_from_mmf(input_queue, bpt::min_date_time));
        else
            input_queue.set_data(input_queue_dao.get_queue_data_by_table_name(
                    input_queue.get_table_name(), bpt::min_date_time, bpt::max_date_time, 0));
    } else {
        const auto new_data = input_queue.get_uses_fix_connection() ?
                              load_latest_from_mmf(input_queue, bpt::max_date_time) :
                              input_queue_dao.get_queue_data_by_table_name(
                                      input_queue.get_table_name(),
                                      data.back()->get_value_time() + input_queue.get_resolution(),
                                      bpt::max_date_time, 0);
        if (!new_data.empty() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
    }
}


void
InputQueueService::load(
        datamodel::InputQueue &input_queue,
        const bpt::time_period &range,
        const size_t limit)
{
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Getting up to " << limit << " rows during " << range << " from " << input_queue.get_table_name());

    data_row_container &data = input_queue.get_data();
    if (data.empty()) {
        if (input_queue.get_uses_fix_connection())
            input_queue.set_data(load_latest_from_mmf(input_queue, range.begin()));
        else
            input_queue.set_data(input_queue_dao.get_queue_data_by_table_name(input_queue.get_table_name(), range.begin(), range.end(), limit));
    } else {
        const auto new_data = input_queue.get_uses_fix_connection() ?
                              load_latest_from_mmf(input_queue, range.end()) :
                              input_queue_dao.get_queue_data_by_table_name(
                                      input_queue.get_table_name(), range.begin(), range.end(), limit);
        if (!new_data.empty() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
    }
}


data_row_container
InputQueueService::load_latest_from_mmf(
        const datamodel::InputQueue &input_queue,
        const bpt::ptime &last_time)
{
    data_row_container result;
#ifndef BUILD_WITHOUT_SVR_FIX
    if (not input_queue.get_uses_fix_connection()) return result;

    svr::fix::mm_file_reader::bid_ask_spread_container values;

    try {
        svr::fix::mm_file_reader reader(input_queue.get_logical_name());
        values = reader.read_new(last_time);
    }
    catch (std::runtime_error const &) {
        return result;
    }
    catch (boost::interprocess::interprocess_exception &) {
        return result;
    }

    if (last_time < values.front().time - input_queue.get_resolution())
        return result;

    for (const auto &bas: values)
        result.emplace_back(ptr<svr::datamodel::DataRow>(bas.time, bas.time, 0,
                                                         std::vector<double>{bas.ask_px, double(bas.ask_qty), bas.bid_px,
                                                                             double(bas.bid_qty)}));
#endif //BUILD_WITHOUT_SVR_FIX
    return result;
}


void
InputQueueService::load_latest(
        datamodel::InputQueue_ptr p_input_queue,
        const size_t limit,
        const bpt::ptime &last_time)
{
    // TODO: This method takes last_time inclusively, and the specification for that is not clear.
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Getting " << limit << " rows until " << last_time << " from " << p_input_queue->get_table_name());

    auto result = load_latest_from_mmf(*p_input_queue, last_time);
    if (result.empty()) result = input_queue_dao.get_latest_queue_data_by_table_name(p_input_queue->get_table_name(), limit, last_time);
    LOG4_DEBUG("Got " << result.size() << " rows successfully.");
    if (result.empty())
        return;
    if (p_input_queue->get_data().size()) {
        p_input_queue->get_data().erase(lower_bound_back(p_input_queue->get_data(), result.front()->get_value_time()), p_input_queue->get_data().end());
        p_input_queue->get_data().insert(p_input_queue->get_data().end(), result.begin(), result.end());
    } else {
        p_input_queue->set_data(result);
    }
}


datamodel::DataRow_ptr
InputQueueService::load_nth_last_row(const datamodel::InputQueue_ptr &input_queue, const size_t position, const bpt::ptime target_time)
{
    LOG4_DEBUG("Getting " << position << "th row before " << target_time << " from " << input_queue->get_table_name());
    return input_queue_dao.get_nth_last_row(input_queue->get_table_name(), position, target_time);
}


size_t InputQueueService::save(const datamodel::InputQueue_ptr &queue, const bpt::ptime &start_time)
{
    REJECT_NULLPTR(queue);
    REJECT_EMPTY_1(queue->get_data());

    return input_queue_dao.save_data(queue, start_time);
}


size_t InputQueueService::save(const datamodel::InputQueue_ptr &p_input_queue)
{
    LOG4_BEGIN();
    return input_queue_dao.save(p_input_queue);
    LOG4_END();
}

void InputQueueService::upsert_row_str(CRPTR(char) table_name, CRPTR(char) value_time, CRPTR(char) update_time, CRPTR(char) volume, CRPTR(char *) values, const uint16_t n_values)
{
    LOG4_BEGIN();
    input_queue_dao.upsert_row_str(table_name, value_time, update_time, volume, values, n_values);
    LOG4_END();
}

bool InputQueueService::exists(
        const std::string &user_name,
        const std::string &logical_name,
        const bpt::time_duration &resolution)
{
    return input_queue_dao.exists(user_name, logical_name, resolution);
}


int InputQueueService::remove(const datamodel::InputQueue_ptr &p_input_queue)
{
    REJECT_NULLPTR(p_input_queue);
    return input_queue_dao.remove(p_input_queue);
}

int InputQueueService::clear(const datamodel::InputQueue_ptr &p_input_queue)
{
    REJECT_NULLPTR(p_input_queue);
    return input_queue_dao.clear(p_input_queue);
}


datamodel::InputQueue_ptr
InputQueueService::clone_with_data(
        const datamodel::InputQueue_ptr &p_input_queue,
        const time_period &time_range,
        const size_t minimum_rows_count)
{
    auto result_queue = p_input_queue->clone_empty();
    result_queue->set_data(input_queue_dao.get_queue_data_by_table_name(p_input_queue->get_table_name(), time_range.begin(), time_range.end()));
    if (result_queue->size() < minimum_rows_count)
        THROW_EX_FS(insufficient_data,
                    "Not enough data in p_input_queue in time range " << time_range << " number of observations is lesser than " << minimum_rows_count);

    return result_queue;
}


data_row_container
InputQueueService::get_column_data(
        const datamodel::InputQueue &input_queue,
        const std::string &column_name)
{
    return get_column_data(input_queue, get_value_column_index(input_queue, column_name));
}


data_row_container
InputQueueService::get_column_data(const datamodel::InputQueue &input_queue, const size_t column_index)
{
    LOG4_BEGIN();
    const auto &column_data = input_queue.get_data();
    DataRow::container result(column_data.size());
    if (column_data.empty()) {
        LOG4_ERROR("Column data in " << input_queue.get_table_name() << " is empty.");
        return {};
    }
    if (column_data.front()->size() <= column_index) {
        LOG4_ERROR("Column index out of bounds " << column_index << ", columns in input queue " << column_data.front()->get_values().size());
        return {};
    }
    OMP_FOR_i(column_data.size()) {
        const auto r = *(column_data.cbegin() + i);
        if (!r)
            LOG4_ERROR("Invalid row at " << result.size() << ", skipping.");
        else
            result[i] = ptr<DataRow>(r->get_value_time(), r->get_update_time(), r->get_tick_volume(), std::vector{r->get_value(column_index)});
    }
    LOG4_END();
    return result;
}


std::deque<std::string>
InputQueueService::get_db_table_column_names(const datamodel::InputQueue_ptr &queue)
{
    const auto db_all_columns = input_queue_dao.get_db_table_column_names(queue);
    std::deque<std::string> result;
    auto iut = std::find_if(
            C_default_exec_policy, db_all_columns.cbegin(), db_all_columns.cend(), [](std::shared_ptr<std::string> const &col) { return *col == "tick_volume"; });
    if (iut == db_all_columns.cend()) return result;
    ++iut;
    for (; iut != db_all_columns.cend(); ++iut) result.emplace_back(**iut);
    return result;
}


size_t
InputQueueService::get_value_column_index(
        const datamodel::InputQueue &input_queue,
        const std::string &column_name)
{
    const auto &cols = input_queue.get_value_columns();
    const auto pos = std::find(C_default_exec_policy, cols.cbegin(), cols.cend(), column_name);
    if (pos == cols.cend()) THROW_EX_FS(std::invalid_argument, "Column " << column_name << " is not part of input queue " << input_queue.get_table_name());
    return size_t(pos - cols.cbegin());
}


datamodel::DataRow_ptr
InputQueueService::find_oldest_record(const datamodel::InputQueue_ptr &queue)
{
    REJECT_NULLPTR(queue);

    return input_queue_dao.find_oldest_record(queue);
}

datamodel::DataRow_ptr
InputQueueService::find_newest_record(const datamodel::InputQueue_ptr &queue)
{
    LOG4_BEGIN();
    REJECT_NULLPTR(queue);

    return input_queue_dao.find_newest_record(queue);
}


std::deque<datamodel::InputQueue_ptr>
InputQueueService::get_all_user_queues(const std::string &user_name)
{
    return input_queue_dao.get_all_user_queues(user_name);
}


std::deque<datamodel::InputQueue_ptr> InputQueueService::get_all_queues_with_sign(const bool uses_fix_connector)
{
    return input_queue_dao.get_all_queues_with_sign(uses_fix_connector);
}


svr::dao::OptionalTimeRange
InputQueueService::get_missing_hours(const datamodel::InputQueue_ptr &queue, svr::dao::TimeRange const &from_range) const
{
    return input_queue_dao.get_missing_hours(queue, from_range);
}


void InputQueueService::purge_missing_hours(datamodel::InputQueue_ptr const &queue)
{
    input_queue_dao.purge_missing_hours(queue);
}


boost::posix_time::ptime
InputQueueService::validate_decon_data(const datamodel::InputQueue &input_queue, const datamodel::DeconQueue &decon_queue)
{
    // max_date_time means don't decompose anything, min_date_time means decompose all data from the input queue
    LOG4_BEGIN();

    if (std::find(C_default_exec_policy, input_queue.get_value_columns().cbegin(), input_queue.get_value_columns().cend(),
                  decon_queue.get_input_queue_column_name()) == input_queue.get_value_columns().cend()) {
        LOG4_ERROR("Decon queue column " << decon_queue.get_input_queue_column_name() << " is missing in input queue " << input_queue);
        return boost::posix_time::max_date_time;
    }

    if (input_queue.empty()) return boost::posix_time::max_date_time;

    if (decon_queue.empty()) return boost::posix_time::min_date_time;

    if (input_queue.front()->get_value_time() > decon_queue.front()->get_value_time()) {
        LOG4_ERROR("Inconsistency between input queue " << input_queue << " and decon queue " << decon_queue << " detected.");
        return boost::posix_time::min_date_time;
    }

    if (input_queue.back()->get_value_time() < decon_queue.back()->get_value_time()) {
        LOG4_WARN("Ignoring inconsistency between input queue " << input_queue << " and decon queue " << decon_queue << " last input time " << input_queue.back()->get_value_time()
                                                                << " is before last decon time " << decon_queue.back()->get_value_time());
        return boost::posix_time::max_date_time;
    }

    if (input_queue.back()->get_value_time() == decon_queue.back()->get_value_time()
        && input_queue.front()->get_value_time() == decon_queue.front()->get_value_time()
        && input_queue.size() == decon_queue.size()) {
        LOG4_DEBUG("Input queue " << input_queue.get_table_name() << " and decon queue " << decon_queue.get_table_name() << " seem to be equal.");
        return boost::posix_time::max_date_time;
    }

#ifdef INDEPTH_COMPARE

    boost::posix_time::ptime missing_time = boost::posix_time::max_date_time;
    t_omp_lock missing_time_l;
    OMP_FOR_(decon_queue.size(),)
    for (auto range_iter = lower_bound(input_queue.get_data(), decon_queue.front()->get_value_time()); range_iter != input_queue.cend(); ++range_iter) {
        if ((**range_iter).get_value_time() > missing_time) continue;
        if (std::any_of(C_default_exec_policy, decon_queue.cbegin(), decon_queue.cend(),
                        [&range_iter](const auto &item) { return item->get_value_time() == (**range_iter).get_value_time(); }))
            continue;
        LOG4_DEBUG("Input row at " << (**range_iter).get_value_time() << " not found in decon queue " << decon_queue.get_input_queue_table_name() <<
                                   " " << decon_queue.get_input_queue_column_name());
        missing_time_l.set();
        if (missing_time > (**range_iter).get_value_time()) missing_time = (**range_iter).get_value_time();
        missing_time_l.unset();
    }

    LOG4_DEBUG("Compared input queue " << input_queue.get_table_name() << " with data from " << input_queue.front()->get_value_time() << " to " <<
                                       input_queue.back()->get_value_time() << " with decon queue with data from " << decon_queue.front()->get_value_time() << " to "
                                       << decon_queue.back()->get_value_time());

    return missing_time;

#else

    if (input_queue.back()->get_value_time() > decon_queue.back()->get_value_time()) {
        LOG4_DEBUG("New data " << input_queue.back()->get_value_time() << " in input queue " << input_queue.get_table_name() << " and decon queue " << decon_queue.get_table_name() <<
            " detected " << decon_queue.back()->get_value_time());
        return (**upper_bound(input_queue, decon_queue.back()->get_value_time())).get_value_time();
    }

    return bpt::max_date_time;

#endif
}


void
InputQueueService::prepare_queues(datamodel::Dataset &dataset)
{
    LOG4_DEBUG("Preparing input queue " << dataset.get_input_queue()->get_table_name());

    prepare_input_data(dataset);
    APP.iq_scaling_factor_service.prepare(dataset, true);
    const bool save_decon = getenv("SVRWAVE_SAVE_DECON_QUEUES") != nullptr;

#pragma omp parallel ADJ_THREADS(C_n_cpu)
#pragma omp single
    {
        OMP_TASKLOOP_1()
        for (const auto &p_ensemble: dataset.get_ensembles()) {
            DeconQueueService::prepare_decon(dataset, *dataset.get_input_queue(), *p_ensemble->get_decon_queue());
            std::map<InputQueue_ptr, DeconQueue_ptr> aux_decon_queues;
//            OMP_TASKLOOP_1() // Started crashing because ICPX
            for (const auto &p_ensemble_aux_decon: p_ensemble->get_aux_decon_queues())
                DeconQueueService::prepare_decon(dataset, *dataset.get_aux_input_queue(p_ensemble_aux_decon->get_input_queue_table_name()), *p_ensemble_aux_decon);
        }

        if (save_decon) {
            const auto decon_queues = dataset.get_decon_queues();
            OMP_TASKLOOP_1()
            for (const auto &p_ensemble: dataset.get_ensembles()) {
                APP.decon_queue_service.save(p_ensemble->get_decon_queue());
//                OMP_TASKLOOP_1()
                for (const auto &p_aux_decon_decon_queue: p_ensemble->get_aux_decon_queues())
                    APP.decon_queue_service.save(p_aux_decon_decon_queue);
            }
        }
    }

    LOG4_END();
}

void InputQueueService::prepare_input_data(datamodel::Dataset &dataset)
{
    LOG4_BEGIN();
#pragma omp parallel ADJ_THREADS(dataset.get_aux_input_queues().size() + 1)
#pragma omp single
    {
#pragma omp task
        APP.input_queue_service.load(*dataset.get_input_queue());
#pragma omp taskloop default(none) shared(dataset) mergeable grainsize(1)
        for (const auto &p_input_queue: dataset.get_aux_input_queues()) APP.input_queue_service.load(*p_input_queue);
    }
    LOG4_END();
}

#if 0
data_row_container
InputQueueService::shift_times_forward(const data_row_container &data, const bpt::time_duration &resolution)
{
    data_row_container new_data_cont;
    for (size_t ix = 0; ix < data.size() - 1; ++ix) {
        if (ix == 0) {
            auto new_data_row = ptr<DataRow>(*data.front());
            new_data_cont.insert({new_data_row->get_value_time() - resolution, new_data_row});
        } else {
            auto data_row = std::next(data.begin(), ix);
            auto data_row_next = std::next(data_row);
            auto new_data_row = ptr<DataRow>(*data_row_next->get());

            new_data_row->set_value_time(data_row->get()->get_value_time());
            new_data_cont.insert({new_data_row->get_value_time(), new_data_row});
        }
    }
    return new_data_cont;
}
#endif

bpt::ptime InputQueueService::adjust_time_on_grid(const InputQueue &queue, const bpt::ptime &value_time)
{
    static const auto epoch = bpt::from_time_t(0);
    const auto duration_since_epoch = value_time - epoch;
    const uint32_t seconds_since_epoch = duration_since_epoch.total_seconds();

    // check whether value_time may be present in two different timegrid slots
    // Hint: this can(should) be caught on object construction time.
    const uint32_t deviation_secs = queue.get_legal_time_deviation().total_seconds();
    const uint32_t resolution_secs = queue.get_resolution().total_seconds();
    if (resolution_secs / 2 < deviation_secs)
        THROW_EX_FS(std::invalid_argument,
                    "Invalid Resolution/Legal time deviation arguments! "
                    "Cannot calculate time on grid because value time may be valid on two different time slots!");

    uint32_t seconds_on_grid_since_epoch = 0;
    // check if the value_time can fit on the previous time grid slot
    if (seconds_since_epoch % resolution_secs <= deviation_secs) seconds_on_grid_since_epoch = seconds_since_epoch - seconds_since_epoch % resolution_secs;
        // else check if the value_time can fit on the next timegrid slot
    else if (((seconds_on_grid_since_epoch = seconds_since_epoch + deviation_secs) % resolution_secs) < deviation_secs)
        seconds_on_grid_since_epoch -= seconds_on_grid_since_epoch % resolution_secs;
    else
        return bpt::not_a_date_time;

    return epoch + bpt::seconds(seconds_on_grid_since_epoch);
}


void InputQueueService::add_row(data_row_container &data, const DataRow_ptr &p_row)
{
    if (data.size() && data.back()->get_value_time() <= p_row->get_value_time()) {
        auto it = lower_bound(data, p_row->get_value_time());
        if (it != data.cend()) {
            if ((**it).get_value_time() == p_row->get_value_time() && false) *it = p_row;
            else (void) data.insert(it, p_row);
        }
    }
    data.emplace_back(p_row);
}


bool InputQueueService::add_row(datamodel::InputQueue &queue, const datamodel::DataRow_ptr &p_row)
{
    REJECT_NULLPTR(p_row);
    LOG4_TRACE("Adding " << p_row->to_string());
    const auto adjusted_value_time = adjust_time_on_grid(queue, p_row->get_value_time());
    if (adjusted_value_time == bpt::not_a_date_time) {
        LOG4_WARN("The following row doesn't fit in the queue " << p_row->to_string());
        return false;
    }
    add_row(queue.get_data(), p_row);
    return true;
}


} /* namespace business */
} /* namespace svr */
