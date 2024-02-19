#include "InputQueueService.hpp"
#include <model/InputQueue.hpp>
#include <boost/interprocess/exceptions.hpp>

#include "appcontext.hpp"

#include "DAO/InputQueueDAO.hpp"
#include "DAO/ScopedTransaction.hpp"

#include "InterprocessReader.hpp"
#include "util/TimeUtils.hpp"


std::string svr::business::InputQueueService::make_queue_table_name(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution)
{
    std::string result = common::sanitize_db_table_name(
            common::C_input_queue_table_name_prefix + "_" + user_name + "_" + logical_name + "_" + std::to_string(resolution.total_seconds()));
    std::transform(std::execution::par_unseq, result.begin(), result.end(), result.begin(), tolower);
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

datamodel::InputQueue_ptr
InputQueueService::get_queue_metadata(
        const std::string &user_name,
        const std::string &logical_name,
        const time_duration &resolution)
{
    return input_queue_dao.get_queue_metadata(user_name, logical_name, resolution);
}

datamodel::InputQueue_ptr
InputQueueService::get_queue_metadata(const std::string &input_queue_table_name)
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
InputQueueService::load(const datamodel::InputQueue_ptr &p_input_queue)
{
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Loading " << *p_input_queue);

    data_row_container &data = p_input_queue->get_data();
    if (data.empty()) {
        if (p_input_queue->get_uses_fix_connection())
            p_input_queue->set_data(load_latest_from_mmf(p_input_queue, bpt::min_date_time));
        else
            p_input_queue->set_data(input_queue_dao.get_queue_data_by_table_name(
                    p_input_queue->get_table_name(), bpt::min_date_time, bpt::max_date_time, 0));
    } else {
        const auto new_data = p_input_queue->get_uses_fix_connection() ?
                              load_latest_from_mmf(p_input_queue, bpt::max_date_time) :
                              input_queue_dao.get_queue_data_by_table_name(
                                      p_input_queue->get_table_name(),
                                      data.back()->get_value_time() + p_input_queue->get_resolution(),
                                      bpt::max_date_time, 0);
        if (!new_data.empty() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
    }
}


void
InputQueueService::load(
        const datamodel::InputQueue_ptr &p_input_queue,
        const bpt::time_period &range,
        const size_t limit)
{
    // Because of that, callers that want to exclude this last time must make sure to subtract a small amount of time from that parameter.
    LOG4_DEBUG("Getting up to " << limit << " rows during " << range << " from " << p_input_queue->get_table_name());

    data_row_container &data = p_input_queue->get_data();
    if (data.empty()) {
        if (p_input_queue->get_uses_fix_connection())
            p_input_queue->set_data(load_latest_from_mmf(p_input_queue, range.begin()));
        else
            p_input_queue->set_data(input_queue_dao.get_queue_data_by_table_name(p_input_queue->get_table_name(), range.begin(), range.end(), limit));
    } else {
        const auto new_data = p_input_queue->get_uses_fix_connection() ?
                              load_latest_from_mmf(p_input_queue, range.end()) :
                              input_queue_dao.get_queue_data_by_table_name(
                                      p_input_queue->get_table_name(), range.begin(), range.end(), limit);
        if (!new_data.empty() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
    }
}


data_row_container
InputQueueService::load_latest_from_mmf(
        datamodel::InputQueue_ptr const &input_queue,
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

    for (const auto &bas: values)
        result.emplace_back(std::make_shared<svr::datamodel::DataRow>(bas.time, bas.time, 0,
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

    auto result = load_latest_from_mmf(p_input_queue, last_time);
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
    reject_nullptr(queue);
    reject_empty(queue->get_data());

    return input_queue_dao.save_data(queue, start_time);
}


size_t InputQueueService::save(const datamodel::InputQueue_ptr &p_input_queue)
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


int InputQueueService::remove(const datamodel::InputQueue_ptr &p_input_queue)
{
    reject_nullptr(p_input_queue);
    return input_queue_dao.remove(p_input_queue);
}

int InputQueueService::clear(const datamodel::InputQueue_ptr &p_input_queue)
{
    reject_nullptr(p_input_queue);
    return input_queue_dao.clear(p_input_queue);
}


/* TODO Optimize! */
bpt::ptime InputQueueService::adjust_time_on_grid(
        const datamodel::InputQueue_ptr &p_input_queue,
        const bpt::ptime &value_time)
{
    ptime epoch = from_time_t(0);
    time_duration duration_since_epoch = value_time - epoch;

    // check whether value_time may be present in two different timegrid slots
    // Hint: this can(should) be caught on object construction time.
    if (p_input_queue->get_resolution().total_seconds() / 2
        < p_input_queue->get_legal_time_deviation().total_seconds()) {
        throw std::invalid_argument(
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
        datamodel::InputQueue_ptr &p_input_queue,
        datamodel::DataRow_ptr p_row,
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


datamodel::InputQueue_ptr
InputQueueService::clone_with_data(
        const datamodel::InputQueue_ptr &p_input_queue,
        const time_period &time_range,
        const size_t minimum_rows_count)
{
    auto result_queue = p_input_queue->clone_empty();
    result_queue->set_data(input_queue_dao.get_queue_data_by_table_name(
            p_input_queue->get_table_name(), time_range.begin(), time_range.end()));

    if (result_queue->size() < minimum_rows_count)
        THROW_EX_FS(insufficient_data,
                    "Not enough data in inputQueue in time range " << time_range << " number of observations is lesser than " << minimum_rows_count);

    return result_queue;
}


data_row_container
InputQueueService::get_column_data(
        const datamodel::InputQueue_ptr &queue,
        const std::string &column_name)
{
    return get_column_data(queue, get_value_column_index(queue, column_name));
}


data_row_container
InputQueueService::get_column_data(const datamodel::InputQueue_ptr &queue, const size_t column_index)
{
    LOG4_BEGIN();

    const DataRow::container &column_data = queue->get_data();
    DataRow::container result(column_data.size());

    if (column_data.empty()) {
        LOG4_ERROR("Column data in " << queue->get_table_name() << " is empty. Aborting.");
        return {};
    }

    if (column_data.front()->get_values().size() <= column_index || column_index < 0) {
        LOG4_ERROR("Column index out of bounds " << column_index << ", columns in input queue " << column_data.front()->get_values().size());
        return {};
    }

#pragma omp parallel for num_threads(adj_threads(column_data.size()))
    for (size_t r_ix = 0; r_ix < column_data.size(); ++r_ix) {
        const auto r = *(column_data.begin() + r_ix);
        if (!r) {
            LOG4_ERROR("Invalid row at " << result.size() << ", skipping.");
            continue;
        }
        result[r_ix] = std::make_shared<DataRow>(r->get_value_time(), r->get_update_time(), r->get_tick_volume(), std::vector{r->get_value(column_index)});
    }

    LOG4_END();

    return result;
}


std::deque<std::string>
InputQueueService::get_db_table_column_names(const datamodel::InputQueue_ptr &queue)
{
    const auto db_all_columns = input_queue_dao.get_db_table_column_names(queue);

    std::deque<std::string> result;

    auto iut = std::find_if(db_all_columns.begin(), db_all_columns.end(), [](std::shared_ptr<std::string> const &col) { return *col == "tick_volume"; });
    if (iut == db_all_columns.end()) return result;

    iut += 1;
    for (auto ir = result.begin(); iut != db_all_columns.end(); ++iut, ++ir) result.emplace_back(**iut);

    return result;
}


size_t
InputQueueService::get_value_column_index(
        const datamodel::InputQueue_ptr &p_input_queue,
        const std::string &column_name)
{
    const auto &cols = p_input_queue->get_value_columns();
    const auto pos = find(cols.begin(), cols.end(), column_name);
    if (pos == cols.end())
        THROW_EX_FS(std::invalid_argument, "Column " << column_name << " is not part of input queue " << p_input_queue->get_table_name());

    return (size_t) std::distance(cols.begin(), pos);
}


datamodel::DataRow_ptr
InputQueueService::find_oldest_record(const datamodel::InputQueue_ptr &queue)
{
    reject_nullptr(queue);

    return input_queue_dao.find_oldest_record(queue);
}

datamodel::DataRow_ptr
InputQueueService::find_newest_record(const datamodel::InputQueue_ptr &queue)
{
    reject_nullptr(queue);

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
InputQueueService::get_missing_hours(const datamodel::InputQueue_ptr &queue, svr::dao::TimeRange const &fromRange) const
{
    return input_queue_dao.get_missing_hours(queue, fromRange);
}


void InputQueueService::purge_missing_hours(datamodel::InputQueue_ptr const &queue)
{
    input_queue_dao.purge_missing_hours(queue);
}


boost::posix_time::ptime
InputQueueService::compare_to_decon_queue(
        const datamodel::InputQueue_ptr &p_input_queue,
        const datamodel::DeconQueue_ptr &p_decon_queue)
{
    // max_date_time means don't decompose anything, min_date_time means decompose all data from the input queue
    LOG4_BEGIN();
    if (p_input_queue->get_data().empty())
        return boost::posix_time::max_date_time;

    if (p_decon_queue->get_data().empty())
        return boost::posix_time::min_date_time;

    if (p_input_queue->front()->get_value_time() > p_decon_queue->front()->get_value_time()) {
        LOG4_ERROR("Inconsistency between input queue " << *p_input_queue << " and decon queue " << *p_decon_queue << " detected.");
        return boost::posix_time::min_date_time;
    }

    if (std::find(std::execution::par_unseq, p_input_queue->get_value_columns().begin(), p_input_queue->get_value_columns().end(),
                  p_decon_queue->get_input_queue_column_name()) == p_input_queue->get_value_columns().end()) {
        LOG4_ERROR("Input queue column " << p_decon_queue->get_input_queue_column_name() << " is missing in " << *p_input_queue);
        return boost::posix_time::max_date_time;
    }

    boost::posix_time::ptime missing_time{boost::posix_time::max_date_time};
#pragma omp parallel for num_threads(adj_threads(p_decon_queue->size()))
    for (auto range_iter = lower_bound(p_input_queue->get_data(), p_decon_queue->front()->get_value_time()); range_iter != p_input_queue->end(); ++range_iter) {
        if (std::any_of(
                std::execution::par_unseq, p_decon_queue->begin(), p_decon_queue->end(),
                [&range_iter](const auto &item) { return item->get_value_time() == range_iter->get()->get_value_time(); }))
            continue;
        LOG4_DEBUG("Input row at " << range_iter->get()->get_value_time() << " not found in decon queue " << p_decon_queue->get_input_queue_table_name() <<
                                   " " << p_decon_queue->get_input_queue_column_name());
#pragma omp critical
        {
            if (missing_time > range_iter->get()->get_value_time())
                missing_time = range_iter->get()->get_value_time();
        }
    }

    LOG4_DEBUG("Compared input queue " << p_input_queue->get_table_name() << " with data from " << p_input_queue->front()->get_value_time() << " to " <<
                                       p_input_queue->back()->get_value_time() << " with decon queue with data from " << p_decon_queue->front()->get_value_time()
                                       << " to " <<
                                       p_decon_queue->back()->get_value_time());

    return missing_time;
}


void
InputQueueService::prepare_queues(
        datamodel::Dataset_ptr &p_dataset,
        const bool trim /* false */)
{
    LOG4_DEBUG("Preparing input queue " << p_dataset->get_input_queue()->get_table_name());

    prepare_input_data(p_dataset);

    APP.iq_scaling_factor_service.prepare(*p_dataset);
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size()))
    for (const auto &p_ensemble: p_dataset->get_ensembles()) {
        DeconQueueService::prepare_decon(p_dataset, p_dataset->get_input_queue(), p_ensemble->get_decon_queue());
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_aux_input_queues().size()))
        for (const auto &p_aux_input: p_dataset->get_aux_input_queues()) {
#pragma omp parallel for num_threads(adj_threads(p_aux_input->get_value_columns().size()))
            for (const auto &aux_input_column_name: p_aux_input->get_value_columns()) {
                auto p_decon_queue = DeconQueueService::find_decon_queue(p_ensemble->get_aux_decon_queues(), p_aux_input->get_table_name(), aux_input_column_name);
                DeconQueueService::prepare_decon(p_dataset, p_aux_input, p_decon_queue);
            }
        }
    }

    if (false && getenv("BACKTEST")) {
        const auto decon_queues = p_dataset->get_decon_queues();
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_ensembles().size())) schedule(static, 1)
        for (const auto &p_ensemble: p_dataset->get_ensembles()) {
            APP.decon_queue_service.save(p_ensemble->get_decon_queue());
#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_aux_decon_queues().size())) schedule(static, 1)
            for (const auto &p_aux_decon_decon_queue: p_ensemble->get_aux_decon_queues())
                APP.decon_queue_service.save(p_aux_decon_decon_queue);
        }
    }
    LOG4_END();
}

void InputQueueService::prepare_input_data(datamodel::Dataset_ptr &p_dataset)
{
    LOG4_BEGIN();
    APP.input_queue_service.load(p_dataset->get_input_queue());
#pragma omp parallel for num_threads(adj_threads(p_dataset->get_aux_input_queues().size())) schedule(static, 1)
    for (auto &iq: p_dataset->get_aux_input_queues()) APP.input_queue_service.load(iq);
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
