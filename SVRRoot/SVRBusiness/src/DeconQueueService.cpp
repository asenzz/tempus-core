#include "DeconQueueService.hpp"

#include "model/FramesContainer.hpp"
#include "util/ValidationUtils.hpp"
#include "util/math_utils.hpp"
#include "model/InputQueue.hpp"
#include "model/DeconQueue.hpp"
#include "model/Dataset.hpp"
#include "DAO/DeconQueueDAO.hpp"
#include "InputQueueService.hpp"
#include "appcontext.hpp"

#include "modwt_transform.hpp"
#include "online_emd.hpp"
#include "fast_cvmd.hpp"


using namespace svr::datamodel;
using namespace svr::common;
using namespace bpt;

namespace svr {
namespace business {


datamodel::DeconQueue_ptr DeconQueueService::get_by_table_name(const std::string &table_name)
{
    return decon_queue_dao.get_decon_queue_by_table_name(table_name);
}

datamodel::DeconQueue_ptr DeconQueueService::get_queue_metadata(const std::string &table_name)
{
    return decon_queue_dao.get_decon_queue_by_table_name(table_name);
}

void DeconQueueService::save(datamodel::DeconQueue_ptr const &p_decon_queue, boost::posix_time::ptime start_time)
{
    reject_nullptr(p_decon_queue);
    if (start_time == bpt::min_date_time && !p_decon_queue->get_data().empty()) {
        const auto last_saved_row = decon_queue_dao.get_latest_data(p_decon_queue->get_table_name(), bpt::max_date_time, 1);
        if (!last_saved_row.empty()) start_time = last_saved_row.back()->get_value_time();
    }
    decon_queue_dao.save(p_decon_queue, start_time);
}

bool DeconQueueService::exists(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    reject_nullptr(p_decon_queue);
    return decon_queue_dao.exists(p_decon_queue->get_table_name());
}

bool DeconQueueService::exists(const std::string &decon_queue_table_name)
{
    return decon_queue_dao.exists(decon_queue_table_name);
}


int DeconQueueService::remove(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    if (exists(p_decon_queue)) return decon_queue_dao.remove(p_decon_queue);
    else return 0;
}


void DeconQueueService::prepare_decon(const datamodel::Dataset_ptr &p_dataset, const datamodel::InputQueue_ptr &p_input_queue, datamodel::DeconQueue_ptr &p_decon_queue)
{
    LOG4_BEGIN();

    if (!p_decon_queue) return;
    const auto first_offending = InputQueueService::compare_to_decon_queue(p_input_queue, p_decon_queue);
    if (first_offending == bpt::max_date_time) return;
    else if (first_offending == bpt::min_date_time) p_decon_queue->get_data().clear();
    else if (first_offending < p_decon_queue->back()->get_value_time())
        p_decon_queue->get_data().erase(lower_bound_back(p_decon_queue->get_data(), first_offending), p_decon_queue->end());
    deconstruct(p_dataset, p_input_queue, p_decon_queue);

    LOG4_END();
}


std::vector<double>
DeconQueueService::get_actual_values(
        const data_row_container &data,
        const data_row_container::const_iterator &target_iter)
{
    LOG4_BEGIN();

    if (data.empty()) {
        LOG4_WARN("Data is empty!");
        return {};
    }
    if (target_iter == data.end()) {
        LOG4_WARN("Target iterator is empty.");
        return {};
    }

    return target_iter->get()->get_values();
}


static datamodel::DeconQueue_ptr
trim_delta(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue, const boost::posix_time::time_period &period)
{
    auto p_new_decon_queue = p_decon_queue->clone_empty();
    for (auto row_iter = lower_bound_back(p_decon_queue->get_data(), period.begin());
         row_iter != p_decon_queue->end() && row_iter->get()->get_value_time() <= period.end(); ++row_iter)
        p_new_decon_queue->get_data().emplace_back(std::make_shared<DataRow>(**row_iter));

    if (p_new_decon_queue->get_data().empty())
        LOG4_ERROR("Empty decon queue " << p_new_decon_queue->to_string() << " for period " << period);
    else
        LOG4_DEBUG(
            "Returning " << p_new_decon_queue->size() << " for " << p_new_decon_queue->get_input_queue_table_name() << " "
             << p_new_decon_queue->get_input_queue_column_name() << " from " << p_new_decon_queue->front()->get_value_time() << " until " << p_new_decon_queue->back()->get_value_time() << " for period " << period);

    return p_new_decon_queue;
}


std::deque<datamodel::DeconQueue_ptr>
DeconQueueService::extract_copy_data(
        const datamodel::Dataset_ptr &p_dataset,
        const boost::posix_time::time_period &period)
{
    LOG4_BEGIN();
    const auto decon_queues = p_dataset->get_decon_queues();
    std::deque<datamodel::DeconQueue_ptr> new_decon_queues;
#pragma omp parallel for num_threads(adj_threads(decon_queues.size()))
    for (size_t i = 0; i < decon_queues.size(); ++i) {
        const auto p_new_decon_queue = trim_delta(p_dataset, decon_queues ^ i, period);
#pragma omp critical
        new_decon_queues.emplace_back(p_new_decon_queue);
    }

    LOG4_DEBUG("Returning " << new_decon_queues.size() << " decon queues for dataset " << p_dataset->get_id());

    return new_decon_queues;
}


datamodel::DeconQueue_ptr
DeconQueueService::deconstruct(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::InputQueue_ptr &p_input_queue,
        const std::string &column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue = p_dataset->get_decon_queue(p_input_queue, column_name);
    if (!p_decon_queue) {
        p_decon_queue = std::make_shared<DeconQueue>(
                make_queue_table_name(p_input_queue->get_table_name(), p_dataset->get_id(), column_name),
                p_input_queue->get_table_name(),
                column_name,
                p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        APP.decon_queue_service.load(p_decon_queue);
        p_dataset->set_decon_queue(p_decon_queue);
    }
#if 0 // TODO Implement and test when hardware resources are available to deconstruct high frequency (1 ms / 1000 Hz) data
    if (p_input_queue->is_tick_queue())
        deconstruct_ticks(p_input_queue, p_dataset, column_name, p_decon_queue->get_data());
    else
#endif
    deconstruct(p_dataset, p_input_queue, p_decon_queue);

    LOG4_END();

    return p_decon_queue;
}


void
DeconQueueService::deconstruct(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::InputQueue_ptr &p_input_queue,
        datamodel::DeconQueue_ptr &p_decon_queue)
{
    LOG4_BEGIN();

    if (!p_input_queue) {
        LOG4_ERROR("Input queue not initialized!");
        return;
    }

    data_row_container &input_data = p_input_queue->get_data();
    if (input_data.empty()) {
        LOG4_ERROR("Data is empty for " << *p_input_queue);
        return;
    }

    const auto input_column_index = InputQueueService::get_value_column_index(p_input_queue, p_decon_queue->get_input_queue_column_name());
    if (p_dataset->get_transformation_levels() < 2) {
        p_decon_queue->set_data(InputQueueService::get_column_data(p_input_queue, input_column_index));
        return;
    }

    if (input_column_index >= input_data.front()->get_values().size())
        THROW_EX_FS(std::range_error, "Column index out of bounds " << input_column_index << " > " << input_data.front()->get_values().size());

    const auto levct = p_dataset->get_transformation_levels();
    const auto main_resolution = p_dataset->get_input_queue()->get_resolution();
    const double res_ratio = main_resolution / p_input_queue->get_resolution();
    const auto test_offset = res_ratio * MANIFOLD_TEST_VALIDATION_WINDOW;

#ifdef NO_MAIN_DECON // Hack to avoid proper deconstruction of unused data
    if (p_input_queue->get_resolution() == main_resolution || p_dataset->get_transformation_levels() < 8) {
        dummy_decon(p_input_queue, p_decon_queue, levct);
        return;
    }
#endif
    const auto residuals_ct = p_dataset->get_residuals_length(p_decon_queue->get_table_name());

    LOG4_DEBUG("Input data size " << input_data.size() << " columns " << input_data.begin()->get()->get_values().size() << ", VMD residual count " << residuals_ct <<
                    ", input column index " << input_column_index << ", test offset " << test_offset << ", res_ratio " << res_ratio);
    const auto pre_decon_size = p_decon_queue->size();
    PROFILE_EXEC_TIME(p_dataset->get_cvmd_transformer().transform(input_data, *p_decon_queue, input_column_index, test_offset), "CVMD transform");
    PROFILE_EXEC_TIME(p_dataset->get_oemd_transformer().transform(p_decon_queue, pre_decon_size, test_offset, residuals_ct), "OEMD fat transform");

// Trim not needed data
    const auto trim_diff = std::max<ssize_t>(
            p_dataset->get_max_lag_count() + p_decon_queue->size() - pre_decon_size,
            p_dataset->get_residuals_length(p_decon_queue->get_table_name()));
    if (trim_diff < ssize_t(p_decon_queue->size()))
        p_decon_queue->get_data().erase(p_decon_queue->begin(), (p_decon_queue->get_data().rbegin() + trim_diff).base());

    LOG4_END();
}


void DeconQueueService::dummy_decon(const datamodel::InputQueue_ptr &p_input_queue, datamodel::DeconQueue_ptr &p_decon_queue, const size_t levct)
{
    LOG4_DEBUG("Dummy decon of main input queue " << p_input_queue->get_table_name());
    const auto &input_data = p_input_queue->get_data();
    const size_t modct = levct / 2 - 1;
    auto initer = input_data.begin();
    if (!p_decon_queue->get_data().empty()) initer = upper_bound_back(input_data, p_decon_queue->back()->get_value_time());
    const auto inct = std::distance(initer, input_data.end());
    const auto prev_outct = p_decon_queue->size();
    p_decon_queue->get_data().resize(prev_outct + inct);
    const auto outiter = p_decon_queue->begin() + ssize_t(prev_outct);
    const auto timenow = boost::posix_time::second_clock::local_time();
#pragma omp parallel for num_threads(adj_threads(inct))
    for (ssize_t t = 0; t < inct; ++t) {
        std::vector<double> v(levct, 0.);
        const auto inrow = **(initer + t);
        for (size_t l = 0; l < levct; l += 2)
            if (l != levct / 2)
                v[l] = inrow.get_value(l) / double(modct);
        *(outiter + t) = std::make_shared<DataRow>(
                inrow.get_value_time(), timenow, inrow.get_tick_volume(), v);
    }
    return;
}


#if 0 // TODO Test when hardware for deconstructing high-frequency data becomes available
void
DeconQueueService::deconstruct_ticks(
        const datamodel::InputQueue_ptr &p_input_queue,
        const datamodel::Dataset_ptr &p_dataset,
        const std::string &column_name,
        data_row_container &decon_data)
{
    LOG4_BEGIN();

    if (p_dataset->get_transformation_levels() < 2) {
        decon_data = InputQueueService::get_column_data(p_input_queue, column_name);
        return;
    }
    const auto &input_data = p_input_queue->get_data();
    if (input_data.empty()) {
        LOG4_ERROR("Data is empty.");
        return;
    }

    const auto input_column_index = InputQueueService::get_value_column_index(p_input_queue, column_name);
    const auto p_decon_queue = p_dataset->get_decon_queue(p_input_queue, column_name);
    if (input_column_index >= input_data.begin()->get()->get_values().size())
        THROW_EX_FS(std::range_error, "Column index out of bounds " << input_column_index);
    auto row_iter = p_decon_queue->get_data().empty() ? input_data.begin() : upper_bound(input_data, p_decon_queue->get_data().rbegin()->get()->get_value_time());
    if (row_iter == input_data.end()) {
        LOG4_DEBUG("No new data in input queue to deconstruct!");
        return;
    }

    double last_price = row_iter->get()->get_value(input_column_index);
    const auto start_time_iter = round_second(row_iter->get()->get_value_time());
    const auto end_time = round_second(input_data.rbegin()->get()->get_value_time());
    std::mutex ins_mx;

    __omp_tpfor (ssize_t, tr, 0, (end_time - start_time_iter).total_seconds(),
                 const auto decon_row_time = start_time_iter + bpt::seconds(tr);
        if (decon_row_time.date().day_of_week() != 6 and decon_row_time.date().day_of_week() != 0) {
            bpt::ptime time_iter = decon_row_time;
            std::vector<double> msec_prices(1000);
            for (size_t ms = 0; ms < msec_prices.size(); ++ms) {
                while (round_millisecond(row_iter->get()->get_value_time()) < time_iter) {
                    last_price = row_iter->get()->get_value(input_column_index);
                    ++row_iter;
                }
                if (round_millisecond(row_iter->get()->get_value_time()) == time_iter) {
                    last_price = row_iter->get()->get_value(input_column_index);
                    msec_prices[ms] = last_price;
                    ++row_iter;
                } else if (round_millisecond(row_iter->get()->get_value_time()) > time_iter) {
                    msec_prices[ms] = last_price;
                }
                time_iter += bpt::milliseconds(ms);
            }

            std::vector<std::vector<double>> decon_values(msec_prices.size());
            for (size_t t = 0; t < msec_prices.size(); ++t) {
                //p_dataset->p_oemd_transformer_fat->transform(msec_prices[t], decon_values[t]); // TODO Not implemented
                std::vector<double> decon_values_thin;
                //p_dataset->oemd_transformer_thin->transform(decon_values[t].back(), decon_values_thin); // TODO Not implemented
                decon_values[t].insert(decon_values[t].end(), decon_values_thin.begin(), decon_values_thin.end());
            }
            std::vector<double> avg_val(decon_values[0].size(), 0.);
            for (size_t l = 0; l < decon_values[0].size(); ++l) {
                for (size_t t = 0; t < decon_values.size(); ++t)
                    avg_val[l] += decon_values[t][l];
                avg_val[l] /= double(decon_values.size());
            }
            const auto row = std::make_shared<svr::datamodel::DataRow>(decon_row_time, bpt::second_clock::local_time(), 0, avg_val);
            std::scoped_lock lg(ins_mx);
            decon_data.insert(decon_data.end(), row);
        }
    )
    LOG4_END();
}
#endif


data_row_container
DeconQueueService::reconstruct(
        const svr::datamodel::datarow_range &data,
        const recon_type_e type)
{
    data_row_container res;
    reconstruct(data, type, res);
    return res;
}


void DeconQueueService::reconstruct(
        const datamodel::datarow_range &decon,
        const recon_type_e type,
        data_row_container &recon)
{
    LOG4_BEGIN();
    if (decon.distance() < 1) LOG4_THROW("No deconstructed data to reconstruct.");
    std::function<void(double &, const double)> op;
    switch (type) {
        case recon_type_e::ADDITIVE:
            op = [](double &inout, const double in) -> void { inout += in; };
            break;

        case recon_type_e::MULTIPLICATIVE:
            op = [](double &inout, const double in) -> void { inout *= in; };
            break;

        default:
            LOG4_THROW("Reconstruction type " << int(type) << " not supported!");
    }

    const auto levct = decon.front()->size();
    const size_t half_levct = levct / 2;
    if (!recon.empty()) recon.erase(lower_bound(recon, decon.front()->get_value_time()), recon.end());
    const auto startout = recon.size();
    const auto ct = decon.distance();
    recon.resize(startout + ct);
    LOG4_DEBUG("Reconstructing " << ct << " rows of " << levct << " levels, type " << int(type) << ", output starting " << startout);

#pragma omp parallel for num_threads(adj_threads(ct))
    for (ssize_t i = 0; i < ct; ++i) {
        const datamodel::DataRow &d = **(decon.begin() + i);
        double v = 0;
        for (size_t l = 0; l < levct; l += 2)
            if (l != half_levct)
                op(v, d.get_value(l));
        recon[startout + i] = std::make_shared<DataRow>(d.get_value_time(), bpt::second_clock::local_time(), d.get_tick_volume(), std::vector{v});
    }

    LOG4_END();
}


void
DeconQueueService::load(const datamodel::DeconQueue_ptr &p_decon_queue, const ptime &time_from, const ptime &time_to, const size_t limit)
{
    reject_nullptr(p_decon_queue);
    data_row_container &data = p_decon_queue->get_data();
    if (data.empty()) {
        p_decon_queue->set_data(decon_queue_dao.get_data(p_decon_queue->get_table_name(), time_from, time_to, limit));
    } else {
        std::deque<datamodel::DataRow_ptr> new_data = decon_queue_dao.get_data(p_decon_queue->get_table_name(), time_from, time_to, limit);
        if (!new_data.empty() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
    }
}


void
DeconQueueService::load_latest(const datamodel::DeconQueue_ptr &p_decon_queue, const bpt::ptime &time_to, const size_t limit)
{
    LOG4_DEBUG("Loading " << limit << " values until " << time_to << " from decon queue " << p_decon_queue->get_table_name());
    reject_nullptr(p_decon_queue);
    data_row_container &data = p_decon_queue->get_data();
    if (data.empty()) {
        p_decon_queue->set_data(decon_queue_dao.get_latest_data(p_decon_queue->get_table_name(), time_to, limit));
    } else {
        const auto new_data = decon_queue_dao.get_latest_data(p_decon_queue->get_table_name(), time_to, limit);
        if (!new_data.empty() and new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound_back(data, new_data.front()->get_value_time()), data.end());
        data.insert(data.end(), new_data.begin(), new_data.end());
        LOG4_DEBUG("Retrieved " << new_data.size() << " rows.");
    }
}


int DeconQueueService::clear(const datamodel::DeconQueue_ptr &decon_queue)
{
    reject_nullptr(decon_queue);
    return decon_queue_dao.clear(decon_queue);
}


long DeconQueueService::count(const datamodel::DeconQueue_ptr &decon_queue)
{
    reject_nullptr(decon_queue);
    return decon_queue_dao.count(decon_queue);
}


datamodel::DeconQueue_ptr DeconQueueService::find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue_iter = find_if(
            decon_queues.begin(),
            decon_queues.end(),
            [&input_queue_table_name, &input_queue_column_name](const datamodel::DeconQueue_ptr &p_decon_queue) {
                return p_decon_queue->get_input_queue_table_name() == input_queue_table_name &&
                       p_decon_queue->get_input_queue_column_name() == input_queue_column_name;
            }
    );

    if (p_decon_queue_iter == decon_queues.end())
        THROW_EX_FS(std::invalid_argument, "Couldn't not find decon queue for input table name " << input_queue_table_name <<
                                                                                                 ", input column " << input_queue_column_name << ", decon queues ct "
                                                                                                 << decon_queues.size());

    return *p_decon_queue_iter;
}

const datamodel::DeconQueue_ptr &DeconQueueService::find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues, const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Looking for " << decon_queue_table_name);

    auto p_decon_queue_iter = find_if(
            decon_queues.begin(),
            decon_queues.end(),
            [&decon_queue_table_name](const datamodel::DeconQueue_ptr &p_decon_queue) {
                return p_decon_queue->get_table_name() == decon_queue_table_name;
            }
    );

    if (p_decon_queue_iter == decon_queues.end())
        THROW_EX_FS(std::invalid_argument, "Couldn't not find decon queue for table name " << decon_queue_table_name << ", decon queues ct " << decon_queues.size());

    return *p_decon_queue_iter;
}

std::string
DeconQueueService::make_queue_table_name(
        const std::string &input_queue_table_name,
        const bigint dataset_id,
        const std::string &input_queue_column_name)
{
    if (input_queue_table_name.empty() || input_queue_column_name.empty())
        LOG4_THROW("Illegal arguments, input queue table name " << input_queue_table_name << ", input queue column name " << input_queue_column_name << ", dataset id " << dataset_id);
    std::string result = common::sanitize_db_table_name(
            common::C_decon_queue_table_name_prefix + "_" +
            input_queue_table_name + "_" +
            std::to_string(dataset_id) + "_" +
            input_queue_column_name);
    std::transform(std::execution::par_unseq, result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // business
} // svr
