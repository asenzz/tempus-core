#include "DeconQueueService.hpp"
#include "ModelService.hpp"
#include "model/FramesContainer.hpp"
#include "util/validation_utils.hpp"
#include "util/math_utils.hpp"
#include "model/InputQueue.hpp"
#include "model/DeconQueue.hpp"
#include "model/Dataset.hpp"
#include "DAO/DeconQueueDAO.hpp"
#include "InputQueueService.hpp"
#include "appcontext.hpp"
#include "modwt_transform.hpp"
#include "fast_cvmd.hpp"

using namespace svr;

namespace svr {
namespace business {

DeconQueueService::DeconQueueService(svr::dao::DeconQueueDAO &deconQueueDao) : decon_queue_dao(deconQueueDao)
{}

datamodel::DeconQueue_ptr DeconQueueService::get_by_table_name(const std::string &table_name)
{
    return decon_queue_dao.get_decon_queue_by_table_name(table_name);
}

datamodel::DeconQueue_ptr DeconQueueService::get_by_table_name(const std::string &input_queue_table_name, const bigint dataset_id, const std::string &input_queue_column_name)
{
    return get_by_table_name(make_queue_table_name(input_queue_table_name, dataset_id, input_queue_column_name));
}

void DeconQueueService::save(datamodel::DeconQueue_ptr const &p_decon_queue, boost::posix_time::ptime start_time)
{
    common::reject_nullptr(p_decon_queue);
    if (start_time == bpt::min_date_time && !p_decon_queue->get_data().empty()) {
        const auto last_saved_row = decon_queue_dao.get_latest_data(p_decon_queue->get_table_name(), bpt::max_date_time, 1);
        if (!last_saved_row.empty()) start_time = last_saved_row.back()->get_value_time();
    }
    decon_queue_dao.save(p_decon_queue, start_time);
}

bool DeconQueueService::exists(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    common::reject_nullptr(p_decon_queue);
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


void DeconQueueService::prepare_decon(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue)
{
    LOG4_BEGIN();

    const auto first_offending = InputQueueService::compare_to_decon_queue(input_queue, decon_queue);
    if (first_offending == bpt::max_date_time) return;
    else if (first_offending == bpt::min_date_time) decon_queue.get_data().clear();
    else if (first_offending < decon_queue.back()->get_value_time())
        decon_queue.get_data().erase(lower_bound_back(decon_queue.get_data(), first_offending), decon_queue.end());
    deconstruct(dataset, input_queue, decon_queue);

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
trim_delta(const datamodel::DeconQueue_ptr &p_decon_queue, const boost::posix_time::time_period &period)
{
    auto p_new_decon_queue = p_decon_queue->clone_empty();
    for (auto row_iter = lower_bound_back(p_decon_queue->get_data(), period.begin());
         row_iter != p_decon_queue->end() && row_iter->get()->get_value_time() <= period.end(); ++row_iter)
        p_new_decon_queue->get_data().emplace_back(ptr<datamodel::DataRow>(**row_iter));

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
    std::deque<datamodel::DeconQueue_ptr> new_decon_queues(decon_queues.size());
#pragma omp parallel for num_threads(adj_threads(decon_queues.size())) schedule(static, 1)
    for (size_t i = 0; i < decon_queues.size(); ++i)
        new_decon_queues[i] = trim_delta(decon_queues ^ i, period);

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
        p_decon_queue = ptr<datamodel::DeconQueue>(
                std::string{}, p_input_queue->get_table_name(), column_name, p_dataset->get_id(), p_dataset->get_transformation_levels());
        APP.decon_queue_service.load(p_decon_queue);
        p_dataset->set_decon_queue(p_decon_queue);
    }
#if 0 // TODO Implement and test when hardware resources are available to deconstruct high frequency (1 ms / 1000 Hz) data
    if (p_input_queue->is_tick_queue())
        deconstruct_ticks(p_input_queue, p_dataset, column_name, p_decon_queue->get_data());
    else
#endif
    deconstruct(*p_dataset, *p_input_queue, *p_decon_queue);

    LOG4_END();

    return p_decon_queue;
}


void
DeconQueueService::deconstruct(
        datamodel::Dataset &dataset,
        const datamodel::InputQueue &input_queue,
        datamodel::DeconQueue &decon_queue)
{
    LOG4_BEGIN();

    if (input_queue.empty()) {
        LOG4_ERROR("Input data is empty for " << input_queue);
        return;
    }

    const auto input_column_index = InputQueueService::get_value_column_index(input_queue, decon_queue.get_input_queue_column_name());
    if (input_column_index >= input_queue.front()->size())
        THROW_EX_FS(std::range_error, "Column index out of bounds " << input_column_index << " > " << input_queue.front()->size());

    const auto scaler = IQScalingFactorService::get_scaler(dataset, input_queue, decon_queue.get_input_queue_column_name());
    const auto levct = dataset.get_transformation_levels();
    const auto main_resolution = dataset.get_input_queue()->get_resolution();
#ifdef NO_MAIN_DECON // Hack to avoid proper deconstruction of unused data
    if (input_queue.get_resolution() == main_resolution || dataset.get_transformation_levels() < MIN_LEVEL_COUNT)
        return dummy_decon(input_queue, decon_queue, input_column_index, levct, scaler);
#endif

    const double res_ratio = main_resolution / input_queue.get_resolution();
#ifdef INTEGRATION_TEST
    const auto test_offset = res_ratio * common::INTEGRATION_TEST_VALIDATION_WINDOW;
#else
    constexpr size_t test_offset = 0;
#endif
    const auto residuals_ct = dataset.get_residuals_length(decon_queue.get_table_name());
    LOG4_DEBUG("Input data size " << input_queue.size() << " columns " << input_queue.front()->size() << ", combined residual length " << residuals_ct <<
                    ", input column index " << input_column_index << ", test offset " << test_offset << ", main to aux queue resolution ratio " << res_ratio);
    const auto pre_decon_size = decon_queue.size();
    PROFILE_EXEC_TIME(dataset.get_cvmd_transformer().transform(input_queue, decon_queue, input_column_index, test_offset, scaler), "CVMD transform");
    PROFILE_EXEC_TIME(dataset.get_oemd_transformer().transform(decon_queue, pre_decon_size, test_offset, residuals_ct), "OEMD fat transform");

// Trim not needed data
    const auto trim_diff = std::max<ssize_t>(
            dataset.get_max_lag_count() + decon_queue.size() - pre_decon_size,
            dataset.get_residuals_length(decon_queue.get_table_name()));
    if (trim_diff < ssize_t(decon_queue.size()))
        decon_queue.get_data().erase(decon_queue.begin(), (decon_queue.get_data().rbegin() + trim_diff).base());

    if (SVR_LOG_LEVEL > LOG_LEVEL_T::TRACE) return;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(dataset.get_transformation_levels()))
    for (size_t i = 0; i < dataset.get_transformation_levels(); ++i)
        BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "RMS power for level " << i << " is " << common::meanabs(decon_queue.get_column_values(i, 0));
}


void DeconQueueService::dummy_decon(
        const datamodel::InputQueue &input_queue,
        datamodel::DeconQueue &decon_queue,
        const size_t levix, const size_t levct, const t_iqscaler &iq_scaler)
{
    LOG4_DEBUG("Dummy decon of main input queue " << input_queue.get_table_name());
    const auto &input_data = input_queue.get_data();
    const auto modct = ModelService::to_model_ct(levct);
    auto initer = input_data.begin();
    if (!decon_queue.get_data().empty())
        initer = upper_bound_back(input_data, decon_queue.back()->get_value_time());
    const auto inct = std::distance(initer, input_data.end());
    const auto prev_outct = decon_queue.size();
    decon_queue.get_data().resize(prev_outct + inct);
    const auto outiter = decon_queue.begin() + ssize_t(prev_outct);
    const auto timenow = boost::posix_time::second_clock::local_time();
#pragma omp parallel for num_threads(adj_threads(inct))
    for (ssize_t t = 0; t < inct; ++t) {
        std::vector v(levct, 0.);
        const auto p_inrow = *(initer + t);
        for (size_t l = 0; l < levct; l += 2)
            if (levct < MIN_LEVEL_COUNT || l != levct / 2)
                v[l] = iq_scaler((*p_inrow)[levix]) / double(modct);
        *(outiter + t) = ptr<datamodel::DataRow>(p_inrow->get_value_time(), timenow, p_inrow->get_tick_volume(), v);
    }
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
            const auto row = ptr<svr::datamodel::DataRow>(decon_row_time, bpt::second_clock::local_time(), 0, avg_val);
            std::scoped_lock lg(ins_mx);
            decon_data.insert(decon_data.end(), row);
        }
    )
    LOG4_END();
}
#endif


data_row_container
DeconQueueService::reconstruct(
        const svr::datamodel::datarow_range &decon,
        const recon_type_e type,
        const t_iqscaler &unscaler)
{
    data_row_container recon;
    reconstruct(decon, type, recon, unscaler);
    return recon;
}


void DeconQueueService::reconstruct(
        const datamodel::datarow_range &decon,
        const recon_type_e type,
        data_row_container &recon,
        const t_iqscaler &unscaler)
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
    if (recon.size()) recon.erase(lower_bound(recon, decon.front()->get_value_time()), recon.end());
    const auto startout = recon.size();
    const auto ct = decon.distance();
    recon.resize(startout + ct);
    LOG4_DEBUG("Reconstructing " << ct << " rows of " << levct << " levels, type " << int(type) << ", output starting " << startout);

#pragma omp parallel for num_threads(adj_threads(ct)) schedule(static, 1 + ct / std::thread::hardware_concurrency())
    for (ssize_t i = 0; i < ct; ++i) {
        const datamodel::DataRow &d = **(decon.begin() + i);
        double v = 0;
        for (size_t l = 0; l < levct; l += 2)
            if (l != half_levct)
                op(v, d.get_value(l));
        recon[startout + i] = ptr<datamodel::DataRow>(d.get_value_time(), bpt::second_clock::local_time(), d.get_tick_volume(), std::vector{unscaler(v)});
    }

    LOG4_END();
}


void
DeconQueueService::load(const datamodel::DeconQueue_ptr &p_decon_queue, const bpt::ptime &time_from, const bpt::ptime &time_to, const size_t limit)
{
    common::reject_nullptr(p_decon_queue);
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
    common::reject_nullptr(p_decon_queue);
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
    common::reject_nullptr(decon_queue);
    return decon_queue_dao.clear(decon_queue);
}


long DeconQueueService::count(const datamodel::DeconQueue_ptr &decon_queue)
{
    common::reject_nullptr(decon_queue);
    return decon_queue_dao.count(decon_queue);
}


datamodel::DeconQueue_ptr DeconQueueService::find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue_iter = std::find_if(std::execution::par_unseq, decon_queues.cbegin(), decon_queues.cend(),
            [&input_queue_table_name, &input_queue_column_name](const auto &p_decon_queue) {
                return p_decon_queue->get_input_queue_table_name() == input_queue_table_name &&
                       p_decon_queue->get_input_queue_column_name() == input_queue_column_name;
            }
    );

    if (p_decon_queue_iter == decon_queues.cend()) {
        LOG4_ERROR("Unable to find decon queue for input table name " << input_queue_table_name << ", input column " << input_queue_column_name << ", decon queues ct "
                    << decon_queues.size());
        return nullptr;
    }

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

void DeconQueueService::mirror_tail(const datamodel::datarow_range &input, const size_t needed_data_ct, std::vector<double> &tail)
{
    const size_t in_colix = input.begin()->get()->get_values().size() / 2;
    const auto input_size = input.distance();
    LOG4_WARN("Adding mirrored tail of size " << needed_data_ct - input_size << ", to input of size " <<
                                              input_size << ", total size " << needed_data_ct);
    const auto empty_ct = needed_data_ct - input_size;
    tail.resize(empty_ct);
#pragma omp parallel for num_threads(adj_threads(empty_ct))
    for (size_t i = 0; i < empty_ct; ++i) {
        const auto phi = double(i) / double(input_size);
        tail[empty_ct - 1 - i] = input[(size_t) std::round((input_size - 1) * std::abs(std::round(phi) - phi))]->get_value(in_colix);
    }
}

} // business
} // svr

