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

DeconQueueService::DeconQueueService(dao::DeconQueueDAO &decon_queue_dao) : decon_queue_dao(decon_queue_dao)
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
    REJECT_NULLPTR(p_decon_queue);
    if (start_time == bpt::min_date_time && p_decon_queue->get_data().size()) {
        const auto last_saved_row = decon_queue_dao.get_latest_data(p_decon_queue->get_table_name(), bpt::max_date_time, 1);
        if (!last_saved_row.empty()) start_time = last_saved_row.back()->get_value_time();
    }
    decon_queue_dao.save(p_decon_queue, start_time);
}

bool DeconQueueService::exists(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    REJECT_NULLPTR(p_decon_queue);
    return decon_queue_dao.exists(p_decon_queue->get_table_name());
}

bool DeconQueueService::exists(const std::string &decon_queue_table_name)
{
    return decon_queue_dao.exists(decon_queue_table_name);
}


int DeconQueueService::remove(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    return exists(p_decon_queue) ? decon_queue_dao.remove(p_decon_queue) : 0;
}


void DeconQueueService::prepare_decon(datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue)
{
    LOG4_BEGIN();

    const auto first_offending = InputQueueService::validate_decon_data(input_queue, decon_queue);
    if (first_offending == bpt::max_date_time) return;
    else if (first_offending == bpt::min_date_time && decon_queue.get_data().size()) decon_queue.get_data().clear();
    else if (decon_queue.size() && first_offending < decon_queue.back()->get_value_time())
        decon_queue.get_data().erase(lower_bound(decon_queue.get_data(), first_offending), decon_queue.end());
    deconstruct(dataset, input_queue, decon_queue);

    LOG4_END();
}


std::vector<double> DeconQueueService::get_actual_values(const data_row_container &data, const data_row_container::const_iterator &target_iter)
{
    LOG4_BEGIN();

    if (data.empty()) {
        LOG4_WARN("Data is empty!");
        return {};
    }
    if (target_iter == data.cend()) {
        LOG4_WARN("Target iterator is empty.");
        return {};
    }

    return (**target_iter).get_values();
}


std::deque<datamodel::DeconQueue_ptr>
DeconQueueService::extract_copy_data(
        const datamodel::Dataset_ptr &p_dataset,
        const boost::posix_time::time_period &period)
{
    LOG4_BEGIN();
    const auto decon_queues = p_dataset->get_decon_queues();
    std::deque<datamodel::DeconQueue_ptr> new_decon_queues(decon_queues.size());
    OMP_FOR_i(decon_queues.size()) {
        const auto p_decon_queue = decon_queues ^ i;
        new_decon_queues[i] = p_decon_queue->clone_empty();
        new_decon_queues[i]->set_data(p_decon_queue->get_data(period));
    }

    LOG4_DEBUG("Returning " << new_decon_queues.size() << " decon queues for dataset " << p_dataset->get_id());

    return new_decon_queues;
}


datamodel::DeconQueue_ptr
DeconQueueService::deconstruct(
        datamodel::Dataset &dataset,
        const datamodel::InputQueue &input_queue,
        const std::string &column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue = dataset.get_decon_queue(input_queue, column_name);
    if (!p_decon_queue) {
        p_decon_queue = ptr<datamodel::DeconQueue>(
                std::string{}, input_queue.get_table_name(), column_name, dataset.get_id(), dataset.get_spectral_levels());
        APP.decon_queue_service.load(*p_decon_queue);
        dataset.set_decon_queue(p_decon_queue);
    }
#if 0 // TODO Implement and test when hardware resources are available to deconstruct high frequency (1 ms / 1000 Hz) data
    if (p_input_queue.is_tick_queue())
        deconstruct_ticks(input_queue, dataset, column_name, p_decon_queue->get_data());
    else
#endif
    deconstruct(dataset, input_queue, *p_decon_queue);

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
    const auto levct = dataset.get_spectral_levels();
    const auto resolution = input_queue.get_resolution();
    const auto main_resolution = dataset.get_input_queue()->get_resolution();
    if (dataset.get_input_queue()->get_table_name() == input_queue.get_table_name() || dataset.get_spectral_levels() < MIN_LEVEL_COUNT) // avoid deconstruction of unused data
        return dummy_decon(input_queue, decon_queue, input_column_index, levct, scaler);

    const double res_ratio = main_resolution / resolution;
#ifdef INTEGRATION_TEST
    const auto test_offset = res_ratio * common::C_integration_test_validation_window;
#else
    constexpr uint32_t test_offset = 0;
#endif
    const auto residuals = dataset.get_residuals_length(decon_queue.get_table_name());
    LOG4_DEBUG("Input data length " << input_queue.size() << ", columns " << input_queue.front()->size() << ", combined residual length " << residuals <<
                                    ", input column index " << input_column_index << ", test offset " << test_offset << ", main to aux queue resolution ratio " << res_ratio);
    const auto pre_decon_size = decon_queue.size();

#if defined(VMD_ONLY) && !defined(EMD_ONLY)
    PROFILE_EXEC_TIME(dataset.get_cvmd_transformer().transform(input_queue, decon_queue, input_column_index, test_offset, scaler), "CVMD transform");
#endif
#ifndef VMD_ONLY
#if defined(EMD_ONLY)
    PROFILE_EXEC_TIME(dataset.get_oemd_transformer().transform(input_queue, decon_queue, input_column_index, test_offset, scaler, residuals, main_resolution), "OEMD transform");
#else
    PROFILE_EXEC_TIME(dataset.get_oemd_transformer().transform(decon_queue, pre_decon_size, test_offset, residuals, resolution), "OEMD fat transform");
#endif
#endif

    // Trim not needed data
    const auto trim_diff = std::max<size_t>(
            dataset.get_max_lag_count() * dataset.get_max_quantise() + decon_queue.size() - pre_decon_size,
            dataset.get_residuals_length(decon_queue.get_table_name()) /* leave next decon residuals */);
    if (trim_diff < decon_queue.size()) {
        LOG4_DEBUG("Trimming to " << trim_diff << " rows decon queue " << decon_queue.get_table_name());
        decon_queue.get_data().erase(decon_queue.begin(), (decon_queue.get_data().rbegin() + trim_diff).base());
    }

    if (common::AppConfig::S_log_threshold > boost::log::trivial::severity_level::trace) return;

    const auto chunk_len_res = common::C_default_kernel_max_chunk_len * res_ratio;
    const auto start_rms = decon_queue.size() > chunk_len_res ? decon_queue.size() - chunk_len_res : 0;
    OMP_FOR_i(dataset.get_spectral_levels())
        BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "RMS power for level " << i << " is " << common::meanabs(decon_queue.get_column_values(i, start_rms)) << ", start " << start_rms;
}


void DeconQueueService::dummy_decon(
        const datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon_queue,
        const uint16_t levix, const uint16_t levct, const datamodel::t_iqscaler &iq_scaler)
{
    LOG4_DEBUG("Dummy decon of main input queue " << input_queue.get_table_name());

    const auto &input_data = input_queue.get_data();
    const double modct = ModelService::to_model_ct(levct);
    auto initer = decon_queue.empty() ? input_data.cbegin() : upper_bound(input_data, decon_queue.back()->get_value_time());
    const auto inct = std::distance(initer, input_data.cend());
    const auto prev_outct = decon_queue.size();
    decon_queue.get_data().resize(prev_outct + inct);
    const auto outiter = decon_queue.begin() + ssize_t(prev_outct);
    const auto timenow = boost::posix_time::second_clock::local_time();
    const auto trans_levix = SVRParametersService::get_trans_levix(levct);
    OMP_FOR_i(inct) {
        std::vector v(levct, 0.);
        const auto p_inrow = *(initer + i);
        for (size_t l = 0; l < levct; l += LEVEL_STEP)
            if (l != trans_levix)
                v[l] = iq_scaler(p_inrow->at(levix)) / modct;
        *(outiter + i) = ptr<datamodel::DataRow>(p_inrow->get_value_time(), timenow, p_inrow->get_tick_volume(), v);
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
    if (input_column_index >= input_data.front()->get_values().size())
        THROW_EX_FS(std::range_error, "Column index out of bounds " << input_column_index);
    auto row_iter = p_decon_queue->get_data().empty() ? input_data.begin() : upper_bound(input_data, p_decon_queue->get_data().back()->get_value_time());
    if (row_iter == input_data.end()) {
        LOG4_DEBUG("No new data in input queue to deconstruct!");
        return;
    }

    double last_price = row_iter->get()->get_value(input_column_index);
    const auto start_time_iter = round_second(row_iter->get()->get_value_time());
    const auto end_time = round_second(input_data.back()->get_value_time());
    std::mutex ins_mx;

    omp_tpfor__ (ssize_t, tr, 0, (end_time - start_time_iter).total_seconds(),
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
        const datamodel::t_iqscaler &unscaler)
{
    data_row_container recon;
    reconstruct(decon, type, recon, unscaler);
    return recon;
}


void DeconQueueService::reconstruct(
        const datamodel::datarow_range &decon,
        const recon_type_e type,
        data_row_container &recon,
        const datamodel::t_iqscaler &iq_unscaler)
{
    LOG4_BEGIN();

    if (decon.distance() < 1) LOG4_THROW("No deconstructed data to reconstruct.");

    const auto levct = decon.levels();

    if (levct < 1) LOG4_THROW("No levels to reconstruct.");

    std::function<void(double &, const double)> op;
    if (levct > 1)
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

    const auto trans_levix = SVRParametersService::get_trans_levix(levct);
    if (recon.size()) recon.erase(lower_bound(recon, decon.front()->get_value_time()), recon.end());
    const auto startout = recon.size();
    const auto ct = decon.distance();
    recon.resize(startout + ct);
    const auto time_nau = bpt::second_clock::local_time();
    LOG4_DEBUG("Reconstructing " << ct << " rows of " << levct << " levels, type " << int(type) << ", output starting " << startout);
    OMP_FOR_i(ct) {
        const datamodel::DataRow &d = **(decon.cbegin() + i);
        double v;
        if (levct == 1) v = d[0];
        else {
            v = 0;
            UNROLL()
            for (DTYPE(levct) l = 0; l < levct; l += LEVEL_STEP)
                if (l != trans_levix)
                    op(v, d.get_value(l));
        }
        LOG4_TRACE("Reconstructed value " << v << " for row " << i << " at " << d.get_value_time() << ", unscaled " << iq_unscaler(v));
        recon[startout + i] = otr<datamodel::DataRow>(d.get_value_time(), time_nau, d.get_tick_volume(), std::vector{iq_unscaler(v)});
    }
    LOG4_END();
}


void
DeconQueueService::load(datamodel::DeconQueue &decon_queue, const bpt::ptime &time_from, const bpt::ptime &time_to, const size_t limit)
{
    auto &data = decon_queue.get_data();
    if (data.empty()) decon_queue.set_data(decon_queue_dao.get_data(decon_queue.get_table_name(), time_from, time_to, limit));
    else {
        std::deque<datamodel::DataRow_ptr> new_data = decon_queue_dao.get_data(decon_queue.get_table_name(), time_from, time_to, limit);
        if (new_data.size() && new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound(data, new_data.front()->get_value_time()), data.cend());
        data.insert(data.cend(), new_data.cbegin(), new_data.cend());
    }
}


void
DeconQueueService::load_latest(datamodel::DeconQueue &decon_queue, const bpt::ptime &time_to, const size_t limit)
{
    LOG4_DEBUG("Loading " << limit << " values until " << time_to << " from decon queue " << decon_queue.get_table_name());
    auto &data = decon_queue.get_data();
    if (data.empty())
        decon_queue.set_data(decon_queue_dao.get_latest_data(decon_queue.get_table_name(), time_to, limit));
    else {
        const auto new_data = decon_queue_dao.get_latest_data(decon_queue.get_table_name(), time_to, limit);
        if (!new_data.empty() and new_data.front()->get_value_time() <= data.back()->get_value_time())
            data.erase(lower_bound(data, new_data.front()->get_value_time()), data.cend());
        data.insert(data.cend(), new_data.cbegin(), new_data.cend());
        LOG4_DEBUG("Retrieved " << new_data.size() << " rows.");
    }
}


int DeconQueueService::clear(const datamodel::DeconQueue_ptr &decon_queue)
{
    REJECT_NULLPTR(decon_queue);
    return decon_queue_dao.clear(decon_queue);
}


long DeconQueueService::count(const datamodel::DeconQueue_ptr &decon_queue)
{
    REJECT_NULLPTR(decon_queue);
    return decon_queue_dao.count(decon_queue);
}


datamodel::DeconQueue_ptr DeconQueueService::find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue_iter = std::find_if(C_default_exec_policy, decon_queues.cbegin(), decon_queues.cend(),
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
    LOG4_END();
    return *p_decon_queue_iter;
}


const datamodel::DeconQueue_ptr &DeconQueueService::find_decon_queue(
        const std::deque<datamodel::DeconQueue_ptr> &decon_queues, const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Looking for " << decon_queue_table_name);

    auto p_decon_queue_iter = std::find_if(C_default_exec_policy, decon_queues.cbegin(), decon_queues.cend(),
                                      [&decon_queue_table_name](const datamodel::DeconQueue_ptr &p_decon_queue) {
                                          return p_decon_queue->get_table_name() == decon_queue_table_name;
                                      });

    if (p_decon_queue_iter == decon_queues.cend())
        THROW_EX_FS(std::invalid_argument, "Couldn't not find decon queue for table name " << decon_queue_table_name << ", decon queues ct " << decon_queues.size());

    return *p_decon_queue_iter;
}


std::string DeconQueueService::make_queue_table_name(
        const std::string &input_queue_table_name,
        const bigint dataset_id,
        const std::string &input_queue_column_name)
{
    if (input_queue_table_name.empty() || input_queue_column_name.empty())
        LOG4_THROW(
                "Illegal arguments, input queue table name " << input_queue_table_name << ", input queue column name " << input_queue_column_name << ", dataset id " << dataset_id);
    std::string result = common::sanitize_db_table_name( common::formatter() <<
            common::C_decon_queue_table_name_prefix << "_" << input_queue_table_name << "_" << std::to_string(dataset_id) << "_" << input_queue_column_name);
    std::for_each(C_default_exec_policy, result.begin(), result.end(), ::tolower);
    return result;
}

void DeconQueueService::mirror_tail(const datamodel::datarow_crange &input, const size_t needed_data_ct, std::vector<double> &tail, const unsigned in_colix)
{
    const auto input_size = input.distance();
    const auto empty_ct = needed_data_ct - input_size;
    LOG4_WARN("Adding mirrored tail of size " << empty_ct << ", to input of size " << input_size << ", total size " << needed_data_ct);
    tail.resize(empty_ct);
    const auto fade_in = std::min<unsigned>(empty_ct, C_mirror_fade_in);
    OMP_FOR_i(empty_ct) {
        const auto phi = double(i) / double(input_size);
        const auto out_i = empty_ct - 1 - i;
        tail[out_i] = input[(size_t) std::round((input_size - 1) * std::abs(std::round(phi) - phi))]->at(in_colix);
        if (out_i < fade_in) tail[out_i] *= out_i / fade_in;
    }
}


} // business
} // svr

