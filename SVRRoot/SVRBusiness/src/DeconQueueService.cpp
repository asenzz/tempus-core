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


DeconQueue_ptr DeconQueueService::get_by_table_name(const std::string &table_name)
{
    return decon_queue_dao.get_decon_queue_by_table_name(table_name);
}

DeconQueue_ptr DeconQueueService::get_queue_metadata(const std::string &table_name)
{
    return decon_queue_dao.get_decon_queue_by_table_name(table_name);
}

void DeconQueueService::save(DeconQueue_ptr const &p_decon_queue, const boost::posix_time::ptime &start_time)
{
    reject_nullptr(p_decon_queue);
    decon_queue_dao.save(p_decon_queue, start_time);
}

bool DeconQueueService::exists(DeconQueue_ptr const &p_decon_queue)
{
    reject_nullptr(p_decon_queue);
    return decon_queue_dao.exists(p_decon_queue->get_table_name());
}

bool DeconQueueService::exists(const std::string &decon_queue_table_name)
{
    return decon_queue_dao.exists(decon_queue_table_name);
}

int DeconQueueService::remove(DeconQueue_ptr const &p_decon_queue)
{
    if (exists(p_decon_queue)) return decon_queue_dao.remove(p_decon_queue);
    else return 0;
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


static DeconQueue_ptr
trim_delta(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue, const boost::posix_time::time_period &period)
{
    auto p_new_decon_queue = p_decon_queue->clone_empty();
    for (auto row_iter = lower_bound_back(p_decon_queue->get_data(), period.begin());
        row_iter != p_decon_queue->get_data().end() && row_iter->get()->get_value_time() <= period.end(); ++row_iter)
        p_new_decon_queue->get_data().emplace_back(std::make_shared<DataRow>(**row_iter));

    if (p_new_decon_queue->get_data().empty())
        LOG4_ERROR("Empty decon queue " << p_new_decon_queue->to_string() << " for period " << period);
    else
        LOG4_DEBUG(
                "Returning " << p_new_decon_queue->get_data().size() << " for " << p_new_decon_queue->get_input_queue_table_name() << " " << p_new_decon_queue->get_input_queue_column_name() <<
                " from " << p_new_decon_queue->get_data().front()->get_value_time() << " until " << p_new_decon_queue->get_data().back()->get_value_time() << " for period " << period);

    return p_new_decon_queue;
}


std::vector<DeconQueue_ptr>
DeconQueueService::
extract_copy_data(
        const Dataset_ptr &p_dataset,
        const boost::posix_time::time_period &period)
{
    LOG4_BEGIN();
    const auto decon_queues = p_dataset->get_decon_queues();
    std::vector<DeconQueue_ptr> new_decon_queues;
    std::mutex mx;
    __pxt_pfor_i(0, decon_queues.size(),
        const auto p_decon_queue = std::next(decon_queues.begin(), i)->second;
        const auto p_new_decon_queue = trim_delta(p_dataset, p_decon_queue, period);
        std::scoped_lock lg(mx);
        new_decon_queues.emplace_back(p_new_decon_queue);
    )

    LOG4_DEBUG("Returning " << new_decon_queues.size() << " decon queues for dataset " << p_dataset->get_id());

    return new_decon_queues;
}


std::vector<DeconQueue_ptr>
DeconQueueService::deconstruct(
        const InputQueue_ptr &p_input_queue,
        const Dataset_ptr &p_dataset,
        const bool get_data_from_database)
{
    if (!p_input_queue) LOG4_THROW("Input queue not initialized!");
    const auto column_names = p_input_queue->get_value_columns();
    std::vector<DeconQueue_ptr> decon_queues(column_names.size());
    __pxt_pfor_i(0, column_names.size(), decon_queues[i] = deconstruct(p_input_queue, p_dataset, column_names[i], get_data_from_database) )
    return decon_queues;
}


DeconQueue_ptr
DeconQueueService::deconstruct(
        const InputQueue_ptr &p_input_queue,
        const Dataset_ptr &p_dataset,
        const std::string &column_name,
        const bool get_data_from_database)
{
    LOG4_BEGIN();
    auto p_decon_queue = p_dataset->get_decon_queue(p_input_queue, column_name);
    if (!p_decon_queue) {
        p_decon_queue = std::make_shared<DeconQueue>(
                "",
                p_input_queue->get_table_name(),
                column_name,
                p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        p_dataset->set_decon_queue(p_decon_queue);
    }
    if (p_input_queue->is_tick_queue())
        deconstruct_ticks(p_input_queue, p_dataset, column_name, p_decon_queue->get_data());
    else
        deconstruct(p_input_queue, p_dataset, column_name, p_decon_queue->get_data(), get_data_from_database);

    LOG4_END();

    return p_decon_queue;
}


void
DeconQueueService::deconstruct_ticks(
        const InputQueue_ptr &p_input_queue,
        const Dataset_ptr &p_dataset,
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


void
DeconQueueService::deconstruct(
        const InputQueue_ptr &p_input_queue,
        const Dataset_ptr &p_dataset,
        const std::string &column_name,
        svr::datamodel::DataRow::container &decon_out,
        const bool get_data_from_database)
{
    LOG4_BEGIN();

    if (p_dataset->get_transformation_levels() < 2) {
        decon_out = InputQueueService::get_column_data(p_input_queue, column_name);
        return;
    }
    const auto input_data = get_data_from_database ? InputQueueService::get_column_data(p_input_queue, column_name) : p_input_queue->get_data();
    if (input_data.empty()) {
        LOG4_ERROR("Data is empty.");
        return;
    }

    const auto input_column_index = get_data_from_database ? 0 : InputQueueService::get_value_column_index(p_input_queue, column_name);
    const auto p_decon_queue = p_dataset->get_decon_queue(p_input_queue, column_name);
    if (!p_decon_queue) {
        LOG4_ERROR("Could not find decon queue for " << p_input_queue->get_table_name() << " " << column_name);
        return;
    }

    if (input_column_index >= input_data.begin()->get()->get_values().size())
        THROW_EX_FS(std::range_error, "Column index out of bounds " << input_column_index);

    // TODO Rewrite CVMD data harvesting
    const auto cvmd_residuals_ct_orig = p_dataset->p_cvmd_transformer->get_residuals_length(p_decon_queue->get_table_name());
    auto cvmd_residuals_ct = p_decon_queue and not p_decon_queue->get_data().empty() ? 0 : p_dataset->p_cvmd_transformer->get_residuals_length(p_decon_queue->get_table_name());
    auto oemd_residuals_ct = p_dataset->p_oemd_transformer_fat->get_residuals_length();
    if (cvmd_residuals_ct and cvmd_residuals_ct < oemd_residuals_ct) cvmd_residuals_ct = oemd_residuals_ct; // get()->get_value_time() deconstruct
    LOG4_DEBUG("OEMD residual length " << oemd_residuals_ct << " CVMD residuals length " << cvmd_residuals_ct);
    auto start_row_iter = input_data.begin();
    if (p_decon_queue and not p_decon_queue->get_data().empty()) { // Decon queue not empty
        start_row_iter = upper_bound(input_data, p_decon_queue->get_data().rbegin()->get()->get_value_time());
        if (start_row_iter == input_data.end()) {
            LOG4_DEBUG("No data needs to be deconstructed for " << p_input_queue->get_table_name() << " " << column_name);
            return;
        }
        LOG4_TRACE("Start row " << start_row_iter->get()->get_value_time() << " input data rbegin value_time " << input_data.rbegin()->get()->get_value_time() << ", p_decon_queue->get_data().rbegin()->get()->get_value_time() " << p_decon_queue->get_data().rbegin()->get()->get_value_time());
        const auto input_dist_begin = std::distance(input_data.begin(), start_row_iter);
        std::advance(start_row_iter, - std::min(ssize_t(cvmd_residuals_ct), input_dist_begin));
        if (input_dist_begin < ssize_t(cvmd_residuals_ct))
            LOG4_WARN("Not enough data in input queue " << input_dist_begin << " for deconstruction residuals " << cvmd_residuals_ct);
    }

    LOG4_DEBUG("Input data size " << input_data.size() << " columns " << input_data.begin()->get()->get_values().size() <<
                  " VMD residual count " << cvmd_residuals_ct << " std::distance(input_data.begin(), start_row_iter) " << std::distance(input_data.begin(), start_row_iter) <<
                  " std::distance(start_row_iter, input_data.end()) " << std::distance(start_row_iter, input_data.end()) << " input_column_index " << input_column_index);

    const size_t half_levels_ct = p_dataset->get_transformation_levels() / 2;
    std::vector<std::vector<double>> cvmd_decon, oemd_decon;
    { // Deconstruct CVMD
        std::vector<double> column_values(std::distance(start_row_iter, input_data.end()));
        __par_iter(start_row_iter, column_values.size(),column_values[_IX] = _ITER->get()->get_value(input_column_index); );

#ifdef NO_MAIN_DECON
        if (p_input_queue->get_resolution().total_seconds() == MAIN_DECON_QUEUE_RES_SECS) {
            LOG4_DEBUG("Dummy decon of main input queue " << p_input_queue->get_table_name());
            std::vector<std::vector<double>> decon(column_values.size());
            __omp_pfor(t, 0, column_values.size(),
                decon[t].resize(p_dataset->get_transformation_levels(), 0.);
                for (size_t half_l = 0; half_l < half_levels_ct; ++half_l)
                    decon[t][half_l * 2] = column_values[t] / double(half_levels_ct);
            )
            copy_decon_data_to_container(decon_out, input_data, decon, p_input_queue->get_resolution());
            return;
        }
#endif

        APP.decon_queue_service.deconstruct_cvmd(p_dataset, p_input_queue, column_name, column_values, cvmd_decon);
    }

    size_t input_oemd_ct = cvmd_decon.size() + oemd_residuals_ct;
    if (cvmd_decon.size() > cvmd_residuals_ct_orig) input_oemd_ct -= cvmd_residuals_ct_orig;
    std::vector<double> input_oemd(input_oemd_ct);
    { // Deconstruct with OEMD
        // Copy CVMD data // TODO Omit tail?
        const size_t oemd_cvmd_input_ct = std::min<size_t>(input_oemd.size(), cvmd_decon.size());
        for (size_t i = 0; i < oemd_cvmd_input_ct; ++i)
            input_oemd[input_oemd.size() - oemd_cvmd_input_ct + i] = cvmd_decon[cvmd_decon.size() - oemd_cvmd_input_ct + i][0];

        // Copy prev decon data
        int64_t dist_decon = input_oemd.size() - cvmd_decon.size();
        if (dist_decon > int64_t(p_decon_queue->get_data().size())) dist_decon = p_decon_queue->get_data().size();
        if (dist_decon < 0) dist_decon = 0;
        auto iter = std::prev(p_decon_queue->get_data().end(), dist_decon);
        for (int64_t i = 0; i < dist_decon and iter != p_decon_queue->get_data().end(); ++i, ++iter)
            input_oemd[input_oemd.size() - dist_decon - cvmd_decon.size() + i] = iter->get()->get_value(half_levels_ct);

        if (input_oemd.size() > dist_decon + cvmd_decon.size())
        // Add mirror tail
        mirror_tail(input_oemd, dist_decon + cvmd_decon.size());
        if (not p_dataset->p_oemd_transformer_fat->get_fir_coefs_initialized()) {
#ifdef MANIFOLD_TEST
            p_dataset->p_oemd_transformer_fat->find_fir_coefficients({input_oemd.begin() + cvmd_residuals_ct_orig, input_oemd.end() - MAIN_DECON_QUEUE_RES_SECS * MANIFOLD_TEST_VALIDATION_WINDOW});
#else
            p_dataset->p_oemd_transformer_fat->find_fir_coefficients({input_oemd.begin() + cvmd_residuals_ct_orig, input_oemd.end() - MAIN_DECON_QUEUE_RES_SECS * MODEL_TRAIN_OFFSET});
#endif
        }
        // Transform
        PROFILE_EXEC_TIME(p_dataset->p_oemd_transformer_fat->transform(input_oemd, oemd_decon, input_oemd.size()),
                          "OEMD fat transform of " << input_oemd.size() << " values.");
    }

    int64_t decon_ct;
    if (cvmd_residuals_ct)
        decon_ct = std::max<int64_t>(oemd_decon.size(), cvmd_decon.size()); // First deconstruct include the tail
    else
        decon_ct = cvmd_decon.size();
    std::vector<std::vector<double>> decon(decon_ct); // Only new are in CVMD decon data while OEMD always contains a tail
    __omp_pfor(t, 0, decon_ct,
               decon[t].resize(p_dataset->get_transformation_levels());

        int64_t t_oemd = -1;
        if (decon_ct < int64_t(oemd_decon.size()))
            t_oemd = t + oemd_decon.size() - decon_ct;
        else if (t >= decon_ct - int64_t(oemd_decon.size()))
            t_oemd = t - (decon_ct - int64_t(oemd_decon.size()));

        int64_t t_cvmd = -1;
        if (decon_ct < int64_t(cvmd_decon.size()))
            t_cvmd = t + cvmd_decon.size() - decon_ct;
        else if (t >= decon_ct - int64_t(cvmd_decon.size()))
            t_cvmd = t - (decon_ct - int64_t(cvmd_decon.size()));

        for (size_t l = 0; l < p_dataset->get_transformation_levels(); ++l) {
            if (l == half_levels_ct) {
                decon[t][l] = t_oemd < 0 ? 0 : input_oemd[t_oemd]; // Mirrored tail for OEMD
            } if (l < half_levels_ct) {
                decon[t][l] = l % 2 == 1 ? 0 : (t_oemd < 0 ? 0 : oemd_decon[t_oemd][l / 2]); // OEMD decon
            } else {
                decon[t][l] = t_cvmd < 0 ? 0 : cvmd_decon[t_cvmd][l - half_levels_ct]; // CVMD decon
            }
        }
        if (t_cvmd >= 0 && t_cvmd < (int64_t) cvmd_decon.size()) cvmd_decon[t_cvmd].clear();
        if (t_oemd >= 0 && t_oemd < (int64_t) oemd_decon.size()) oemd_decon[t_oemd].clear();
    )
    input_oemd.clear();
    cvmd_decon.clear();
    oemd_decon.clear();

    if (input_data.size() != decon.size()) LOG4_DEBUG("Input data size " << input_data.size() << " differs from decon matrix rows count  " << decon.size());

    copy_decon_data_to_container(decon_out, input_data, decon, p_input_queue->get_resolution());

    LOG4_END();
}


void
DeconQueueService::deconstruct_cvmd(
    const Dataset_ptr &p_dataset,
    const InputQueue_ptr &p_input_queue,
    const std::string &column_name,
    const std::vector<double> &column_values,
    std::vector<std::vector<double>> &raw_deconstructed_data,
    const bool do_batch) const
{
    const auto p_decon_queue = p_dataset->get_decon_queue(p_input_queue, column_name);
    const auto decon_queue_table_name = p_decon_queue->get_table_name();

    if (!p_dataset->p_cvmd_transformer->initialized(decon_queue_table_name)) PROFILE_EXEC_TIME(
            p_dataset->p_cvmd_transformer->initialize(
#ifdef MANIFOLD_TEST
                p_input_queue->get_column_values(column_name, std::min<size_t>(0, p_input_queue->get_data().size() - PART_CVMD_COUNT - MAIN_DECON_QUEUE_RES_SECS * MANIFOLD_TEST_VALIDATION_WINDOW), PART_CVMD_COUNT), decon_queue_table_name),
#else
                p_input_queue->get_column_values(column_name, std::min<size_t>(0, p_input_queue->get_data().size() - PART_CVMD_COUNT), PART_CVMD_COUNT), decon_queue_table_name),
#endif
            "CVMD initialize");

    // Do batch
    if (do_batch or p_decon_queue->get_data().empty()) {
        PROFILE_EXEC_TIME(p_dataset->p_cvmd_transformer->transform(column_values, raw_deconstructed_data, decon_queue_table_name), "CVMD batch transform");
        // Do online
    } else {
        const auto last_decon_values_iter = p_decon_queue->get_data().rbegin();
        std::vector<double> last_decon_vals(std::next(last_decon_values_iter->get()->get_values().begin(), last_decon_values_iter->get()->get_values().size() / 2),
                                       last_decon_values_iter->get()->get_values().end());
        PROFILE_EXEC_TIME(p_dataset->p_cvmd_transformer->transform(column_values, raw_deconstructed_data, decon_queue_table_name, last_decon_vals), "CVMD online transform");
    }
}


inline void
DeconQueueService::copy_decon_data_to_container(
        DataRow::container &decon_out,
        const DataRow::container &input_data,
        std::vector<std::vector<double>> &decon,
        const boost::posix_time::time_duration &resolution)
{
    auto input_iter_start = input_data.size() > decon.size() ? input_data.begin() + input_data.size() - decon.size() : input_data.begin();
    const size_t input_distance = input_data.size() > decon.size() ? decon.size() : input_data.size();
    LOG4_DEBUG("Deconstructed size " << decon.size() << " using input data rows count " << std::distance(input_iter_start, input_data.end()) <<
                                     " from " << input_iter_start->get()->get_value_time() << " to " << input_data.back()->get_value_time());

    const auto start_decon_ix = input_data.size() >= decon.size() ? 0 : decon.size() - input_data.size();
    const auto end_decon_ix = decon.size();
    const auto decon_distance = end_decon_ix - start_decon_ix;
    const size_t loop_len = std::min<size_t>(input_distance, decon_distance);
    decon_out.erase(lower_bound(decon_out, input_iter_start->get()->get_value_time()), decon_out.end()); // Remove old, overlapping decon data
    const size_t out_start_ix = decon_out.size();
    decon_out.resize(out_start_ix + loop_len);
    __tbb_pfor_i(0, loop_len,
        const auto input_i = input_iter_start + i;
        const auto decon_i = start_decon_ix + i;
        decon_out[out_start_ix + i] = std::make_shared<DataRow>((*input_i)->get_value_time(),
                     boost::posix_time::second_clock::local_time(), (*input_i)->get_tick_volume(), decon[decon_i]);
        decon[decon_i].clear();
     )

    LOG4_DEBUG("Output decon data from " << decon_out.begin()->get()->get_value_time() << " to " << decon_out.rbegin()->get()->get_value_time() << " size " << decon_out.size());
}


void
DeconQueueService::copy_recon_frame_to_container(
        const svr::datamodel::DataRow::container &decon_data,
        const std::set<ptime> &times,
        DataRow::container &recon_data,
        const std::vector<double> &recon_out,
        const size_t limit)
{
    if (recon_out.size() != times.size()) LOG4_THROW("Reconstructed data size " << recon_out.size() << " doesn't equal times size " << times.size());
    LOG4_DEBUG("Copying " << (limit < recon_out.size() ? limit : recon_out.size()) <<
                          " rows with starting value time " << *times.begin() << " ending value time " << *times.rbegin());
    recon_data.erase(lower_bound(recon_data, *times.begin()), recon_data.end());
    const size_t start_out = recon_data.size();
    const size_t loop_end = svr::common::min<size_t>(recon_out.size(), limit, times.size());
    recon_data.resize(start_out + loop_end);
    __tbb_pfor_i (0, loop_end,
        const auto i_time_iter = std::next(times.begin(), i);
        const auto row_iter = find(decon_data, *i_time_iter);
        if (row_iter == decon_data.end())
            LOG4_ERROR("Time " << *i_time_iter << " not found in decon data.");
        recon_data[start_out + i] = std::make_shared<DataRow>(
                *i_time_iter, bpt::second_clock::local_time(), (*row_iter)->get_tick_volume(), std::vector<double>{recon_out[i]})
    )
    LOG4_END();
}


data_row_container DeconQueueService::reconstruct(
        const svr::datamodel::datarow_range &data,
        const std::string &transformation_name,
        const size_t n_decomposition_levels)
{
    data_row_container out_data_container;
    reconstruct(data, transformation_name, n_decomposition_levels, out_data_container);
    return out_data_container;
}


void DeconQueueService::reconstruct(
        const svr::datamodel::datarow_range &data,
        const std::string &transformation_name,
        const size_t n_decomposition_levels,
        data_row_container &out_data_container)
{
    LOG4_BEGIN();
    if (data.distance() < 1) LOG4_THROW("No data to reconstruct.");
    const auto count = data.distance();
    LOG4_DEBUG("Reconstructing " << count << " rows of " << n_decomposition_levels << " levels.");
    std::set<bpt::ptime> times;
    std::mutex mx;
    std::vector<double> recon_values(count);
    // TODO: set times correctly when the output decon size is smaller than the input decon size.
    __par_iter (data.begin(), count,
        double row_val = 0;
        for (size_t half_level_ix = 0; half_level_ix < n_decomposition_levels; half_level_ix += 2) {
            if (half_level_ix != n_decomposition_levels / 2)
                row_val += _ITER->get()->get_value(half_level_ix);
        }
        recon_values[_IX] = row_val;
        LOG4_TRACE("Reconstructed value " << row_val << " for " << _ITER->get()->get_value_time());
        std::scoped_lock l(mx);
        times.insert(_ITER->get()->get_value_time());
    );

    copy_recon_frame_to_container(data.get_container(), times, out_data_container, recon_values);

#if 0
    {
        static size_t call_ct = 0;
        {
            std::stringstream ss;
            for (auto row_iter = out_data_container.begin(); row_iter != out_data_container.end(); ++row_iter) {
                ss << row_iter->get()->to_string() << std::endl;
            }
            std::stringstream ss_name;
            ss_name << "/mnt/faststore/recon_" << call_ct << "_out_data_container.csv";
            LOG4_FILE(ss_name.str(), ss.str());
        }
        {
            std::stringstream ss;
            for (auto row_iter = data.get_container().begin(); row_iter != data.get_container().end(); ++row_iter) {
                ss << row_iter->get()->to_string() << std::endl;
            }
            std::stringstream ss_name;
            ss_name << "/mnt/faststore/recon_" << call_ct << "_in_data_container.csv";
            LOG4_FILE(ss_name.str(), ss.str());
        }
        call_ct++;
    }
#endif

    LOG4_END();
}


size_t
DeconQueueService::load_decon_data(const DeconQueue_ptr &decon_queue, const ptime &time_from, const ptime &time_to, const size_t limit)
{
    reject_nullptr(decon_queue);
    std::deque<DataRow_ptr> new_data = decon_queue_dao.get_data(decon_queue->get_table_name(), time_from, time_to, limit);
    data_row_container &decon_queue_data = decon_queue->get_data();
    if (!new_data.empty() and !decon_queue_data.empty() and new_data.front()->get_value_time() <= decon_queue_data.back()->get_value_time())
        decon_queue_data.erase(lower_bound(decon_queue_data, new_data.front()->get_value_time()), decon_queue_data.end());
    decon_queue_data.insert(decon_queue_data.end(), new_data.begin(), new_data.end());
    return new_data.size();
}


size_t DeconQueueService::load_latest_decon_data(const DeconQueue_ptr &decon_queue, const ptime &time_to, const size_t limit)
{
    LOG4_DEBUG("Loading " << limit << " values until " << time_to << " from decon queue " << decon_queue->get_table_name());
    reject_nullptr(decon_queue);
    std::deque<DataRow_ptr> new_data;
    try {
        new_data = decon_queue_dao.get_latest_data(decon_queue->get_table_name(), time_to, limit);
    } catch (const std::exception &ex) {
        LOG4_WARN("Error loading data from decon queue " << decon_queue->get_table_name() << ". " << ex.what());
        return 0;
    }
    data_row_container &decon_queue_data = decon_queue->get_data();
    if (!new_data.empty() and !decon_queue_data.empty() and new_data.front()->get_value_time() <= decon_queue_data.back()->get_value_time())
        decon_queue_data.erase(lower_bound(decon_queue_data, new_data.front()->get_value_time()), decon_queue_data.end());
    decon_queue_data.insert(decon_queue_data.end(), new_data.begin(), new_data.end());
    LOG4_DEBUG("Retrieved " << new_data.size() << " rows.");
    return new_data.size();
}


int DeconQueueService::clear(const DeconQueue_ptr &decon_queue)
{
    reject_nullptr(decon_queue);
    return decon_queue_dao.clear(decon_queue);
}


long DeconQueueService::count(const DeconQueue_ptr &decon_queue)
{
    reject_nullptr(decon_queue);
    return decon_queue_dao.count(decon_queue);
}


const DeconQueue_ptr &DeconQueueService::find_decon_queue(
        const std::vector<DeconQueue_ptr> &decon_queues,
        const std::string &input_queue_table_name,
        const std::string &input_queue_column_name)
{
    LOG4_BEGIN();
    auto p_decon_queue_iter = find_if(
            decon_queues.begin(),
            decon_queues.end(),
            [&input_queue_table_name, &input_queue_column_name](const DeconQueue_ptr &p_decon_queue)
            {
                return p_decon_queue->get_input_queue_table_name() == input_queue_table_name &&
                       p_decon_queue->get_input_queue_column_name() == input_queue_column_name;
            }
    );

    if (p_decon_queue_iter == decon_queues.end())
        THROW_EX_FS(std::invalid_argument, "Couldn't not find decon queue for input table name " << input_queue_table_name <<
                      ", input column " << input_queue_column_name << ", decon queues count " << decon_queues.size());

    return *p_decon_queue_iter;
}

const DeconQueue_ptr &DeconQueueService::find_decon_queue(
        const std::vector<DeconQueue_ptr> &decon_queues, const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Looking for " << decon_queue_table_name);

    auto p_decon_queue_iter = find_if(
            decon_queues.begin(),
            decon_queues.end(),
            [&decon_queue_table_name](const DeconQueue_ptr &p_decon_queue)
            {
                return p_decon_queue->get_table_name() == decon_queue_table_name;
            }
    );

    if (p_decon_queue_iter == decon_queues.end())
        THROW_EX_FS(std::invalid_argument, "Couldn't not find decon queue for table name " << decon_queue_table_name << ", decon queues count " << decon_queues.size());

    return *p_decon_queue_iter;
}


} // business
} // svr
