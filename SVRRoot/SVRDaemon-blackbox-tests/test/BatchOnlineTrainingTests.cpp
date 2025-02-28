#include "include/DaoTestFixture.h"
#include "DAO/RequestDAO.hpp"
#include "model/Request.hpp"

namespace svr {

constexpr char C_test_primary_column[] = "xauusd_avg_bid"; // Ignore tuning or validating other input queue columns in case of aux columns

#define FORECAST_COUNT (1U)

#define BATCH_TRAIN_QUEUE ("q_svrwave_xauusd_avg_3600")
#define BATCH_TRAIN_AUX_QUEUE ("q_svrwave_xauusd_avg_1")
#define NEW_VALUES_QUEUE ("q_svrwave_xauusd_new_avg_3600")
#define NEW_VALUES_AUX_QUEUE ("q_svrwave_xauusd_avg_1")
#define NEW_VALUES_QUEUE_OHLC ("q_svrwave_xauusd_new_3600")


const bigint default_dataset_id = 100;

void
update_with_new_data(const datamodel::InputQueue_ptr &iq, const data_row_container::iterator &new_iq_iter);

void
prepare_forecast_request(const datamodel::InputQueue_ptr &iq, const bpt::ptime &start_predict_time);

std::tuple<double, double, double, double>
verify_results(
        const bpt::ptime &forecast_start_time,
        const datamodel::InputQueue_ptr &iq,
        const datamodel::InputQueue_ptr &new_iq_ohlc_aux);


TEST_F(DaoTestFixture, backtest_xauusd)
{
    LOG4_INFO("Do performance tests: " << DaoTestFixture::DoPerformanceTests);

    tdb.prepareSvrConfig(tdb.TestDbUserName, tdb.dao_type, -1);

    datamodel::Dataset_ptr p_dataset = aci.dataset_service.load(default_dataset_id);
    datamodel::InputQueue_ptr iq = aci.input_queue_service.get_queue_metadata(BATCH_TRAIN_QUEUE);
    iq->set_data(aci.input_queue_service.load(BATCH_TRAIN_QUEUE));
//    datamodel::InputQueue_ptr iq_aux = aci.input_queue_service.get_queue_metadata(BATCH_TRAIN_AUX_QUEUE);
//    iq_aux->set_data(aci.input_queue_service.load(BATCH_TRAIN_AUX_QUEUE));

    datamodel::InputQueue_ptr new_iq = aci.input_queue_service.get_queue_metadata(NEW_VALUES_QUEUE);
    new_iq->set_data(aci.input_queue_service.load(NEW_VALUES_QUEUE));
    datamodel::InputQueue_ptr new_iq_aux = aci.input_queue_service.get_queue_metadata(NEW_VALUES_AUX_QUEUE);
    new_iq_aux->set_data(aci.input_queue_service.load(NEW_VALUES_AUX_QUEUE));
    // new_iq_aux->get_data().erase(new_iq_aux->begin(), lower_bound(new_iq_aux->get_data(), new_iq->front()->get_value_time()));
    //InputQueue_ptr new_ohlc_iq = aci.input_queue_service.get_queue_metadata(NEW_VALUES_QUEUE_OHLC);
    //new_ohlc_iq->set_data(aci.input_queue_service.load(NEW_VALUES_QUEUE_OHLC));

// Setting up daemon communication
    diagnostic_interface_alpha di;
    tdb.run_daemon_nowait();
    di.finish_construction();
    double total_mae = 0, total_mae_ohlc = 0, total_last_known_mae = 0, total_last_known_ohlc = 0;
    size_t ctr = 0, hit_ctr = 0, last_known_hit_ctr = 0;
    for (auto new_iq_iter = new_iq->begin(); new_iq_iter != new_iq->end(); ++new_iq_iter)
    {
        const auto t_start = std::chrono::high_resolution_clock::now();
        LOG4_DEBUG("Start iteration, training until " << iq->get_data().back()->get_value_time());

        di.trigger_next_iteration(); di.wait_iteration_finished();
        di.trigger_next_iteration(); di.wait_iteration_finished();

        bpt::ptime pred_time = new_iq_iter->get()->get_value_time();
        prepare_forecast_request(iq, pred_time);

        di.trigger_next_iteration(); di.wait_iteration_finished();
        di.trigger_next_iteration(); PROFILE_EXEC_TIME(di.wait_iteration_finished(), "Predicting " << pred_time);

        double mae = 0, mae_ohlc = 0, last_known_mae = 0, last_known_ohlc = 0;
        std::tie( mae, mae_ohlc, last_known_mae, last_known_ohlc) = verify_results(pred_time, new_iq, new_iq_aux);

        if (mae != common::C_bad_validation and mae_ohlc != common::C_bad_validation) {
            total_mae += mae;
            total_mae_ohlc += mae_ohlc;
            total_last_known_mae += last_known_mae;
            total_last_known_ohlc += last_known_ohlc;
            if (mae_ohlc < std::numeric_limits<double>::epsilon()) ++hit_ctr;
            if (last_known_ohlc < std::numeric_limits<double>::epsilon()) ++last_known_hit_ctr;
            ++ctr;
            const double cur_mae = total_mae / double(ctr);
            const double cur_last_mae = total_last_known_mae / double(ctr);
            const double last_hit_pct = 100. * double(last_known_hit_ctr) / double(ctr);
            const double hit_rate_pct = 100. * double(hit_ctr) / double(ctr);
            LOG4_INFO("Cycle " << ctr << " trained until " << iq->get_data().back()->get_value_time() << ", predicted " << pred_time << ", hit rate " << hit_rate_pct << " pct, last known hit rate " << last_hit_pct << " pct, MAE " << cur_mae << ", last known MAE " << cur_last_mae << ", alpha " << 100. * (cur_last_mae/cur_mae - 1.) << " pct, MAE OHLC " << double(total_mae_ohlc) / double(ctr) << ", last known MAE OHLC " << double(total_last_known_ohlc) / double(ctr));
        } else {
            LOG4_ERROR("Forecasts validation failed for cycle at " << pred_time);
        }

        update_with_new_data(iq, new_iq_iter);

        const auto t_elapsed = std::chrono::duration<double, std::ratio<1,1>>(std::chrono::high_resolution_clock::now() - t_start).count();
        LOG4_INFO("Cycle at " << new_iq_iter->get()->get_value_time() << " took " << t_elapsed << " secs.");
    }
    LOG4_INFO("Mean absolute error of " << ctr << " comparisons is " << total_mae / double(ctr) << ".");
}


void
update_with_new_data(const datamodel::InputQueue_ptr &iq, const data_row_container::iterator &new_iq_iter)
{
    LOG4_DEBUG("Adding new datarow for time " << new_iq_iter->get()->get_value_time());
    const auto prev_last_time = iq->get_data().back()->get_value_time() + iq->get_resolution() - bpt::seconds(1);
    iq->get_data().emplace_back(*new_iq_iter);
    APP.input_queue_service.save(iq, prev_last_time);
    APP.flush_dao_buffers();
}


void
prepare_forecast_request(const datamodel::InputQueue_ptr &iq, const bpt::ptime &start_predict_time)
{
    datamodel::MultivalRequest_ptr p_request = ptr<datamodel::MultivalRequest>(
            bigint(0),
            iq->get_owner_user_name(),
            default_dataset_id,
            bpt::second_clock::local_time(),
            start_predict_time,
            start_predict_time + iq->get_resolution() * FORECAST_COUNT,
            iq->get_resolution(),
            common::formatter() << "{" << C_test_primary_column << "}");
    APP.request_service.save(p_request);

    APP.flush_dao_buffers();
}



data_row_container
get_results(const bpt::ptime &request_time, const datamodel::InputQueue_ptr &iq, const std::string &column_name)
{
    const auto results = APP.request_service.get_multival_results(
            iq->get_owner_user_name(),
            column_name,
            default_dataset_id,
            request_time,
            request_time + iq->get_resolution() * FORECAST_COUNT,
            iq->get_resolution());
    return datamodel::DataRow::construct(results);
}


std::pair<double, double>
compare_by_value_mean_error(
        const data_row_container &forecasts,
        const data_row_container &etalon,
        const size_t etalon_col = 0)
{
    double res = 0.;
    int n_items = 0;
    for (auto forecast_row = forecasts.begin(); forecast_row != forecasts.end(); ++forecast_row) {
        if (forecast_row->get()->get_values().empty()) {
            LOG4_ERROR("Forecasts row for time " << forecast_row->get()->get_value_time() << " is empty, skipping.");
            continue;
        }
        const auto etalon_row = find(etalon, forecast_row->get()->get_value_time());
        if (etalon_row == etalon.end()) {
            LOG4_ERROR("Not found etalon row for time " << forecast_row->get()->get_value_time());
            continue;
        }
        if (etalon_row->get()->get_values().size() < etalon_col) {
            LOG4_ERROR("Not found etalon row data for column " << etalon_col);
            continue;
        }

        const auto forecast_val = forecast_row->get()->get_value(0);
        const auto etalon_val = etalon_row->get()->get_value(etalon_col);
        const auto abs_diff = std::abs(forecast_val - etalon_val);
        res += abs_diff;
        LOG4_INFO("Etalon col " << etalon_col << " time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", error " << abs_diff << ", predicted " << forecast_val << ", etalon avg " << etalon_val);
        ++n_items;
    }
    if (n_items > 0) res /= double(n_items);
    else res = common::C_bad_validation;

    LOG4_DEBUG("Compared " << n_items << " values, mean absolute error is " << res << " column " << etalon_col);
    return {res, 0};
}



std::pair<double, double>
compare_by_value_mean_erroraux(
        const data_row_container &forecasts,
        const data_row_container &etalon,
        const bpt::time_duration &forecast_resolution)
{
    double mae = 0, last_known_mae = 0;
    int n_items = 0;
    for (auto forecast_row = forecasts.begin(); forecast_row != forecasts.end(); ++forecast_row) {
        const auto forecast_time = forecast_row->get()->get_value_time();
        if (forecast_row->get()->get_values().empty()) {
            LOG4_ERROR("Forecasts row for time " << forecast_time << " is empty, skipping.");
            continue;
        }

        const auto it_label_start = lower_bound(etalon, forecast_time);
        const auto last_known_iter = lower_bound_back(etalon, it_label_start, forecast_time - forecast_resolution * PROPS.get_prediction_horizon());
        const double last_known = last_known_iter == etalon.end() ? common::C_bad_validation : std::prev(last_known_iter)->get()->get_value(0);
        const auto last_known_time = std::prev(last_known_iter)->get()->get_value_time();
        const auto etalon_val = 0;// TODO Port generate_twap(std::prev(it_label_start), etalon.end(), forecast_time, forecast_time + forecast_resolution, onesec, 0);
        const auto forecast_val = forecast_row->get()->get_value(0);
        const auto abs_diff = std::abs(forecast_val - etalon_val);
        const auto last_known_diff = std::abs(last_known - etalon_val);
        mae += abs_diff;
        last_known_mae += last_known_diff;
        LOG4_INFO("Forecast time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", error " << abs_diff << ", predicted " << forecast_val << ", etalon avg " << etalon_val);
        LOG4_INFO("Forecast time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", error " << last_known_diff << ", last known price " << last_known << ", last known time " << last_known_time << ", etalon avg " << etalon_val);
        ++n_items;
    }
    if (n_items > 0) {
        mae /= double(n_items);
        last_known_mae /= double(n_items);
    } else {
        mae = common::C_bad_validation;
        last_known_mae = common::C_bad_validation;
    }

    LOG4_DEBUG("Compared " << n_items << " values, mean absolute error is " << mae << " last known mean absolute error is " << last_known_mae);
    return {mae, last_known_mae};
}


std::pair<double, double>
compare_by_value_error_ohlc(
        const data_row_container &forecasts,
        const data_row_container &etalon)
{
    double mae = 0.;
    int n_items = 0;
    for (auto forecast_row = forecasts.begin(); forecast_row != forecasts.end(); ++forecast_row) {
        if (forecast_row->get()->get_values().empty()) {
            LOG4_ERROR("Forecasts row for time " << forecast_row->get()->get_value_time() << " is empty, skipping.");
            continue;
        }
        const auto etalon_row = find(etalon, forecast_row->get()->get_value_time());
        if (etalon_row == etalon.end()) {
            LOG4_ERROR("Not found etalon row for time " << forecast_row->get()->get_value_time());
            continue;
        }

        const auto forecast_val = forecast_row->get()->get_value(0);
        const auto etalon_max = *std::max_element(etalon_row->get()->get_values().begin(), etalon_row->get()->get_values().end());
        const auto etalon_min = *std::min_element(etalon_row->get()->get_values().begin(), etalon_row->get()->get_values().end());
        double err = 0.;
        if (forecast_val < etalon_min) err = etalon_min - forecast_val;
        else if (forecast_val > etalon_max) err = forecast_val - etalon_max;
        mae += err;
        LOG4_INFO("Time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", error " << err << ", predicted " << forecast_val << ", etalon high " << etalon_max << ", etalon low " << etalon_min);
        ++n_items;
    }
    if (n_items > 0) mae /= double(n_items); else mae = common::C_bad_validation;

    LOG4_DEBUG("Compared " << n_items << " values, mean absolute error is " << mae);
    return {mae, 0};
}


std::pair<double, double>
compare_by_value_error_ohlcaux(
        const data_row_container &forecasts,
        const data_row_container &etalon,
        const bpt::time_duration &forecast_resolution)
{
    double mae = 0, last_known_mae = 0;
    int n_items = 0;
    for (auto forecast_row = forecasts.begin(); forecast_row != forecasts.end(); ++forecast_row) {
        if (forecast_row->get()->get_values().empty()) {
            LOG4_ERROR("Forecasts row for time " << forecast_row->get()->get_value_time() << " is empty, skipping.");
            continue;
        }
        std::vector<double> etalon_aux_vals;
        auto etalon_iter = lower_bound(etalon, forecast_row->get()->get_value_time());
        if (etalon_iter == etalon.end()) {
            LOG4_ERROR("Not found etalon row for time " << forecast_row->get()->get_value_time());
            continue;
        }
        for (;etalon_iter->get()->get_value_time() < forecast_row->get()->get_value_time() + forecast_resolution; ++etalon_iter)
            etalon_aux_vals.push_back(etalon_iter->get()->get_value(0));
        if (etalon_aux_vals.size() < 5) {
            LOG4_WARN("Not found enough etalon values " << etalon_aux_vals.size());
            continue;
        }
        const auto etalon_max = *std::max_element(etalon_aux_vals.begin(), etalon_aux_vals.end());
        const auto etalon_min = *std::min_element(etalon_aux_vals.begin(), etalon_aux_vals.end());

        const auto last_known_time = forecast_row->get()->get_value_time() - forecast_resolution * PROPS.get_prediction_horizon();
        const auto last_known_iter = lower_bound_back_before(etalon, etalon_iter, last_known_time);
        const double last_known = last_known_iter == etalon.end() ? common::C_bad_validation : last_known_iter->get()->get_value(0);

        const auto forecast_val = forecast_row->get()->get_value(0);
        double err = 0, last_known_err = 0;
        if (forecast_val < etalon_min) err = etalon_min - forecast_val;
        else if (forecast_val > etalon_max) err = forecast_val - etalon_max;
        mae += err;

        if (last_known < etalon_min) last_known_err = etalon_min - last_known;
        else if (last_known > etalon_max) last_known_err = last_known - etalon_max;
        last_known_mae += last_known_err;

        LOG4_INFO("Time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", error " << err << ", predicted " << forecast_val << ", etalon high " << etalon_max << ", etalon low " << etalon_min);
        LOG4_INFO("Time " << forecast_row->get()->get_value_time() << ", position " << n_items << ", last known error " << last_known_err << ", predicted " << last_known << ", last known time " << last_known_time << ", etalon high " << etalon_max << ", etalon low " << etalon_min);
        ++n_items;
    }
    if (n_items > 0) {
        mae /= double(n_items);
        last_known_mae /= double(n_items);
    }
    else {
        mae = common::C_bad_validation;
        last_known_mae = common::C_bad_validation;
    }

    LOG4_DEBUG("Compared " << n_items << " values, mean absolute error is " << mae);
    return {mae, last_known_mae};
}


std::tuple<double, double, double, double>
verify_results(
        const bpt::ptime &forecast_start_time,
        const datamodel::InputQueue_ptr &new_iq,
        const datamodel::InputQueue_ptr &new_iq_ohlc_aux)
{
    const auto forecasts = get_results(forecast_start_time, new_iq, C_test_primary_column);
    if (forecasts.empty()) {
        LOG4_ERROR("Forecasted data for column xauusd_avg_bid is zero!");
        return {common::C_bad_validation, common::C_bad_validation, common::C_bad_validation, common::C_bad_validation};
    }
    LOG4_TRACE("Forecasted data is of size " << forecasts.size());
    double mae, mae_ohlc, last_known_mae, last_known_ohlc;
    std::tie(mae, last_known_mae) = compare_by_value_mean_erroraux(forecasts, new_iq_ohlc_aux->get_data(), new_iq->get_resolution());
    std::tie(mae_ohlc, last_known_ohlc) = compare_by_value_error_ohlcaux(forecasts, new_iq_ohlc_aux->get_data(), new_iq->get_resolution());
    LOG4_INFO("Mean absolute error to OHLC " << mae_ohlc << ", mean absolute error to average " << mae << ", last known OHLC " << last_known_ohlc << ", last known MAE " <<
        last_known_mae << ", column " << C_test_primary_column << ", time " << forecast_start_time);

    return {mae, mae_ohlc, last_known_mae, last_known_ohlc};
}

}
