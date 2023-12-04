#include <unistd.h>
#include <cstdlib>
#include <syslog.h>
#include <sys/stat.h>
#include <csignal>
#include <queue>
#include <stdexcept>


#include "common/Logging.hpp"
#include "util/TimeUtils.hpp"
#include "appcontext.hpp"
#include "DaemonFacade.hpp"
#include <model/Request.hpp>


using namespace svr::datamodel;
using namespace svr::dao;
using namespace svr::business;
using namespace svr::context;
using namespace svr::common;
using namespace svr::daemon;
using namespace bpt;


namespace svr::daemon {


std::string DaemonFacade::S_MAX_LOOP_COUNT = "MAX_LOOP_COUNT";
std::string DaemonFacade::S_DECONSTRUCTING_FRAME_SIZE = "DECONSTRUCTING_FRAME_SIZE";
std::string DaemonFacade::S_FILL_MISSING_QUEUE_VALUES = "FILL_MISSING_QUEUE_VALUES";
std::string DaemonFacade::S_LOOP_INTERVAL = "LOOP_INTERVAL_MS";
std::string DaemonFacade::S_DAEMONIZE = "DAEMONIZE";


DaemonFacade::DaemonFacade(const std::string &app_properties_path) : loop_count(0)
{
    initialize(app_properties_path);
}

void DaemonFacade::initialize(const std::string &app_properties_path)
{
    // parse necessary daemon configuration parameters
    loop_interval = AppContext::get_instance().app_properties.get_property<long>(app_properties_path, S_LOOP_INTERVAL, "1000");
    daemonize = AppContext::get_instance().app_properties.get_property<bool>(app_properties_path, S_DAEMONIZE, "1");
    max_loop_count = AppContext::get_instance().app_properties.get_property<size_t>(app_properties_path, S_MAX_LOOP_COUNT, "-1");
}

void DaemonFacade::uninitialize()
{
}

void DaemonFacade::do_fork()
{
    pid_t pid;

    /* Catch, ignore and handle signals */
    //TODO: Implement a working signal handler */
    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    /* Fork off the parent process */
    pid = fork();
    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    /* If we got a good PID, then
    we can exit the parent process.
    */
    if (pid > 0) {
        LOG4_INFO("Exiting parent process");
        exit(EXIT_SUCCESS);
    }

    /* Open the log file */
    openlog("SVRDaemon", LOG_PID, LOG_DAEMON);
    syslog(LOG_NOTICE, "Daemon process started");

    /* Change the file mode masks */
    umask(0);

    /* Create a new SID for the child process */
    if (setsid() < 0) {
        /* Log the failure */
        syslog(LOG_ERR, "Cannot create new SID!");
        exit(EXIT_FAILURE);
    }

    /* Close out the standard file descriptors */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wembedded-directive"

struct save_response
{
    const std::string &value_column;
    const MultivalRequest_ptr &p_request;

    save_response(const std::string &value_column_, const MultivalRequest_ptr &p_request_) : value_column(value_column_), p_request(p_request_)
    {}

    void operator()(const std::shared_ptr<svr::datamodel::DataRow> &p_result_row)
    {
        if (p_result_row->get_value_time() < p_request->value_time_start or p_result_row->get_value_time() > p_request->value_time_end) {
            LOG4_DEBUG("Skipping save p_response " << p_result_row->to_string());
            return;
        }

        auto resp_val = p_result_row->get_value(0);
        const auto resp_time = p_result_row->get_value_time();
#ifndef OFFSET_PRED_MUL
        // Small hack for improving performance consequent p_predictions
        {
            const auto input_col_ix = InputQueueService::get_value_column_index(p_input_queue, value_column);
            static std::atomic<data_row_container::iterator> aux_hint{p_aux_input_queue->get_data().begin()};
            const auto last_minute_pc = std::prev(
                    lower_bound(p_aux_input_queue->get_data(), aux_hint.load(), resp_time))->second->get_value(input_col_ix);
            const auto last_hour_pc = std::prev(
                    p_input_queue->get_data().lower_bound(resp_time))->second->get_value(input_col_ix);

            if (last_minute_pc < last_hour_pc != resp_val > last_hour_pc) resp_val = last_minute_pc;
            aux_hint.store(last_minute_pc);
        }
#endif
        MultivalResponse_ptr p_response = std::make_shared<MultivalResponse>(0, p_request->get_id(), resp_time, value_column, resp_val);
        LOG4_TRACE("Saving " << p_response->to_string());
        AppContext::get_instance().request_service.save(p_response); // Requests get marked processed here
    }
};

// By specification, a multival request's time period to predict is
// [start_predict_time, end_predict_time) i.e. right-hand exclusive.
// TODO Rewrite, fully parallelize and move to Dataset service
void DaemonFacade::process_mimo_multival_requests(const User_ptr &p_user, Dataset_ptr &p_dataset)
{
    LOG4_DEBUG("Processing " << p_user->to_string() << " requests for dataset " << p_dataset->to_string());
    InputQueue_ptr p_input_queue = p_dataset->get_input_queue();
    const auto resolution = p_input_queue->get_resolution();
    const auto main_to_aux_period_ratio = resolution / p_dataset->get_aux_input_queue(0)->get_resolution();

    auto active_requests = AppContext::get_instance().request_service.get_active_multival_requests(
            *p_user, *p_dataset, *p_input_queue);
    if (active_requests.empty()) LOG4_DEBUG("No active requests!");
    __tbb_pfor(req_ix, 0, active_requests.size(),
        const MultivalRequest_ptr p_active_request = active_requests[req_ix];
        if (!p_active_request->sanity_check()) {
            LOG4_ERROR("Request " << p_active_request->to_string() << " incorrect!");
            continue;
        }
        const auto start_predict_time = p_active_request->value_time_start;
        const auto end_predict_time = p_active_request->value_time_end;
        LOG4_DEBUG("Processing request " << p_active_request->to_string());
        std::atomic_bool request_answered = true;
        const auto request_columns = from_sql_array(p_active_request->value_columns);
        const auto prediction_features_period = boost::posix_time::time_period(
                context::AppContext::get_instance().input_queue_service.get_nth_last_row(
                        p_dataset->get_aux_input_queue(0), QUANTIZE_FIXED * 2 * p_dataset->get_max_lag_count(), start_predict_time)->get_value_time(),
                end_predict_time);

        auto predict_dequeues = DeconQueueService::extract_copy_data(p_dataset, prediction_features_period);
        std::set<bpt::ptime> predict_start_times_grid;
        if (start_predict_time == end_predict_time)
            predict_start_times_grid.insert(start_predict_time);
        else
            for (auto current_time = start_predict_time; current_time < end_predict_time; current_time += MULTISTEP_PRED_PERIOD)
                predict_start_times_grid.insert(current_time);

        if (predict_dequeues.empty() || predict_start_times_grid.empty()) {
            LOG4_ERROR("Predict decon queues or times grid are empty.");
            continue;
        }

        const auto lev_ct = p_dataset->get_transformation_levels();
        std::map<bpt::ptime, std::vector<std::vector<arma::rowvec>>> pred_features;
        // Prepare per ensemble features
        __tbb_pfor(time_ix, 0, predict_start_times_grid.size(),
               const auto pred_time = *std::next(predict_start_times_grid.begin(), time_ix);
               pred_features[pred_time] = std::vector<std::vector<arma::rowvec>>(predict_start_times_grid.size());
               __tbb_pfor(ens_ix, 0, request_columns.size(),
                   pred_features[pred_time][ens_ix].resize(lev_ct);
                   const auto p_ensemble = p_dataset->get_ensemble(request_columns[ens_ix]);
                   if (!p_ensemble) {
                       LOG4_ERROR("Ensemble for request column " << request_columns[ens_ix] << " not found.");
                       request_answered &= false;
                       continue;
                   }
                   const auto p_aux_decon_queue_predicted = DeconQueueService::find_decon_queue(predict_dequeues, p_ensemble->get_aux_decon_queue()->get_table_name());
                   const auto aux_dq_scaling_factors = APP.dq_scaling_factor_service.slice(p_dataset, p_aux_decon_queue_predicted);
                   __tbb_spfor(mod_ix, 0, lev_ct, 2,
                        if (mod_ix == lev_ct / 2) continue;
                        APP.model_service.get_features_row(
                               pred_time,
                               p_aux_decon_queue_predicted->get_data(),
                               aux_dq_scaling_factors,
                               p_ensemble->get_model(mod_ix)->get_learning_levels(),
                               mod_ix,
                               p_dataset->get_max_lookback_time_gap(),
                               main_to_aux_period_ratio,
                               p_ensemble->get_model(mod_ix)->get_svr_model()->get_svr_parameters().get_lag_count(),
                               p_input_queue->get_resolution(),
                               pred_features[pred_time][ens_ix][mod_ix]);
                        request_answered &= !pred_features[pred_time][ens_ix][mod_ix].empty();
                   )
               )
        )
        if (!request_answered) {
            LOG4_ERROR("Skipping request " << p_active_request->get_id());
            continue;
        }

        // Join rows of prediction features for all dataset ensembles (for aux input queue feature)
        __tbb_pfor(time_ix, 0, predict_start_times_grid.size(),
            __tbb_spfor(mod_ix, 0, lev_ct, 2,
                if (mod_ix == lev_ct / 2) continue;
                const auto pred_time = *std::next(predict_start_times_grid.begin(), time_ix);
                for (size_t ens_ix = 1; ens_ix < p_dataset->get_ensembles().size(); ++ens_ix) { // Features past ens_ix 0 become invalid // Not parallelizable
                    pred_features[pred_time][0][mod_ix].insert_cols(pred_features[pred_time][0][mod_ix].n_cols, pred_features[pred_time][ens_ix][mod_ix]);
                    pred_features[pred_time][ens_ix][mod_ix].clear();
                }
            )
        )

        // Do actual p_predictions
        std::mutex ins_lck;
        __pxt_pfor(time_ix, 0, predict_start_times_grid.size(),
           const auto pred_time = *std::next(predict_start_times_grid.begin(), time_ix);
           __pxt_pfor(col_ix, 0, request_columns.size(),
                const auto p_ensemble = p_dataset->get_ensemble(request_columns[col_ix]);
                if (!p_ensemble) {
                    LOG4_ERROR("Ensemble for request column " << request_columns[col_ix] << " not found.");
                    request_answered &= false;
                    continue;
                }
                arma::mat predictions(1, lev_ct, arma::fill::zeros);
                __tbb_spfor(level_ix, 0, lev_ct, 2,
                    if (level_ix == lev_ct / 2) continue;
                    const auto aux_dq_scaling_factors = APP.dq_scaling_factor_service.slice(p_dataset->get_dq_scaling_factors(),
                        p_dataset->get_id(), p_ensemble->get_aux_decon_queue()->get_input_queue_table_name(), p_ensemble->get_aux_decon_queue()->get_input_queue_column_name(), p_ensemble->get_model(level_ix)->get_learning_levels());
                    predictions(0, level_ix) = APP.dq_scaling_factor_service.unscale_decon_prediction(
                        arma::as_scalar(arma::mean(p_ensemble->get_model(level_ix)->get_svr_model()->chunk_predict(pred_features[pred_time][0][level_ix], pred_time), 1)),
                    level_ix, aux_dq_scaling_factors);
                )
                auto p_aux_decon = DeconQueueService::find_decon_queue(predict_dequeues, p_ensemble->get_aux_decon_queue()->get_table_name());
                const std::scoped_lock l(ins_lck);
                datamodel::DataRow::insert_rows(p_aux_decon->get_data(), predictions, pred_time, resolution);
           )
        )
        if (!request_answered) {
            LOG4_ERROR("Skipping request " << p_active_request->get_id());
            continue;
        }

        __tbb_pfor(col_ix, 0, request_columns.size(),
            const auto p_ensemble = p_dataset->get_ensemble(request_columns[col_ix]);
            if (!p_ensemble) {
                LOG4_ERROR("Ensemble for request column " << request_columns[col_ix] << " not found.");
                request_answered &= false;
                continue;
            }
            const auto reconstructed_data = APP.decon_queue_service.reconstruct(
                DeconQueueService::find_decon_queue(predict_dequeues, p_ensemble->get_aux_decon_queue()->get_table_name())->get_data(),
                p_dataset->get_transformation_name(),
                p_dataset->get_transformation_levels());

            if (reconstructed_data.empty()) {
                LOG4_ERROR("Empty reconstructed data.");
                request_answered &= false;
                continue;
            }
            LOG4_DEBUG("Saving reconstructed p_predictions from " << reconstructed_data.begin()->get()->get_value_time() << " until " << reconstructed_data.rbegin()->get()->get_value_time());
            for_each(find(reconstructed_data, start_predict_time), reconstructed_data.end(), save_response(request_columns[col_ix], p_active_request));
        )

        if (request_answered)
            APP.request_service.force_finalize(p_active_request);
        else
            LOG4_WARN("Failed finalizing request id " << p_active_request->get_id());
    )
}

#pragma GCC diagnostic pop

void DaemonFacade::start_loop()
{
    // configure the daemon
    if (daemonize) {
        do_fork();
        LOG4_INFO("Daemon process started");
    }

    syslog(LOG_NOTICE, "Daemon loop started");
    svr::business::DatasetService::UserDatasetPairs datasets;

    diagnostic_interface_zwei di;

    size_t ctr = 0;
    // Run normal daemon cycle
    while (continue_loop()) {
        if (!(ctr++ % 10)) APP.dataset_service.update_active_datasets(datasets);
        for (svr::business::DatasetService::DatasetUsers &dsu: datasets) {
            di.wait();
            try {
                PROFILE_EXEC_TIME(DatasetService::process_dataset(dsu.dataset), "Process dataset");
                for (const auto &user: dsu.users)
                    PROFILE_EXEC_TIME(process_mimo_multival_requests(user, dsu.dataset), "MIMO Process multival requests " << user->get_user_name());
            } catch (const svr::common::bad_model &ex) {
                LOG4_ERROR("Bad model while processing dataset " << dsu.dataset->get_id() << " " << dsu.dataset->get_dataset_name()
                            << ". Level " << ex.get_decon_level() << ", column " << ex.get_column_name() << ". Error " << ex.what());
            } catch (const std::exception &ex) {
                LOG4_ERROR("Failed processing dataset " << dsu.dataset->get_id() << " " << dsu.dataset->get_dataset_name() << " " << ex.what());
            }
        }
    }
}


bool DaemonFacade::continue_loop()
{
    // wait some time before next iteration
    std::this_thread::sleep_for(std::chrono::milliseconds(loop_interval));
    if (max_loop_count < 0) return true;
    else return (++loop_count) < max_loop_count;
}


/*
 * SVRDaemon-whitebox-test related logic follows below.
 */

struct diagnostic_interface_zwei::diagnostic_interface_impl
{
    static const std::string their_pipe_name, mine_pipe_name;

    std::ifstream their_pipe;
    std::ofstream mine_pipe;

    diagnostic_interface_impl(std::ifstream &&their_pipe, std::ofstream &&mine_pipe)
            : their_pipe(std::move(their_pipe)), mine_pipe(std::move(mine_pipe))
    {}

    ~diagnostic_interface_impl()
    {
        if (!getenv("BACKTEST")) return;

        mine_pipe.close();
        std::remove(mine_pipe_name.c_str());
    }

    static std::ifstream get_their_stream()
    {
        return std::ifstream(their_pipe_name.c_str());
    }

    static std::ofstream get_mine_stream()
    {
        return std::ofstream(mine_pipe_name.c_str());
    }

    virtual void wait()
    {
        if (!getenv("BACKTEST")) return;

        mine_pipe << "S: iteration finished" << std::endl;

        std::string command;
        while (command.empty()) {
            if (their_pipe.eof()) their_pipe.clear(std::ios::eofbit);

            std::getline(their_pipe, command);
            if (command.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (command == "M: proceed to next iteration")
                return;

            if (command == "M: close session")
                exit(0);
        }
    };
};

std::string const diagnostic_interface_zwei::diagnostic_interface_impl::their_pipe_name = "/tmp/SVRDaemon_di_alpha";
std::string const diagnostic_interface_zwei::diagnostic_interface_impl::mine_pipe_name = "/tmp/SVRDaemon_di_zwei";


diagnostic_interface_zwei::diagnostic_interface_zwei()
{
    if (!getenv("BACKTEST")) return;

    std::ifstream pipe(diagnostic_interface_impl::get_their_stream());
    if (!pipe.good()) return;

    std::string command;
    std::getline(pipe, command);

    if (command != "M: start session") return;

    pimpl = std::shared_ptr<diagnostic_interface_impl>(
            new diagnostic_interface_impl(std::move(pipe), diagnostic_interface_impl::get_mine_stream()));
}


void diagnostic_interface_zwei::wait()
{
    if (!getenv("BACKTEST")) return;

    if (pimpl) pimpl->wait();
}


}
