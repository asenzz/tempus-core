#include <unistd.h>
#include <cstdlib>
#include <syslog.h>
#include <sys/stat.h>
#include <csignal>
#include <queue>
#include <stdexcept>


#include "common/logging.hpp"
#include "util/time_utils.hpp"
#include "appcontext.hpp"
#include "DaemonFacade.hpp"


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

void DaemonFacade::start_loop()
{
    // configure the daemon
    if (daemonize) {
        do_fork();
        LOG4_INFO("Daemon process started");
    }

    syslog(LOG_NOTICE, "Daemon loop started");
    svr::business::DatasetService::UserDatasetPairs datasets;
#ifdef INTEGRATION_TEST
    diagnostic_interface_zwei di;
#endif
    const auto self_request = PROPS.get_self_request();
    uint8_t ctr = 0;
    PROFILE_EXEC_TIME(APP.dataset_service.update_active_datasets(datasets), "Update active datasets");
    // Run normal daemon cycle
    while (continue_loop()) {
        if (++ctr == 0) PROFILE_EXEC_TIME(APP.dataset_service.update_active_datasets(datasets), "Update active datasets");
        for (svr::business::DatasetService::DatasetUsers &dsu: datasets) {
#ifdef INTEGRATION_TEST
            di.wait();
#endif
            auto &dataset = *dsu.p_dataset;
            try {
                std::unordered_map<bigint, std::deque<datamodel::MultivalRequest_ptr>> user_requests;
                for (const auto &p_user: dsu.users)
                    user_requests[p_user->get_id()] = context::AppContext::get_instance().request_service.get_active_multival_requests(*p_user, dataset);
                PROFILE_EXEC_TIME(DatasetService::process(*dsu.p_dataset), "Process dataset");

                const auto resolution = dataset.get_input_queue()->get_resolution();
                const auto last_aux_time = dataset.get_aux_input_queue()->back()->get_value_time();
                const auto last_input_res_2 = dataset.get_input_queue()->back()->get_value_time() + (resolution * 2.);
                datamodel::MultivalRequest_ptr p_self_request;
                if (self_request && dataset.get_aux_input_queue()->size() &&
                    last_aux_time - dataset.get_last_self_request() >= resolution &&
                    last_aux_time >= last_input_res_2 - (resolution * PROPS.get_prediction_horizon()))
                {
                    p_self_request = otr<datamodel::MultivalRequest>(
                            0, "", dataset.get_id(), bpt::second_clock::local_time(),
                            last_input_res_2, last_input_res_2 + resolution,
                            resolution, dataset.get_input_queue()->get_value_column_str());
                    dataset.set_last_self_request(last_aux_time);
                }

                for (const auto &p_user: dsu.users) {
                    if (p_self_request) user_requests[p_user->get_id()].emplace_back(p_self_request);
                    PROFILE_EXEC_TIME(DatasetService::process_requests(*p_user, *dsu.p_dataset, user_requests[p_user->get_id()]),
                                      "Process multival requests " << p_user->get_user_name());
                }
            } catch (const svr::common::bad_model &ex) {
                LOG4_ERROR("Bad model while processing dataset " << dsu.p_dataset->get_id() << " " << dsu.p_dataset->get_dataset_name()
                                     << ". Level " << ex.get_decon_level() << ", column " << ex.get_column_name() << ". Error " << ex.what());
            } catch (const std::exception &ex) {
                LOG4_ERROR("Failed processing dataset " << dsu.p_dataset->get_id() << " " << dsu.p_dataset->get_dataset_name() << " " << ex.what());
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

struct diagnostic_interface_zwei::diagnostic_interface_impl final
{
    static const std::string their_pipe_name, mine_pipe_name;

    std::ifstream their_pipe;
    std::ofstream mine_pipe;

    diagnostic_interface_impl(std::ifstream &their_pipe, std::ofstream &mine_pipe)
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

    pimpl = ptr<diagnostic_interface_impl>(pipe, diagnostic_interface_impl::get_mine_stream());
}


void diagnostic_interface_zwei::wait()
{
    if (!getenv("BACKTEST")) return;

    if (pimpl) pimpl->wait();
}


}
