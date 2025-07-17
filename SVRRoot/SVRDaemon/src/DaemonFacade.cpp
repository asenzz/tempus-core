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

namespace svr::daemon {

DaemonFacade::DaemonFacade() :
    loop_interval(PROPS.get_loop_interval()),
    stream_loop_interval(PROPS.get_stream_loop_interval()),
    loop_count(0),
    max_loop_count(PROPS.get_max_loop_count()),
    self_request(PROPS.get_self_request())

{
    PROFILE_INFO(APP.dataset_service.update_active_datasets(datasets), "Update active datasets");
    running = true;
    shm_io_thread = std::thread(&DaemonFacade::shm_io_callback, this);
    save_queues_thread = std::thread(&DaemonFacade::save_queues_callback, this);
}

DaemonFacade::~DaemonFacade()
{
    running = false;
    shm_io_thread.join();
    save_queues_thread.join();
}

void DaemonFacade::do_fork()
{
    pid_t pid;

    /* Catch, ignore and handle signals */
    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    /* Fork off the parent process */
    pid = fork();
    if (pid < 0) {
        LOG4_ERROR("Failed to fork the parent process");
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


void DaemonFacade::save_queues_callback()
{
    while (running) {
        std::this_thread::sleep_for(loop_interval);
        if (modified_queues.empty()) continue;

        const tbb::mutex::scoped_lock lk(save_queues_mx);
        if (modified_queues.empty()) continue;

        std::for_each(C_default_exec_policy, modified_queues.cbegin(), modified_queues.cend(), [](const auto &p_queue) {
                          const tbb::mutex::scoped_lock l2(p_queue->get_update_mutex());
                          PROFILE_INFO(APP.input_queue_service.save(p_queue), "Save input queue " << p_queue->get_table_name())
                      }
        );
        modified_queues.clear();
    }
}

void DaemonFacade::process_streams()
{
    const auto timenau = bpt::second_clock::local_time();
    const auto tables = streaming_messages_protocol::get().receive_queues_data(timenau);
    const tbb::mutex::scoped_lock lk(update_datasets_mx);
    // TODO Parallelize
    for (auto &du: datasets) {
        const auto p_dataset = du.p_dataset;
        const auto p_queue = p_dataset->get_input_queue();
        for (const auto &t: tables) {
            if (p_queue->get_table_name() != t.first || t.second.column_names != p_queue->get_value_columns()) continue;

            if (t.second.rows.size()) {
                {
                    const tbb::mutex::scoped_lock l2(p_queue->get_update_mutex());
                    if (p_queue->back()->get_value_time() >= t.second.rows.front()->get_value_time())
                        p_queue->get_data().erase(lower_bound(p_queue->get_data(), t.second.rows.front()->get_value_time()), p_queue->end());
                    p_queue->get_data().insert(p_queue->end(), std::make_move_iterator(t.second.rows.begin()), std::make_move_iterator(t.second.rows.end()));
                }
                {
                    const tbb::mutex::scoped_lock l2(save_queues_mx);
                    modified_queues.emplace(p_queue);
                }
            }

            const auto resolution = p_queue->get_resolution();
            const auto last_aux_time = p_dataset->get_aux_input_queue()->back()->get_value_time();
            const auto last_input_res_2 = p_queue->back()->get_value_time() + (resolution * 2.);
            if (self_request && p_dataset->get_aux_input_queue()->size() &&
                last_aux_time - p_dataset->get_last_self_request() >= resolution &&
                last_aux_time >= last_input_res_2 - (resolution * PROPS.get_prediction_horizon())) {
                const std::deque self_requests{otr<datamodel::MultivalRequest>(
                        0, "", p_dataset->get_id(), timenau, last_input_res_2, last_input_res_2 + resolution, resolution, p_dataset->get_input_queue()->get_value_columns())};
                business::t_stream_results stream_results;
                for (const auto &p_user: du.users)
                    business::DatasetService::process_requests(*p_user, *p_dataset, self_requests, &stream_results);
                p_dataset->set_last_self_request(last_aux_time);
                streaming_messages_protocol::get().write_response(p_dataset->get_id(), stream_results);
            }
        }
    }
}

void DaemonFacade::shm_io_callback()
{
    while (running) {
        std::this_thread::sleep_for(stream_loop_interval);
        process_streams();
    }
}


void DaemonFacade::start_loop()
{
    // configure the daemon
    if (PROPS.get_daemonize()) {
        do_fork();
        LOG4_INFO("Daemon process started");
    }

    syslog(LOG_NOTICE, "Daemon loop started");

#ifndef NDEBUG
    diagnostic_interface_zwei di;
#endif

    // Run normal daemon cycle
    while (continue_loop()) {
        if (++ctr == 0) {
            const tbb::mutex::scoped_lock lk(update_datasets_mx);
            PROFILE_INFO(APP.dataset_service.update_active_datasets(datasets), "Update active datasets");
        }
        for (business::DatasetService::DatasetUsers &dsu: datasets) {

#ifndef NDEBUG
            di.wait();
#endif

            auto &dataset = *dsu.p_dataset;
            try {
                boost::unordered_flat_map<bigint, std::deque<datamodel::MultivalRequest_ptr>> user_requests;
                for (const auto &p_user: dsu.users)
                    user_requests[p_user->get_id()] = context::AppContext::get_instance().request_service.get_active_multival_requests(*p_user, dataset);
                PROFILE_INFO(business::DatasetService::process(*dsu.p_dataset), "Process dataset");
                for (const auto &p_user: dsu.users) PROFILE_INFO(business::DatasetService::process_requests(*p_user, *dsu.p_dataset, user_requests[p_user->get_id()], nullptr),
                                                                      "Process multival requests " << p_user->get_user_name());
            } catch (const common::bad_model &ex) {
                LOG4_ERROR("Bad model while processing dataset " << dsu.p_dataset->get_id() << " " << dsu.p_dataset->get_dataset_name()
                                                                 << " level " << ex.get_decon_level() << ", column " << ex.get_column_name() << " error " << ex.what());
            } catch (const std::exception &ex) {
                LOG4_ERROR("Failed processing dataset " << dsu.p_dataset->get_id() << " " << dsu.p_dataset->get_dataset_name() << " " << ex.what());
            }
        }
    }
    running = false;
    shm_io_thread.join();
    save_queues_thread.join();
}


bool DaemonFacade::continue_loop()
{
    // wait some time before next iteration
    std::this_thread::sleep_for(loop_interval);
    if (max_loop_count < 0) return true;
    else return (++loop_count) < max_loop_count;
}


/*
 * SVRDaemon-whitebox-test related logic follows below.
 */

#ifndef NDEBUG

struct diagnostic_interface_zwei::diagnostic_interface_impl final {
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

#endif // NDEBUG

}
