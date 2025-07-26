#include "appcontext.hpp"
#include "DaemonFacade.hpp"
#include <boost/program_options.hpp>
#include <model/Request.hpp>
#include <csignal>

using namespace svr;

void signal_handler(const int signum)
{
    LOG4_DEBUG("Interrupt signal " << signum << " received.");
    if (signum == SIGINT || signum == SIGABRT || signum == SIGTERM) {
        LOG4_INFO("Daemon process is shutting down gracefully.");
        exit(signum);
    }
    LOG4_ERROR("Received unexpected signal: " << signum);
}

std::string parse(const int argc, const char **argv)
{
    auto gen_desc = boost::program_options::options_description("Daemon options");
    gen_desc.add_options()
            ("help", "produce help message")
            ("config,c", boost::program_options::value<std::string>()->default_value("daemon.config"), "Path to file with SQL configuration for daemon");

    boost::program_options::variables_map vm;
    // parse command line
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(gen_desc).run(), vm);
    if (vm.count("help") or !vm.count("config")) {
        std::cout << gen_desc << "\n";
        exit(0);
    }
    if (vm["config"].as<std::string>().empty())
        THROW_EX_FS(std::invalid_argument, "Empty path to config file.");

    context::AppContext::init_instance(vm["config"].as<std::string>().c_str());
    return vm["config"].as<std::string>();
}

int main(const int argc, const char **argv)
{
/*
#ifndef NDEBUG
    mtrace();
#endif
*/

#ifdef INTEGRATION_TEST
    LOG4_FATAL("Integration test build cannot be run in production.");
    exit(0xfd);
#else
    (void) signal(SIGINT, signal_handler);
    (void) signal(SIGABRT, signal_handler);
    (void) signal(SIGTERM, signal_handler);

    omp_set_nested(true);
    if (common::gpu_handler_1::get().get_gpu_devices_count()) omp_set_default_device(0);

    std::shared_ptr<daemon::DaemonFacade> p_daemon_facade;
    int rc = 0;
    try {
        const std::string config_path = parse(argc, argv);
        p_daemon_facade = ptr<daemon::DaemonFacade>();
        p_daemon_facade->start_loop();
    } catch (const std::invalid_argument &e) {
        LOG4_ERROR(e.what());
        rc = 1;
    } catch (const std::exception &e) {
        LOG4_ERROR(e.what());
        rc = 0xff;
    } catch (...) {
        LOG4_ERROR("Unknown exception thrown. ");
        rc = 0xfe;
    }
    LOG4_INFO("Daemon process finishing");

    return rc;
#endif
}
