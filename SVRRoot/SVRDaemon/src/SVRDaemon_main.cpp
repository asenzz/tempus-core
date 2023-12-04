#include "appcontext.hpp"
#include "DaemonFacade.hpp"
#include <boost/program_options.hpp>
#include <model/Request.hpp>
#include <csignal>


using namespace svr::daemon;
using namespace svr::context;
using namespace svr::datamodel;


void signal_handler(int signum)
{
    LOG4_DEBUG("Interrupt signal (" << signum << ") received.");

    exit(signum);
}


std::string parse(int argc, char** argv)
{
    boost::program_options::options_description gen_desc = boost::program_options::options_description("Daemon options");
    gen_desc.add_options()
        ("help",     "produce help message")
        ("config,c",   boost::program_options::value<std::string>()->default_value("daemon.config"),
                    "Path to file with SQL configuration for daemon");

    boost::program_options::variables_map vm;
    // parse command line
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(gen_desc).run(), vm);
    if (vm.count("help") or !vm.count("config"))
    {
        std::cout << gen_desc << "\n";
        exit(0);
    }
    if (vm["config"].as<std::string>().empty())
        THROW_EX_F(std::invalid_argument, "Empty path to config file.");

    AppContext::init_instance(vm["config"].as<std::string>().c_str());
    return vm["config"].as<std::string>();
}


namespace {
    static AppContextDeleter appContextDeleter;
}


int main(int argc, char** argv)
{
//    mtrace();
    signal(SIGINT, signal_handler);
    signal(SIGABRT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::shared_ptr<DaemonFacade> p_daemon_facade;
    try
    {
        std::string config_path = parse(argc, argv);
        p_daemon_facade = std::make_shared<DaemonFacade>(config_path);
        p_daemon_facade->start_loop();
    } catch(std::invalid_argument& e) {
        LOG4_ERROR(e.what());
        return 1;
    }
/*
    catch(std::exception& e)
    {
        LOG4_ERROR(e.what());
        return 1;
    }
    catch(...)
    {
        LOG4_ERROR("Unknown exception thrown. ");
        return 1;
    }
*/
    LOG4_INFO("Daemon process finishing");

    return 0;
}
