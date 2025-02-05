#include <cppcms/service.h>
#include <cppcms/mount_point.h>
#include <controller/MainController.hpp>
#include <controller/ConnectorController.hpp>
#include <boost/program_options.hpp>

using namespace svr::web;

bool parse(int argc, char** argv)
{
    boost::program_options::options_description genDesc = boost::program_options::options_description("Web service options");
    genDesc.add_options()
            ("help",
             "produce help message")
            ("app-config,a",
             boost::program_options::value<std::string>()->default_value("daemon.config"),
             "Path to file with SQL configuration for daemon.")
            ("web-config,c",
             boost::program_options::value<std::string>()->default_value("config.json"),
             "Web specific settings configuration file in JSON format.");

    boost::program_options::variables_map vm;

    // parse command line
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(genDesc).run(), vm);

    if (vm.count("help") )
    {
        std::cout << genDesc << "\n";
        return false;
    }

    if (vm["web-config"].as<std::string>().empty())
    {
        throw std::invalid_argument("empty path to config file");
    }

    if (vm["app-config"].as<std::string>().empty())
    {
        throw std::invalid_argument("empty path to config file");
    }

    svr::context::AppContext::init_instance(vm["app-config"].as<std::string>().c_str());
    return true;
}

namespace{
    static svr::context::AppContextDeleter appContextDeleter;
}

int main (int argc, char** argv)
{
    try{
        if (!parse(argc, argv))
            return 0;
        cppcms::service svc(argc, argv);
        svc.applications_pool().mount(applications_factory<ConnectorController>(), cppcms::mount_point("/tempus(/(.*))?"));
        svc.applications_pool().mount(applications_factory<MainController>()), cppcms::mount_point("/web(/(.*))?");
        LOG4_DEBUG("Web application has started at " << svc.settings().at("service.api").str() << "://" << svc.settings().at("service.ip").str() << ":" << svc.settings().at("service.port").number());
        svc.run();
    } catch (const std::exception &e) {
        LOG4_ERROR(e.what());
        std::stringstream ss_args;
        for (int i = 0; i < argc; ++i) ss_args << "'" << argv[i] << "' ";
        LOG4_ERROR("Called with arguments: " << ss_args.str().c_str());
    }
    LOG4_DEBUG("Web application is shutting down.");
    return 0;
}
