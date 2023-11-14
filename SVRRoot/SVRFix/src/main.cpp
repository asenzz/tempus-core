#include "FixConnector.hpp"
#include "appcontext.hpp"

#include <quickfix/FileStore.h>
#include <quickfix/SocketInitiator.h>
#include <quickfix/SessionSettings.h>
#include <quickfix/Log.h>
#include <atomic>
#include <boost/program_options.hpp>

extern "C"
{
void signal_callback_handler(int signum);
}

struct raii_helper
{
    FIX::SessionSettings settings;
    svr::fix::FixConnector connector;
    FIX::FileStoreFactory storeFactory;
    FIX::ScreenLogFactory logFactory;
    FIX::SocketInitiator * initiator;

    raii_helper(std::string const & config_file, bool raw_fix_data)
        : settings(config_file)
        , connector(settings)
        , storeFactory( settings )
        , logFactory( settings )
    {
        if(raw_fix_data)
            initiator = new FIX::SocketInitiator( connector, storeFactory, settings, logFactory );
        else
            initiator = new FIX::SocketInitiator( connector, storeFactory, settings );

        raii_helper::instance.store(this);
        signal(SIGINT, signal_callback_handler);
    }

    ~raii_helper()
    {
        stop();
        delete initiator;
    }

    void stop()
    {
        raii_helper::instance.store(nullptr);
        connector.stop();
    }

    void run()
    {
        initiator->start();
        connector.run();
        initiator->stop();
    }

    static std::atomic<raii_helper*> instance;
};

std::atomic<raii_helper*> raii_helper::instance(nullptr) ;

void signal_callback_handler(int signum)
{
    raii_helper* inst = raii_helper::instance.load();
    if(inst)
        inst->stop();
}

int main( int argc, char** argv )
{

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help")
            ("fix_config,f", po::value<std::string>()->required(), "Quickfix config file path")
            ("app_config,a", po::value<std::string>()->required(), "SVR application config file path")
            ("raw,r", "output raw FiX data")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        desc.print(std::cout);
        return EXIT_SUCCESS;
    }

    try
    {
        po::notify(vm);
    }
    catch(boost::program_options::required_option &)
    {
        desc.print(std::cout);
        return EXIT_FAILURE;
    }

    std::string const qfix_config = vm["fix_config"].as<std::string>()
                    , appl_config = vm["app_config"].as<std::string>();

    svr::context::AppContext::init_instance(appl_config);
    svr::context::AppContextDeleter deleter;

    bool raw_fix_data = false;

    if( vm.count("raw") )
        raw_fix_data = true;

    try
    {
        raii_helper helper( qfix_config, raw_fix_data );
        helper.run();
        return EXIT_SUCCESS;
    }
    catch ( std::exception & e )
    {
        std::cout << e.what();
        return EXIT_FAILURE;
    }


}