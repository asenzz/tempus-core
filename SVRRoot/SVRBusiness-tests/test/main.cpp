#include <gtest/gtest.h>
#include "include/dbutils.h"
#include "include/DaoTestFixture.h"
#include <appcontext.hpp>

#include <boost/program_options.hpp>

namespace{
    static svr::context::AppContextDeleter appContextDeleter;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("dao_type,d", po::value<std::string>(), "Specify dao type (async, postgres)")
            ("perf,p", "Do performance tests")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::string daoType = "postgres";

    if (vm.count("dao_type"))
        daoType = vm["dao_type"].as<std::string>();

    if (vm.count("perf"))
        DaoTestFixture::DoPerformanceTests = true;

    TestEnv tdb;

    char const * dbname = tdb.TestDbUserName;

    if( !tdb.prepareSvrConfig(dbname, daoType))
        return 1;

    std::string hostname = exec("hostname"); erase_after(hostname, '\n');

    if(!tdb.init_test_db(dbname))
        return 1;

    svr::context::AppContext::init_instance(tdb.AppConfigFile, true);

    return RUN_ALL_TESTS();
}


