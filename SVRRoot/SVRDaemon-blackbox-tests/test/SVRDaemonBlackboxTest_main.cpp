#include <gtest/gtest.h>
#include "include/testutils.h" // "" here is not en error
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
            ("daoType,d", po::value<std::string>(), "Specify dao type (async, postgres)")
            ("perf,p", "Do performance tests")
            ("h1,1", "Do the 1st half of the test - prepare requests, not run the Daemon")
            ("h2,1", "Do the 2nd half of the test - check results, not run the Daemon");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    std::string daoType = "postgres";

    if (vm.count("daoType")) daoType = vm["daoType"].as<std::string>();
    if (vm.count("perf")) DaoTestFixture::DoPerformanceTests = true;
    
    DaoTestFixture::DoTestPart = DaoTestFixture::test_completeness::full;
    
    if (vm.count("h1"))
        DaoTestFixture::DoTestPart = DaoTestFixture::test_completeness::first_half;

    if (vm.count("h2"))
        DaoTestFixture::DoTestPart = DaoTestFixture::test_completeness::second_half;

    TestEnv& tdb = TestEnv::global_test_env();
    tdb.dao_type = daoType;
    
    char const * dbname = tdb.TestDbUserName;

    if( !tdb.prepareSvrConfig(dbname, daoType, 1)) return 1;
    std::string hostname = exec("hostname");
    erase_after(hostname, '\n');
    if(!tdb.init_test_db_98(dbname)) return 1;

    svr::context::AppContext::init_instance(tdb.AppConfigFile);
    return RUN_ALL_TESTS();
}


