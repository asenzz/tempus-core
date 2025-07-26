#include <gtest/gtest.h>
#include <boost/program_options.hpp>
#include "TestSuite.hpp"

bool DoPerformanceTests = false;

bool TestSuite::doPerformanceTests()
{
    return DoPerformanceTests;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("perf,p", "Do performance tests")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("perf"))
        DoPerformanceTests = true;

    return RUN_ALL_TESTS();
}