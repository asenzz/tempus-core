#include "test-config.hpp"
#include "appcontext.hpp"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
	bpt::ptime startTime = bpt::microsec_clock::local_time();
    LOG4_INFO("Testing started at " << bpt::to_simple_string(startTime));

    ::testing::InitGoogleTest(&argc, argv);
    int testsRet = RUN_ALL_TESTS();

    LOG4_INFO("Tests finished in " << (bpt::microsec_clock::local_time() - startTime) << " seconds.");

    return testsRet;
}