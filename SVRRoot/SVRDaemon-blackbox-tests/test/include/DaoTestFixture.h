#ifndef DAOTESTFIXTURE_H
#define DAOTESTFIXTURE_H

#include <gtest/gtest.h>

#include "testutils.h"
#include <appcontext.hpp>

class DaoTestFixture : public ::testing::Test {
public:
    svr::context::AppContext & aci;
    TestEnv & tdb;
    DaoTestFixture(): aci(APP), tdb(TestEnv::global_test_env()) {}
    ~DaoTestFixture()
    {
        if (HasFatalFailure())
        {
            aci.destroy_instance();

            char const *dbname = tdb.TestDbUserName;
            tdb.init_test_db(dbname);

            svr::context::AppContext::init_instance(TestEnv::AppConfigFile);
        }
    }
    static bool DoPerformanceTests;

    enum class test_completeness {full = 0, first_half, second_half};
    static test_completeness DoTestPart;
};

#endif /* DAOTESTFIXTURE_H */
