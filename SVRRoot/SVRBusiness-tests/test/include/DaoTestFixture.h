#ifndef DAOTESTFIXTURE_H
#define DAOTESTFIXTURE_H

#include <gtest/gtest.h>

#include "dbutils.h"
#include <appcontext.hpp>

class DaoTestFixture : public ::testing::Test {
public:
    svr::context::AppContext &aci;
    DaoTestFixture(): aci(APP) {}
    ~DaoTestFixture()
    {
        if (HasFatalFailure())
        {
            aci.destroy_instance();

            TestEnv tdb;

            char const * dbname = tdb.TestDbUserName;
            tdb.init_test_db(dbname);

            svr::context::AppContext::init_instance(TestEnv::AppConfigFile);
        }
    }
    static bool DoPerformanceTests;
};



#endif /* DAOTESTFIXTURE_H */

