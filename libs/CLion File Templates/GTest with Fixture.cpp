#[[#include]]# "test-config.hpp"
#[[#include]]# "test-appcontext.hpp"
#[[#include]]# "gtest/gtest.h"

using namespace svr::datamodel;
using namespace svr::dao;
using namespace svr::business;
using namespace svr::context;

using namespace std;

namespace {

class ${NAME} : public ::testing::Test {
protected:
    // Setup accessible fields here

    virtual void SetUp() override{
        //initialize accessible fields here
    }
    
    virtual void TearDown() override{
        // clean up code run after all tests 
    }
};

TEST_F(${NAME}, testSometing ) {

    // write test code here
    
    // e.g. ASSERT_TRUE(1 != 2);
    
}
    
} // anon namespace