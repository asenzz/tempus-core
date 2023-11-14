#include <vector>

#include <gtest/gtest.h>

#include <common/thread_pool.hpp>

#include <common/Logging.hpp>

#include "TestSuite.hpp"


//  Implementation  Test size   Run 1   Run 2   Run 3   
//  std::cout       100000      1668ms  1665    1665
//  boost::log      100000      737     750     714
//  spdlog          100000      685     629     631
//

size_t const test_size = 100000;

TEST(LoggingTests, CurrentPerfTests)
{
    if(!TestSuite::doPerformanceTests())
        return;
    
    std::vector<svr::future<void>> futures; futures.reserve(test_size);
    
    for(size_t i = 0; i < test_size; ++i)
    {
        futures.emplace_back(
            svr::async([i]()
                {
                    LOG4_INFO ("Current logger: " << i << " doubles " << double(i));
                }));
    }
    
    for(auto & future : futures)
        future.get();
}

#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/attributes/function.hpp>

TEST(LoggingTests, BoostPerfTests)
{
    if(!TestSuite::doPerformanceTests())
        return;
    
    boost::log::add_common_attributes();
    
//    auto core = boost::log::core::get();
//    core->add_global_attribute("function", boost::log::attributes::function());
    

    std::vector<svr::future<void>> futures; futures.reserve(test_size);
    
    for(size_t i = 0; i < test_size; ++i)
    {
        futures.emplace_back(
            svr::async([i]()
                {
                    BOOST_LOG_TRIVIAL(info) << "Boost logger: " << i << " doubles " << double(i);
                }));
    }
    
    for(auto & future : futures)
        future.get();
}

