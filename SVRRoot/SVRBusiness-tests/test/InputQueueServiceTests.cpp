#include "include/DaoTestFixture.h"
#include <iostream>

#include <model/User.hpp>
#include <model/InputQueue.hpp>

#include "include/InputQueueRowDataGenerator.hpp"

using namespace svr;

using svr::datamodel::User;
using svr::datamodel::InputQueue;
using svr::datamodel::DataRow;

TEST_F(DaoTestFixture, InputQueueWorkflow)
{
    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    ASSERT_FALSE(aci.input_queue_service.exists(user1->get_name(), user1->get_name(), bpt::seconds(60)));

    datamodel::InputQueue_ptr iq = std::make_shared<InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );

    aci.input_queue_service.save(iq);

    ASSERT_TRUE(aci.input_queue_service.exists(user1->get_name(), "logicalName", bpt::seconds(60)));

     bpt::ptime const st {bpt::time_from_string("2015-01-01 10:00:00") }
        , fn  { bpt::time_from_string("2015-01-01 10:10:00") };

    datamodel::DataRow_ptr row1 = std::make_shared<DataRow>(st, bpt::second_clock::local_time(), 1, 1); row1->set_values({0, 0, 0, 0});
    datamodel::DataRow_ptr row2 = std::make_shared<DataRow>(fn, bpt::second_clock::local_time(), 1, 1); row2->set_values({1, 1, 1, 1});
    aci.input_queue_service.add_row(iq, row1);
    aci.input_queue_service.add_row(iq, row2);

    aci.input_queue_service.save(iq);

    ASSERT_EQ(st, aci.input_queue_service.find_oldest_record(iq)->get_value_time() );
    ASSERT_EQ(fn, aci.input_queue_service.find_newest_record(iq)->get_value_time() );

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

namespace
{
    long const testDataNumberToGenerate = 5000;
}

TEST_F(DaoTestFixture, InputQueueSaveQueueTest) {

    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    ASSERT_FALSE(aci.input_queue_service.exists(user1->get_user_name(), user1->get_user_name(), bpt::seconds(60)));

    datamodel::InputQueue_ptr queue = std::make_shared<InputQueue>("", "simple_queue", user1->get_user_name(),
            "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );

    InputQueueRowDataGenerator dataGenerator(aci.input_queue_service, queue, queue->get_value_columns().size(), testDataNumberToGenerate);
/*
    PROFILE_EXEC_TIME(while(!dataGenerator.isDone()) {

        aci.input_queue_service.add_row(queue, dataGenerator());

    }, "Generating " << testDataNumberToGenerate << " InputQueue rows");
*/
    //PROFILE_EXEC_TIME(EXPECT_TRUE(testDataNumberToGenerate == (ssize_t) aci.input_queue_service.save(queue)), "Saving InputQueue");

    ASSERT_TRUE(aci.input_queue_service.exists(user1->get_user_name(), "simple_queue", bpt::seconds(60)));

    datamodel::InputQueue_ptr queueP2 = aci.input_queue_service.get_queue_metadata(user1->get_user_name(), "simple_queue", bpt::seconds(60));

    EXPECT_TRUE(0 == long(queueP2->size()));

    PROFILE_EXEC_TIME(queueP2->set_data(aci.input_queue_service.load(queueP2->get_table_name())), "Reading InputQueue data from database");

    EXPECT_TRUE(queue->size() == queueP2->size());

    //PROFILE_EXEC_TIME(EXPECT_TRUE(1 == aci.input_queue_service.remove(queue)), "Removing InputQueue table with " << testDataNumberToGenerate << " rows");

    ASSERT_FALSE(aci.input_queue_service.exists(user1->get_user_name(), "simple_queue", bpt::seconds(60)));

    aci.input_queue_service.remove(queue);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, GetColumnInFramesTest){

    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr queue = std::make_shared<InputQueue>("", "simple_queue", user1->get_user_name(),
            "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );


    // assume the queue does NOT exists before running this test
    ASSERT_FALSE(aci.input_queue_service.exists(user1->get_user_name(), "simple_queue", bpt::seconds(60)));

    InputQueueRowDataGenerator dataGenerator(aci.input_queue_service, queue, queue->get_value_columns().size(), testDataNumberToGenerate);

    // generate some random data
    while ( !dataGenerator.isDone() ) {
        datamodel::DataRow_ptr row = dataGenerator();
        aci.input_queue_service.add_row(queue, row);
    };

    // should save all the data
    EXPECT_TRUE(testDataNumberToGenerate == (ssize_t) aci.input_queue_service.save(queue));

    datamodel::DataRow_ptr oldestRecord, newestRecord;
    PROFILE_EXEC_TIME(oldestRecord = aci.input_queue_service.find_oldest_record(queue), "Getting oldest record");
    PROFILE_EXEC_TIME(newestRecord = aci.input_queue_service.find_newest_record(queue), "Getting newest record");

    LOG4_DEBUG("Oldest record: " << oldestRecord->to_string());
    LOG4_DEBUG("Newest record: " << newestRecord->to_string());

    bpt::ptime startTime = oldestRecord->get_value_time();
    bpt::ptime endTime = newestRecord->get_value_time();

    ASSERT_FALSE(startTime.is_special());
    ASSERT_FALSE(endTime.is_special());

    aci.input_queue_service.remove(queue);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, GetDBColumnsTests)
{
    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr queue = std::make_shared<InputQueue>("", "simple_queue", user1->get_user_name(),
            "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"eenie", "meenie", "miney", "mo", "catch", "a", "tiger", "by", "its", "toe"} );

    aci.input_queue_service.save(queue);

    auto columns = aci.input_queue_service.get_db_table_column_names(queue);

    DTYPE(columns) must_be {"eenie", "meenie", "miney", "mo", "catch", "a", "tiger", "by", "its", "toe"};

    ASSERT_EQ(must_be.size(), columns.size());

    auto ic = columns.begin();

    for(auto mb : must_be)
        ASSERT_EQ(mb, *ic++);


    aci.input_queue_service.remove(queue);
    aci.user_service.remove(user1);
}


TEST_F(DaoTestFixture, TestFixInputQueueSelection)
{
    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr queue_f = std::make_shared<InputQueue>("", "simple_queue_f", user1->get_user_name(),
            "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"eenie", "meenie", "miney", "mo", "catch", "a", "tiger", "by", "its", "toe"} );

    aci.input_queue_service.save(queue_f);

    datamodel::InputQueue_ptr queue_t = std::make_shared<InputQueue>("", "simple_queue_t", user1->get_user_name(),
            "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"eenie", "meenie", "miney", "mo", "catch", "a", "tiger", "by", "its", "toe"}, true );

    aci.input_queue_service.save(queue_t);

    APP.flush_dao_buffers();

    auto queues = aci.input_queue_service.get_all_queues_with_sign(true);

    ASSERT_EQ(queues.size(), 1UL);

    ASSERT_EQ(queues[0]->get_table_name() , queue_t->get_table_name());
    ASSERT_EQ(queues[0]->get_logical_name() , queue_t->get_logical_name());
    ASSERT_EQ(queues[0]->get_resolution() , queue_t->get_resolution());
    ASSERT_EQ(queues[0]->get_owner_user_name(), queue_t->get_owner_user_name());
    ASSERT_EQ(queues[0]->get_value_columns(), queue_t->get_value_columns());


    aci.input_queue_service.remove(queue_t);
    aci.input_queue_service.remove(queue_f);
    aci.user_service.remove(user1);
}

namespace {

bpt::ptime operator "" _pt(const char * s, size_t)
{
    return bpt::time_from_string(s);
}

}

TEST_F(DaoTestFixture, InputQueueReconciliationWorkflow)
{
    User_ptr user1 = std::make_shared<User>(
            bigint(), "InputQueueTestUser", "InputQueueTestUser@email", "InputQueueTestUser", "InputQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    bpt::time_duration resolution = bpt::seconds(60);

    aci.user_service.save(user1);

    ASSERT_FALSE(aci.input_queue_service.exists(user1->get_name(), user1->get_name(), resolution));

    datamodel::InputQueue_ptr iq = std::make_shared<InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", resolution, bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );

    aci.input_queue_service.save(iq);

    bpt::ptime st1 {"2015-01-01 10:00:00"_pt }
             , st2 {"2015-01-01 10:30:00"_pt };

    for(size_t i = 0; i < 10; ++i)
    {
        datamodel::DataRow_ptr row1 = std::make_shared<DataRow>(st1 + resolution * i, bpt::second_clock::local_time(), 1, 1);
        row1->set_values({0.0 + i, 0.0 + i, 0.0 + i, 0.0 + i});
        datamodel::DataRow_ptr row2 = std::make_shared<DataRow>(st2 + resolution * i, bpt::second_clock::local_time(), 1, 1);
        row2->set_values({0.0 + i, 0.0 + i, 0.0 + i, 0.0 + i});
        aci.input_queue_service.add_row(iq, row1);
        aci.input_queue_service.add_row(iq, row2);
    }

    aci.input_queue_service.save(iq);

    aci.flush_dao_buffers();

    svr::dao::OptionalTimeRange missing
            = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 10:30:00"_pt, "2015-01-01 10:39:00"_pt));
    ASSERT_TRUE(missing.is_initialized()); ASSERT_EQ("2015-01-01 10:30:00"_pt, missing->first); ASSERT_EQ("2015-01-01 10:30:00"_pt, missing->second);


    missing = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 10:00:00"_pt, "2015-01-01 10:39:00"_pt));
    ASSERT_TRUE(missing.is_initialized()); ASSERT_EQ("2015-01-01 10:10:00"_pt, missing->first); ASSERT_EQ("2015-01-01 10:30:00"_pt, missing->second);


    missing = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 09:50:00"_pt, "2015-01-01 10:39:00"_pt));
    ASSERT_TRUE(missing.is_initialized()); ASSERT_EQ("2015-01-01 09:50:00"_pt, missing->first); ASSERT_EQ("2015-01-01 10:00:00"_pt, missing->second);


    missing = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 09:40:00"_pt, "2015-01-01 10:50:00"_pt));
    ASSERT_TRUE(missing.is_initialized()); ASSERT_EQ("2015-01-01 10:40:00"_pt, missing->first); ASSERT_EQ("2015-01-01 10:51:00"_pt, missing->second);


    missing = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 09:40:00"_pt, "2015-01-01 10:50:00"_pt));
    ASSERT_TRUE(missing.is_initialized()); ASSERT_EQ("2015-01-01 09:40:00"_pt, missing->first); ASSERT_EQ("2015-01-01 09:50:00"_pt, missing->second);

    missing = aci.input_queue_service.get_missing_hours(iq, svr::dao::TimeRange("2015-01-01 09:40:00"_pt, "2015-01-01 10:50:00"_pt));
    ASSERT_FALSE(missing.is_initialized());


    aci.input_queue_service.remove(iq);

    aci.user_service.remove(user1);
}
