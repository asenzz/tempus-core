#include "include/DaoTestFixture.h"
#include <model/User.hpp>
#include <model/InputQueue.hpp>

using namespace svr;

TEST_F(DaoTestFixture, IQSavePerfTests)
{
    using namespace std::chrono;

    if(!DaoTestFixture::DoPerformanceTests)
        return;

    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "JamesBond", "JamesBond@email", "JamesBond", "JamesBond", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    size_t const m = 200;
    size_t const n = 5001;

    duration<double> all_tests_time{0};

    for(size_t j = 0UL; j < m; ++j)
    {
        datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "SomeInputQueue", "SomeInputQueue", user1->get_name(), "SomeInputQueue", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "l_ft"} );

        bpt::ptime nw = bpt::second_clock::local_time();

        auto tm = nw;

        for(size_t i = 0UL; i < n; ++i)
        {
            tm = nw + bpt::minutes(i);

            svr::datamodel::DataRow_ptr row = std::make_shared<svr::datamodel::DataRow>(tm, bpt::second_clock::local_time(), 1, 1);
            row->set_values({0, 1, 2});

            iq->get_data().push_back(row);
        }

        aci.input_queue_service.save(iq);
        high_resolution_clock::time_point start = high_resolution_clock::now();
        aci.input_queue_service.save(iq);
        high_resolution_clock::time_point finish = high_resolution_clock::now();
        all_tests_time += duration_cast<duration<double>>(finish - start);

        aci.input_queue_service.remove(iq);
    }

    std::cout << "DeconQueueSavePerformanceTests: inserting " << n << " rows averaged over " << m << " tests took " << all_tests_time.count() / m << "sec." << std::endl;

//******************************************************************************
// The test is inserting 5001 rows averaged over 200 tests.
// The results are as follows:
// Batch insert     - empty table: 0.103008sec.
// Update/insert    - same row set: 1.94903sec
// Stored procedure - same row set: 0.162161sec
// Stored procedure - SVRWeb SendHistoryData rate increased from 1700 bars per second up to 5374 bars per sec.
//******************************************************************************

    aci.user_service.remove(user1);
}
