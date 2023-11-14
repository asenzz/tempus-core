#include "include/DaoTestFixture.h"
#include <model/User.hpp>
#include <model/InputQueue.hpp>
#include <model/DeconQueue.hpp>
#include <model/Dataset.hpp>

TEST_F(DaoTestFixture, DQSavePerfTests)
{
    using namespace std::chrono;

    if(!DaoTestFixture::DoPerformanceTests)
        return;

    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "JamesBond", "JamesBond@email", "JamesBond", "JamesBond", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "SomeInputQueue", "SomeInputQueue", user1->get_name(), "SomeInputQueue", bpt::seconds(60), bpt::seconds(5), "UTC", std::vector<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "SomeTestDataset", user1->get_user_name(), iq, std::vector<InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 4, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    size_t const m = 200;
    size_t const n = 5000;

    duration<double> all_tests_time;

    for(size_t j = 0UL; j < m; ++j)
    {
        DeconQueue_ptr dq = std::make_shared<svr::datamodel::DeconQueue>("SomeDeconQueuetableName", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());

        bpt::ptime nw = bpt::second_clock::local_time();

        auto tm = nw;

        for(size_t i = 0UL; i < n; ++i)
        {
            tm = nw + bpt::minutes(i);

            DataRow_ptr row = std::make_shared<svr::datamodel::DataRow>(tm);
            row->set_values({0, 1, 2});

            dq->get_data().push_back(row);
        }

        aci.decon_queue_service.save(dq);
        high_resolution_clock::time_point start = high_resolution_clock::now();
        aci.decon_queue_service.save(dq);
        high_resolution_clock::time_point finish = high_resolution_clock::now();
        all_tests_time += duration_cast<duration<double>>(finish - start);

        aci.decon_queue_service.remove(dq);
    }

    std::cout << "DeconQueueSavePerformanceTests: inserting " << n << " rows averaged over " << m << " tests took " << all_tests_time.count() / m << "sec." << std::endl;

//******************************************************************************
// The test is inserting 5000 rows averaged over 200 tests.
// The results are as follows:
// Batch insert     - empty table: 0.108732sec.
// Stored procedure - empty table: 0.152076sec
// Stored procedure - same row set: 0.155222sec
//******************************************************************************

    aci.dataset_service.remove(ds);

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
