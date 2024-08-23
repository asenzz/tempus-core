#include "include/DaoTestFixture.h"
#include "common/constants.hpp"

#include <memory>

#include <model/User.hpp>
#include <model/Request.hpp>
#include <model/Dataset.hpp>
#include <model/InputQueue.hpp>

using namespace svr;
using datamodel::MultivalRequest;

TEST_F(DaoTestFixture, RequestWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "logicalNaDeconme", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{},
                                                                          svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_len, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    bpt::ptime nw = bpt::second_clock::local_time();

    datamodel::MultivalRequest_ptr request = std::make_shared<MultivalRequest>(
        MultivalRequest(bigint(0), user1->get_user_name(), ds->get_id(), nw
            , nw + iq->get_resolution(), nw + iq->get_resolution()*2, iq->get_resolution().total_seconds()
            , "{open,close,high,low}")
    );

    ASSERT_EQ(1, aci.request_service.save(request));

    datamodel::MultivalRequest_ptr rq = aci.request_service.get_multival_request
            ( user1->get_user_name(), ds->get_id(), nw + iq->get_resolution(), nw + iq->get_resolution()*2
            , iq->get_resolution().total_seconds()
            , "{open,close,high,low}"
            );

    ASSERT_EQ(*request, *rq);


    datamodel::MultivalResponse_ptr rs0 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, rq->get_id(), nw + iq->get_resolution(), "high", 1.1));
    aci.request_service.save(rs0);

    datamodel::MultivalResponse_ptr rs1 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, rq->get_id(), nw + iq->get_resolution(), "close", 2.2));
    aci.request_service.save(rs1);

    datamodel::MultivalResponse_ptr rs2 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, rq->get_id(), nw + iq->get_resolution(), "open", 3.3));
    aci.request_service.save(rs2);

    datamodel::MultivalResponse_ptr rs3 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, rq->get_id(), nw + iq->get_resolution(), "low", 4.4));
    aci.request_service.save(rs3);

    aci.request_service.force_finalize(request);

    aci.request_service.prune_finalized_requests(bpt::second_clock::local_time() + onesec);

    aci.dataset_service.remove(ds);

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, CustomColumnsRequest)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "logicalNaDeconme", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}
            , svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_len, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    bpt::ptime nw = bpt::second_clock::local_time();

    datamodel::MultivalRequest_ptr request = std::make_shared<MultivalRequest>(
        MultivalRequest(bigint(0), user1->get_user_name(), ds->get_id(), nw
            , nw + iq->get_resolution(), nw + iq->get_resolution()*2, iq->get_resolution().total_seconds()
            , "{open, some_custom_column}")
    );

    ASSERT_EQ(1, aci.request_service.save(request));

    datamodel::MultivalResponse_ptr rs0 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, request->get_id(), nw + iq->get_resolution(), "open", 1.1));
    aci.request_service.save(rs0);

    datamodel::MultivalResponse_ptr rs1 = datamodel::MultivalResponse_ptr(new svr::datamodel::MultivalResponse(0, request->get_id(), nw + iq->get_resolution(), "some_custom_column", 2.2));
    aci.request_service.save(rs1);

    aci.request_service.force_finalize(request);

    aci.request_service.prune_finalized_requests(bpt::second_clock::local_time() + onesec);

    datamodel::MultivalRequest_ptr rq = aci.request_service.get_multival_request
            ( user1->get_user_name(), ds->get_id(), nw + iq->get_resolution(), nw + iq->get_resolution()*2
            , iq->get_resolution().total_seconds()
            , "{open, some_custom_column}"
            );

    ASSERT_EQ(nullptr, rq);

    aci.dataset_service.remove(ds);

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}


TEST_F(DaoTestFixture, GettingActiveRequests)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "FridayTestUser", "Friday@email", "FridayTestUser", "FridayTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "InputQueueName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "FridayTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{},
                                                                          svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_len, PROPS.get_multistep_len(), 4, "sym7");

    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    bpt::ptime nw = bpt::second_clock::local_time();

    datamodel::MultivalRequest_ptr request1 = std::make_shared<MultivalRequest>(
        MultivalRequest(bigint(0), user1->get_user_name(), ds->get_id(), nw
            , nw + iq->get_resolution(), nw + iq->get_resolution()*2, iq->get_resolution().total_seconds()
            , "{open}")
    );

    datamodel::MultivalRequest_ptr request2 = std::make_shared<MultivalRequest>(
        MultivalRequest(bigint(0), user1->get_user_name(), ds->get_id(), nw
            , nw + iq->get_resolution()*2, nw + iq->get_resolution()*3, iq->get_resolution().total_seconds()
            , "{open}")
    );

    datamodel::MultivalRequest_ptr request3 = std::make_shared<MultivalRequest>(
        MultivalRequest(bigint(0), user1->get_user_name(), ds->get_id(), nw
            , nw + iq->get_resolution()*3, nw + iq->get_resolution()*4, iq->get_resolution().total_seconds()
            , "{open}")
    );


    aci.request_service.save(request1);

    auto active_requests = aci.request_service.get_active_multival_requests(*user1, *ds);
    ASSERT_EQ(1UL, active_requests.size()); ASSERT_EQ(*request1, *active_requests[0]);

    aci.request_service.save(request2);

    active_requests = aci.request_service.get_active_multival_requests(*user1, *ds);
    ASSERT_EQ(2UL, active_requests.size()); ASSERT_EQ(*request2, *active_requests[1]);

    aci.request_service.save(request3);

    active_requests = aci.request_service.get_active_multival_requests(*user1, *ds);
    ASSERT_EQ(3UL, active_requests.size()); ASSERT_EQ(*request3, *active_requests[2]);

    aci.request_service.force_finalize(request1);

    active_requests = aci.request_service.get_active_multival_requests(*user1, *ds);
    ASSERT_EQ(2UL, active_requests.size()); ASSERT_EQ(*request2, *active_requests[0]); ASSERT_EQ(*request3, *active_requests[1]);

    aci.request_service.save(std::make_shared<svr::datamodel::MultivalResponse>(0, request1->get_id(), request1->value_time_start, "open", 1.1));
    aci.request_service.save(std::make_shared<svr::datamodel::MultivalResponse>(0, request2->get_id(), request2->value_time_start, "open", 1.1));
    aci.request_service.save(std::make_shared<svr::datamodel::MultivalResponse>(0, request3->get_id(), request3->value_time_start, "open", 1.1));

    aci.request_service.force_finalize(request2);
    aci.request_service.force_finalize(request3);

    aci.request_service.prune_finalized_requests(bpt::second_clock::local_time() + onesec);

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
