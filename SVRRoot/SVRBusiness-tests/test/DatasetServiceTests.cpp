#include <iostream>
#include "common/defines.h"
#include "common/constants.hpp"
#include "include/DaoTestFixture.h"
#include "model/User.hpp"
#include "model/InputQueue.hpp"
#include "model/Dataset.hpp"
#include "include/InputQueueRowDataGenerator.hpp"
#include "../../OnlineSVR/include/recombine_parameters.cuh"
#include "util/time_utils.hpp"
#include "util/string_utils.hpp"

using namespace svr;

TEST_F(DaoTestFixture, DatasetTuningRecombination)
{
    const uint32_t colct = 31; // levct
    const uint32_t rowct = 34144256;
    double best_score = std::numeric_limits<double>::max();
    std::vector<svr::t_param_preds_cu> params_preds(colct * svr::common::C_tune_keep_preds);
    for (uint32_t i = 0; i < colct * svr::common::C_tune_keep_preds; ++i) {
        params_preds[i].params_ix = 0;
        for (uint32_t j = 0; j < C_emo_max_j; ++j) {
            for (uint32_t el = 0; el < C_emo_test_len; ++el) {
                params_preds[i].labels[j][el] = 1;
                params_preds[i].last_knowns[j][el] = 0;
                params_preds[i].predictions[j][el] = 0;
            }
        }
    }
#if 0
    std::vector<uint8_t> combos(rowct * colct, uint8_t(1));
#endif
    for (uint32_t colix = 0; colix < colct; ++colix) {
#if 0
        combos[.5 * rowct * colct + colix] = 0;
#endif
        params_preds[colix].params_ix = 77;
        for (uint32_t j = 0; j < C_emo_max_j; ++j)
            for (uint32_t el = 0; el < C_emo_test_len; ++el)
                if (el % 2) params_preds[colix].predictions[j][el] = 100;
    }
    arma::uchar_mat combos(rowct, colct, arma::fill::ones);
    combos.row(combos.n_rows / 2).fill(0);
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
    PROFILE_EXEC_TIME(svr::recombine_parameters(rowct, colct, combos.memptr(), params_preds.data(), &best_score, best_params_ixs.data()), "recombine_parameters");
    LOG4_DEBUG("Best score " << best_score << ", best params ixs " << common::to_string(best_params_ixs));
}


TEST_F(DaoTestFixture, DatasetWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}
            , svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);

    ds->set_max_lookback_time_gap(svr::common::date_time_string_to_seconds("38,21:22:23"));

    aci.dataset_service.save(ds);

    svr::business::DatasetService::UserDatasetPairs dsu;
    aci.dataset_service.update_active_datasets(dsu);
    EXPECT_TRUE(2UL == dsu.size());

    datamodel::Dataset_ptr &p_dataset = dsu[0].p_dataset;
    p_dataset->set_input_queue(
            aci.input_queue_service.get_queue_metadata(
                    p_dataset->get_input_queue()->get_table_name()));

    EXPECT_TRUE(p_dataset->get_max_lookback_time_gap() == bpt::hours(38*24 + 21) + bpt::minutes(22) + bpt::seconds(23) );

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, SelectingActiveDatasets)
{
    User_ptr user1Low = std::make_shared<svr::datamodel::User>(
            bigint(), "User2016-07-20-Low", "User2016-07-20-Low@dkdk.dld", "User2016-07-20-Low", "User2016-07-20-Low", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::Low) ;

    aci.user_service.save(user1Low);

    datamodel::InputQueue_ptr iq1 = std::make_shared<svr::datamodel::InputQueue>(
            "InputQueue1", "InputQueue1", user1Low->get_name(), "InputQueue1", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq1);

    datamodel::Dataset_ptr ds1 = std::make_shared<svr::datamodel::Dataset>(0, "Dataset2016-07-20-Low", user1Low->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{}
            , svr::datamodel::Priority::Low, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds1->set_is_active(true);

    aci.dataset_service.save(ds1);

    ////////////////////////////////////////////////////////////////////////////

    svr::business::DatasetService::UserDatasetPairs pairs;
    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(2UL == pairs.size());
    auto iter = pairs.begin();
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "eurusd"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "svrwave");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Low"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    User_ptr user2Normal = std::make_shared<svr::datamodel::User>(
            bigint(), "User2016-07-20-Normal", "User2016-07-20-Normal@dkdk.dld", "User2016-07-20-Normal", "User2016-07-20-Normal"
            , svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::Normal) ;

    aci.user_service.save(user2Normal);

    datamodel::InputQueue_ptr iq2 = std::make_shared<svr::datamodel::InputQueue>(
            "InputQueue2", "InputQueue2", user2Normal->get_name(), "InputQueue2", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq2);

    datamodel::Dataset_ptr ds2 =
            std::make_shared<svr::datamodel::Dataset>(0, "Dataset2016-07-20-Below", user2Normal->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{},
                                                      svr::datamodel::Priority::BelowNormal, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds2->set_is_active(true);

    aci.dataset_service.save(ds2);

    ////////////////////////////////////////////////////////////////////////////

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(3UL == pairs.size());

    iter = pairs.begin();
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Below"); EXPECT_TRUE(1UL == iter->users.size());EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Normal");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "eurusd"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "svrwave");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Low"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    ASSERT_FALSE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );
    ASSERT_TRUE ( aci.dataset_service.link_user_to_dataset( user2Normal, ds1 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(3UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    datamodel::Dataset_ptr ds3 = std::make_shared<svr::datamodel::Dataset>(0, "Dataset2016-07-20-High-3", user2Normal->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::High, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds3->set_is_active(true);

    aci.dataset_service.save(ds3);

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low"); ASSERT_EQ(2UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal"); ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    ASSERT_TRUE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );
    ASSERT_FALSE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    User_ptr user3High = std::make_shared<svr::datamodel::User>(
        bigint(), "User2016-07-20-High-3", "User2016-07-20-High-3@dkdk.dld", "User2016-07-20-High-3", "User2016-07-20-High-3", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user3High);

    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds1 ) );
    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds2 ) );
    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds3 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    aci.user_service.remove(user3High);
    aci.dataset_service.remove(ds3);

    aci.dataset_service.remove(ds2);
    aci.input_queue_service.remove(iq2);
    aci.user_service.remove(user2Normal);


    aci.dataset_service.remove(ds1);
    aci.input_queue_service.remove(iq1);
    aci.user_service.remove(user1Low);
}
