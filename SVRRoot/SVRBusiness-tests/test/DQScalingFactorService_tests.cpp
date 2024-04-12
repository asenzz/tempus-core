#include "include/DaoTestFixture.h"
#include "common/constants.hpp"

#include <model/User.hpp>
#include <model/InputQueue.hpp>
#include <model/Dataset.hpp>

using namespace svr;

TEST_F(DaoTestFixture, DQScalingFactorWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "EmmaWatson", "EmmaWatson@somedorf", "EmmaWatson", "EmmaWatson", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "CursedChildInputQueue", "CursedChildInputQueue_logicalName", user1->get_name(), "description"
            , bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );

    aci.input_queue_service.save(iq);

 datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(
         0, "EmmaWatsonTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);

    aci.dataset_service.save(ds);


    datamodel::DQScalingFactor_ptr dqsf = otr<datamodel::DQScalingFactor>(0, 0, 0, 0, 0, 1.434, 0, 0, 0);

    EXPECT_TRUE(1 == aci.dq_scaling_factor_service.save(dqsf));

    EXPECT_TRUE(1 == aci.dq_scaling_factor_service.save(dqsf));

    ASSERT_NE(0UL, dqsf->get_id());

    ASSERT_TRUE(aci.dq_scaling_factor_service.exists(dqsf));

    auto sfs = aci.dq_scaling_factor_service.find_all_by_model_id(ds->get_id());

    EXPECT_TRUE(1UL == sfs.size());

    EXPECT_TRUE(*dqsf == **sfs.begin());

    EXPECT_TRUE(1 == aci.dq_scaling_factor_service.remove(dqsf));

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

#define ASSERT_FP_EQ(op1, op2) ASSERT_LT(fabs((op1) - (op2)), std::numeric_limits<float>::epsilon())

TEST_F(DaoTestFixture, DQScalingFactorScalingUnscaling)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "EmmaWatson", "EmmaWatson@somedorf", "EmmaWatson", "EmmaWatson", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "CursedChildInputQueue", "CursedChildInputQueue_logicalName", user1->get_name(), "description"
            , bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );

    aci.input_queue_service.save(iq);

 datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "EmmaWatsonTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{},
                                                                       svr::datamodel::Priority::Normal, "", 1, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);

    aci.dataset_service.save(ds);

    datamodel::DeconQueue_ptr dq = std::make_shared<svr::datamodel::DeconQueue>("EmmaWatsonDeconQueue", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());
    auto lt = bpt::second_clock::local_time();

    const size_t dq_len = 1e+6;

    for(size_t i = 0; i < dq_len; ++i)
    {
        double v1 = i%10, v2 = i%11, v3 = i%12, v4 = i%13;
        lt += bpt::seconds(1);
        auto dr = std::make_shared<svr::datamodel::DataRow>(lt, lt, 0.0, std::vector{v1, v2, v3, v4});
        dq->get_data().push_back(dr);
    }
#if 0
    auto sf_cont = aci.dq_scaling_factor_service.calculate(
            dq->begin(), dq->end(), dq->get_input_queue_table_name(), dq->get_input_queue_column_name(), ds->get_id(), ds->get_transformation_levels());

    EXPECT_TRUE(sf_cont.size() == 4UL);

    std::vector<double> sf_vec;
    std::vector<double> mean_val_vec;

    for(auto const & sf: sf_cont) {
        sf_vec.push_back(sf->get_features_factor());
        mean_val_vec.push_back(sf->get_labels_factor());
    }

    ASSERT_FP_EQ(9 , sf_vec[0]);
    ASSERT_FP_EQ(10, sf_vec[1]);
    ASSERT_FP_EQ(11, sf_vec[2]);
    ASSERT_FP_EQ(12, sf_vec[3]);


    // aci.dq_scaling_factor_service.scale_decon_queue(dq, sf_vec, mean_val_vec);

    size_t dq_idx = 0;

    for(auto const & row : dq->get_data())
    {
        ASSERT_FP_EQ(row.get()->get_values()[0], (double(dq_idx % 10) - mean_val_vec[0])/ sf_vec[0] );
        ASSERT_FP_EQ(row.get()->get_values()[1], (double(dq_idx % 11) - mean_val_vec[1])/ sf_vec[1] );
        ASSERT_FP_EQ(row.get()->get_values()[2], (double(dq_idx % 12) - mean_val_vec[2])/ sf_vec[2] );
        ASSERT_FP_EQ(row.get()->get_values()[3], (double(dq_idx % 13) - mean_val_vec[3])/ sf_vec[3] );
        ++dq_idx;
    }

    for (datamodel::DQScalingFactor_ptr const & dq_scaling_factor: sf_cont)
        aci.dq_scaling_factor_service.save(dq_scaling_factor);

    auto sfs_loaded = aci.dq_scaling_factor_service.find_all_by_dataset_id(ds->get_id());
    for (datamodel::DQScalingFactor_ptr const & loaded_scaling_factor: sfs_loaded)
    {
        ASSERT_FP_EQ(loaded_scaling_factor->get_labels_factor(), mean_val_vec[loaded_scaling_factor->get_decon_level()]);
        ASSERT_FP_EQ(loaded_scaling_factor->get_features_factor(), sf_vec[loaded_scaling_factor->get_decon_level()]);
    }

//    aci.dq_scaling_factor_service.unscale_decon_queue(dq, sf_vec, mean_val_vec);

    dq_idx = 0;
    for(auto const & row : dq->get_data())
    {
        ASSERT_FP_EQ(row.get()->get_values()[0], double(dq_idx % 10));
        ASSERT_FP_EQ(row.get()->get_values()[1], double(dq_idx % 11));
        ASSERT_FP_EQ(row.get()->get_values()[2], double(dq_idx % 12));
        ASSERT_FP_EQ(row.get()->get_values()[3], double(dq_idx % 13));
        ++dq_idx;
    }
#endif
    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
