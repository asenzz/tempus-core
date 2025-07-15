#include "include/DaoTestFixture.h"
#include "common/constants.hpp"
#include <iostream>
#include <vector>
#include <model/Model.hpp>
#include <model/User.hpp>
#include <model/Dataset.hpp>
#include <model/AutotuneTask.hpp>

using namespace svr;

using svr::datamodel::AutotuneTask;

TEST_F(DaoTestFixture, AutotuneTaskWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 4, 1, common::AppConfig::C_default_kernel_length, PROPS.get_multistep_len(), "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    bigint                  result_dataset_id   = 101;
    bpt::ptime              creation_time       = bpt::second_clock::local_time();
    bpt::ptime              done_time           = creation_time + bpt::hours(2);
    std::map<std::string, std::string> parameters = {
            {"transformation_levels", "1,4"},
            {"transformation_name", "bior3.1..bior3.5"},
            {"svr_c_0", "5.0..200.0"}, {"svr_epsilon_0", "0.000001..0.003"}, {"svr_kernel_param_0", "0.1..2.5"}, {"svr_kernel_param2_0", "0"}, {"svr_decremental_distance_0", "1000"}, {"svr_adjacent_levels_ratio_0", "0.5"}, {"svr_kernel_type_0", "3"}, {"lag_count_0", "95"},
            {"svr_c_1", "5.0..200.0"}, {"svr_epsilon_1", "0.000001..0.003"}, {"svr_kernel_param_1", "0.1..2.5"}, {"svr_kernel_param2_1", "0"}, {"svr_decremental_distance_1", "1000"}, {"svr_adjacent_levels_ratio_1", "0.5"}, {"svr_kernel_type_1", "3"}, {"lag_count_1", "25"},
            {"svr_c_2", "5.0..200.0"}, {"svr_epsilon_2", "0.000001..0.003"}, {"svr_kernel_param_2", "0.1..2.5"}, {"svr_kernel_param2_2", "0"}, {"svr_decremental_distance_2", "1000"}, {"svr_adjacent_levels_ratio_2", "0.5"}, {"svr_kernel_type_2", "3"}, {"lag_count_2", "25"},
            {"svr_c_3", "5.0..200.0"}, {"svr_epsilon_3", "0.000001..0.003"}, {"svr_kernel_param_3", "0.1..2.5"}, {"svr_kernel_param2_3", "0"}, {"svr_decremental_distance_3", "1000"}, {"svr_adjacent_levels_ratio_3", "0.5"}, {"svr_kernel_type_3", "3"}, {"lag_count_3", "25"}
    };
    bpt::ptime              start_train_time    = bpt::time_from_string("2015-05-20 10:45");
    bpt::ptime              end_train_time      = bpt::time_from_string("2015-05-20 10:47");
    bpt::ptime              start_validation_time = bpt::time_from_string("2015-05-20 10:48");
    bpt::ptime              end_validation_time = bpt::time_from_string("2015-05-20 10:50");
    int                     status              = 0;         // 0 - new, 1 - in process, 2 - done, 3 - error
    double                  mse                 = 0.123;

    size_t vp_sliding_direction = 0, vp_slide_count = 0;
    bpt::seconds vp_slide_period_sec = bpt::seconds(0);
    size_t pso_best_points_counter = 0, pso_iteration_number  = 0, pso_particles_number = 0, pso_topology = 0, nm_max_iteration_number = 0;
    double nm_tolerance  = 0;

    AutotuneTask_ptr test_autotune_task = std::make_shared<AutotuneTask>(0, ds->get_id(), result_dataset_id,
                                                creation_time, done_time, parameters,
                                                start_train_time, end_train_time,
                                                start_validation_time, end_validation_time,
                                                vp_sliding_direction, vp_slide_count, vp_slide_period_sec,
                                                pso_best_points_counter, pso_iteration_number,
                                                pso_particles_number, pso_topology,
                                                nm_max_iteration_number, nm_tolerance,
                                                status, mse);

    ASSERT_FALSE(aci.autotune_task_service.exists(test_autotune_task));

    EXPECT_TRUE(0UL == aci.autotune_task_service.find_all_by_dataset_id(test_autotune_task->get_dataset_id()).size());

    EXPECT_TRUE(nullptr == aci.autotune_task_service.get_by_id(test_autotune_task->get_id()).get());

    EXPECT_TRUE(0 == aci.autotune_task_service.remove(test_autotune_task));

    EXPECT_TRUE(1 == aci.autotune_task_service.save(test_autotune_task));

    ASSERT_NE(bigint(0), test_autotune_task->get_id());

    ASSERT_TRUE(aci.autotune_task_service.exists(test_autotune_task));

    auto userAutotuneTasks = aci.autotune_task_service.find_all_by_dataset_id(ds->get_id());

    EXPECT_TRUE(1UL == userAutotuneTasks.size());

    EXPECT_TRUE(parameters == userAutotuneTasks.front()->get_parameters());

    EXPECT_TRUE(*test_autotune_task == *userAutotuneTasks.front());
    EXPECT_TRUE(*test_autotune_task == *aci.autotune_task_service.get_by_id(test_autotune_task->get_id()));

    EXPECT_TRUE(1 == aci.autotune_task_service.remove(test_autotune_task));

    ASSERT_FALSE(aci.autotune_task_service.exists(test_autotune_task));

    EXPECT_TRUE(0UL == aci.autotune_task_service.find_all_by_dataset_id(test_autotune_task->get_id()).size());

    EXPECT_TRUE(nullptr == aci.autotune_task_service.get_by_id(test_autotune_task->get_id()).get());

    EXPECT_TRUE(0 == aci.autotune_task_service.remove(test_autotune_task));

    aci.autotune_task_service.remove(test_autotune_task);
    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
