#include "include/DaoTestFixture.h"
#include "common/constants.hpp"

#include <model/User.hpp>
#include <model/InputQueue.hpp>
#include <model/Dataset.hpp>
#include <model/IQScalingFactor.hpp>

#include "DatasetService.hpp"
#include "InputQueueService.hpp"
#include "UserService.hpp"

using namespace svr;


TEST_F(DaoTestFixture, IQScalingFactorWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "JamesMay", "JamesMay@snailmail", "JamesMay", "JamesMay", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "SlowInputQueue", "SlowInputQueue_logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "JamesMayTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1, common::AppConfig::C_default_kernel_length, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);

    aci.dataset_service.save(ds);

    auto iqsf = otr<datamodel::IQScalingFactor>(0, ds->get_id(), iq->get_table_name(), "column", 1.434);

    ASSERT_EQ(1, aci.iq_scaling_factor_service.save(iqsf));

    ASSERT_NE(0UL, iqsf->get_id());

    ASSERT_TRUE(aci.iq_scaling_factor_service.exists(iqsf));

    auto sfs = aci.iq_scaling_factor_service.find_all_by_dataset_id(ds->get_id());

    ASSERT_EQ(1UL, sfs.size());

    ASSERT_EQ(*iqsf, *sfs[0]);

    ASSERT_EQ(1, aci.iq_scaling_factor_service.remove(iqsf));

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
