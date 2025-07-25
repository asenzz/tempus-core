#include "DatasetService.hpp"
#include "InputQueueService.hpp"
#include "UserService.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/InputQueueDAO.hpp"
#include "include/DaoTestFixture.h"
#include "model/User.hpp"

using namespace svr;

TEST_F(DaoTestFixture, SVRParametersWorkflow)
{
    std::string const
            iq_name = "tableName"
            , iq_logical_name = "inputQueue_logical"
            , iq_column_name = "up"
            , user_name = "DeconQueueTestUser";
    bpt::time_duration iq_resolution = bpt::seconds(60);
    std::string iq_table_name = business::InputQueueService::make_queue_table_name(user_name, iq_logical_name, iq_resolution);


    bigint dataset_id = 0; {
        constexpr size_t decon_levels = 4;
        User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), user_name, "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser",
            svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);
        aci.user_service.save(user1);

        datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            iq_name, iq_logical_name, user1->get_name(), "description", iq_resolution, bpt::seconds(5), "UTC",
            std::deque<std::string>{iq_column_name});
        aci.input_queue_service.save(iq);

        aci.flush_dao_buffers();

        auto ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(),
                                                                              iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1,
                                                                              common::AppConfig::C_default_kernel_length, PROPS.get_multistep_len(), decon_levels, "sym7");

        // ASSERT_EQ(ds->get_ensemble_svr_parameters().size(), 0UL);

        auto const svr_param_id = std::make_pair(iq_table_name, iq_column_name);
        //        svr::datamodel::ensemble_svr_parameters_t svr_params;

        for (size_t i = 0; i <= decon_levels; ++i) {
            datamodel::SVRParameters_ptr params = std::make_shared<datamodel::SVRParameters>();
            params->set_svr_C(i + 1);
            //            svr_params[svr_param_id].push_back( params );
        }

        //        ds->set_ensemble_svr_parameters(svr_params);

        aci.dataset_service.save(ds);

        dataset_id = ds->get_id();
    }

    aci.flush_dao_buffers(); {
        datamodel::Dataset_ptr ds = aci.dataset_service.load(dataset_id);
        /*
                auto const svr_params = ds->get_ensemble_svr_parameters();

                ASSERT_EQ(svr_params.size(), 1UL);
                ASSERT_EQ(svr_params.begin()->second.size(), decon_levels + 1);

                for(size_t i = 0; i < svr_params.begin()->second.size(); ++i)
                {
                    auto const & param = svr_params.begin()->second[i];
                    ASSERT_EQ(param->get_input_queue_table_name(), iq_table_name);
                    ASSERT_EQ(param->get_input_queue_column_name(), iq_column_name);
                    ASSERT_EQ(param->get_svr_C(), i + 1);
                }
                */
    } {
        datamodel::Dataset_ptr ds = aci.dataset_service.load(dataset_id);

        //   auto svr_params = ds->get_ensemble_svr_parameters();
        //svr_params.begin()->second[4]->set_svr_C(6);

        //   ds->set_ensemble_svr_parameters(svr_params);

        aci.dataset_service.save(ds);
    }

    aci.flush_dao_buffers(); {
        datamodel::Dataset_ptr ds = aci.dataset_service.load(dataset_id);

        // auto const svr_params = ds->get_ensemble_svr_parameters();

        // ASSERT_EQ(svr_params.size(), 1UL);
        // ASSERT_EQ(svr_params.begin()->second.size(), decon_levels + 1);
        /*
                for(size_t i = 0; i < svr_params.begin()->second.size(); ++i)
                {
                    auto const & param = svr_params.begin()->second[i];
                    ASSERT_EQ(param->get_input_queue_table_name(), iq_table_name);
                    ASSERT_EQ(param->get_input_queue_column_name(), iq_column_name);
                    ASSERT_EQ(param->get_svr_C(), i + 1 + i / decon_levels);
                }
                */
    }


    datamodel::Dataset_ptr ds = aci.dataset_service.load(dataset_id);
    datamodel::InputQueue_ptr iq = aci.input_queue_service.get_queue_metadata(iq_table_name);
    User_ptr user1 = aci.user_service.get_user_by_user_name(user_name);

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
