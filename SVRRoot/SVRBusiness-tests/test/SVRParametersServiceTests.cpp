#include "include/DaoTestFixture.h"
#include "DAO/SVRParametersDAO.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/InputQueueDAO.hpp"
#include <iostream>

#include <model/User.hpp>

using svr::datamodel::SVRParameters;

TEST_F(DaoTestFixture, SVRParametersWorkflow)
{
    std::string const
              iq_name = "tableName"
            , iq_logical_name = "inputQueue_logical"
            , iq_column_name = "up"
            , user_name = "DeconQueueTestUser"
            ;
    size_t const decon_levels = 4;
    bpt::time_duration iq_resolution = bpt::seconds(60);
    std::string iq_table_name = svr::datamodel::InputQueue::make_queue_table_name(user_name, iq_logical_name, iq_resolution);


    bigint dataset_id = 0;
    {
        User_ptr user1 = std::make_shared<svr::datamodel::User>(
                bigint(), user_name, "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;
        aci.user_service.save(user1);

        InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
                iq_name, iq_logical_name, user1->get_name(), "description", iq_resolution, bpt::seconds(5), "UTC", std::vector<std::string>{iq_column_name} );
        aci.input_queue_service.save(iq);

        aci.flush_dao_buffers();

        Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::vector<InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", decon_levels, "sym7");

        ASSERT_EQ(ds->get_ensemble_svr_parameters().size(), 0UL);

        auto const svr_param_id = std::make_pair(iq_table_name, iq_column_name);
        svr::datamodel::ensemble_svr_parameters_t svr_params;

        for(size_t i = 0; i <= decon_levels; ++i)
        {
            SVRParameters_ptr params  = SVRParameters_ptr(new SVRParameters());
            params->set_svr_C(i+1);
            svr_params[svr_param_id].push_back( params );
        }

        ds->set_ensemble_svr_parameters(svr_params);

        aci.dataset_service.save(ds);

        dataset_id = ds->get_id();
    }

    aci.flush_dao_buffers();

    {
        Dataset_ptr ds = aci.dataset_service.get(dataset_id);

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
    }

    {
        Dataset_ptr ds = aci.dataset_service.get(dataset_id);

        auto svr_params = ds->get_ensemble_svr_parameters();
        svr_params.begin()->second[4]->set_svr_C(6);

        ds->set_ensemble_svr_parameters(svr_params);

        aci.dataset_service.save(ds);
    }

    aci.flush_dao_buffers();

    {
        Dataset_ptr ds = aci.dataset_service.get(dataset_id);

        auto const svr_params = ds->get_ensemble_svr_parameters();

        ASSERT_EQ(svr_params.size(), 1UL);
        ASSERT_EQ(svr_params.begin()->second.size(), decon_levels + 1);

        for(size_t i = 0; i < svr_params.begin()->second.size(); ++i)
        {
            auto const & param = svr_params.begin()->second[i];
            ASSERT_EQ(param->get_input_queue_table_name(), iq_table_name);
            ASSERT_EQ(param->get_input_queue_column_name(), iq_column_name);
            ASSERT_EQ(param->get_svr_C(), i + 1 + i / decon_levels);
        }
    }



    Dataset_ptr ds = aci.dataset_service.get(dataset_id);
    InputQueue_ptr iq = aci.input_queue_service.get_queue_metadata(iq_table_name);
    User_ptr user1 = aci.user_service.get_user_by_user_name(user_name);

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
