#include "include/DaoTestFixture.h"
#include "DAO/EnsembleDAO.hpp"
#include "common/constants.hpp"
#include <model/User.hpp>

using namespace svr;

using datamodel::Model;
using datamodel::kernel_type;

TEST_F(DaoTestFixture, ModelWorkflow)
{
    size_t decon_level = 2;
    auto paramset = std::make_shared<datamodel::t_param_set>();
    paramset->emplace(std::make_shared<datamodel::SVRParameters>(
            0, 100, "q_svrwave_xauusd_60", "open", 3, decon_level, 0, 0, 0, 0.1, 0.5, .75, 1, 10, 1, 0.5, kernel_type::RBF, 35));
    datamodel::OnlineSVR_ptr svr_model = otr<datamodel::OnlineSVR>(0, 0, *paramset);
    bpt::ptime last_modified = bpt::time_from_string("2015-05-20 10:45:00");
    bpt::ptime last_modeled_value_time = bpt::time_from_string("2015-05-20 10:47:00");

    User_ptr user1 = std::make_shared<datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser",
            datamodel::ROLE::ADMIN, datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<datamodel::InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5),
            "UTC", std::deque<std::string>{"up", "down", "left", "right"});
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<datamodel::Dataset>();//0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{} , datamodel::Priority::Normal, "", 2, "sym7");

    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    auto &ens = ds->get_ensembles();

    ASSERT_EQ(0UL, ens.size());

    datamodel::DeconQueue_ptr p_decon_queue = std::make_shared<datamodel::DeconQueue>("DeconQueuetableName", iq->get_table_name(), "up", ds->get_id(), ds->get_spectral_levels());

    datamodel::DataRow_ptr row = std::make_shared<datamodel::DataRow>(bpt::second_clock::local_time(), bpt::second_clock::local_time(), 1, 1);
    row->set_values({3, 4, 5});

    p_decon_queue->get_data().push_back(row);

    aci.decon_queue_service.save(p_decon_queue);

    datamodel::Ensemble_ptr ensemble(new datamodel::Ensemble(0L, ds->get_id(), p_decon_queue->get_table_name(), std::deque<std::string>{}));

    ds->get_ensembles().push_back(ensemble);

    aci.ensemble_service.save(ensemble);

    auto test_model = std::make_shared<Model>(
            bigint(0), ensemble->get_id(), decon_level, 0, PROPS.get_multistep_len(), 1, common::AppConfig::C_default_kernel_length,
            std::deque{svr_model}, last_modified, last_modeled_value_time);

    datamodel::Model_ptr test_model_0 = std::make_shared<Model>(
            bigint(0), ensemble->get_id(), decon_level, 0, PROPS.get_multistep_len(), 1, common::AppConfig::C_default_kernel_length,
            std::deque{svr_model}, last_modified, last_modeled_value_time);

    ASSERT_FALSE(aci.model_service.exists(*test_model));

    ASSERT_EQ(0UL, aci.model_service.get_all_models_by_ensemble_id(test_model->get_ensemble_id()).size());

    ASSERT_EQ(nullptr, aci.model_service.get_model_by_id(test_model->get_id()).get());

    ASSERT_EQ(0, aci.model_service.remove(test_model));

    ASSERT_EQ(1, aci.model_service.save(test_model));

    ASSERT_EQ(1, aci.model_service.save(test_model_0));

    ASSERT_NE(bigint(0), test_model->get_id());

    ASSERT_NE(bigint(0), test_model_0->get_id());

    ASSERT_TRUE(aci.model_service.exists(*test_model));

    auto models = aci.model_service.get_all_models_by_ensemble_id(ensemble->get_id());

    ASSERT_EQ(2UL, models.size());

    ASSERT_EQ(*test_model, *models.front());
    ASSERT_EQ(*test_model, *aci.model_service.get_model_by_id(test_model->get_id()));

    ASSERT_EQ(1, aci.model_service.remove(test_model));

    ASSERT_EQ(1, aci.model_service.remove_by_ensemble_id(ensemble->get_id()));

    ASSERT_FALSE(aci.model_service.exists(*test_model));

    ASSERT_FALSE(aci.model_service.exists(*test_model_0));

    ASSERT_EQ(0UL, aci.model_service.get_all_models_by_ensemble_id(ensemble->get_id()).size());

    ASSERT_EQ(nullptr, aci.model_service.get_model_by_id(test_model->get_id()).get());

    ASSERT_EQ(0, aci.model_service.remove(test_model));

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}
