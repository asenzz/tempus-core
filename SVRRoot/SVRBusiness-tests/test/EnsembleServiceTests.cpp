#include "appcontext.hpp"
#include "include/DaoTestFixture.h"
#include <DAO/ScopedTransaction.hpp>
#include <model/User.hpp>
#include "include/InputQueueRowDataGenerator.hpp"
#include "InputQueueService.hpp"

using namespace svr;
using svr::datamodel::Priority;
using svr::datamodel::kernel_type;
using svr::common::C_input_queue_table_name_prefix;

class EnsembleIntegrationTests : public DaoTestFixture {
protected:

    scoped_transaction_guard_ptr trx;

    User_ptr testUser;
    datamodel::InputQueue_ptr testQueue;
    datamodel::Dataset_ptr testDataset;
    std::deque<datamodel::DeconQueue_ptr> deconQueues;
    std::deque<datamodel::Ensemble_ptr> ensembles;
    std::map<std::pair<std::string, std::string>, std::vector<datamodel::SVRParameters_ptr>> ensembles_svr_parameters;

    // test configuration
    long testDataNumberToGenerate = 1000;
    int testFrameSize = 256;

    // user details
    std::string userName = "test_user";
    std::string userRealName = "test user";
    std::string userEmail = "svruser@google.com";
    std::string newEmail = "svruser@sf.net";
    std::string userPassword = "svr123456";

    // queue details
    std::string queueName = "simple queue";
    bpt::time_duration resolution = bpt::seconds(60);
    bpt::time_duration legalTimeDeviation = bpt::seconds(5);
    std::string timezone = "Europe/Zurich";
    std::deque<std::string> valueColumns {"high", "low", "open", "close"};
    std::string expectedTableName = business::InputQueueService::make_queue_table_name(userName, queueName, resolution);

    // dataset details
    std::string datasetName = "test dataset";
    Priority priority = Priority::Normal;
    size_t swtLevels = 1;
    std::string swtWaveletName = "db3";
    bpt::time_duration max_lookback_time_gap = bpt::hours(24);
    bool is_active = true;

    //vec_svr_parameters details
    svr::datamodel::SVRParameters svr_parameters;
//            0, 0, expectedTableName, "", 0, 0, 0, mimo_type_e::single,
//            2.05521, 0.16373, 1.4742, 1.40931, 1000, 0.357722, kernel_type::RBF, 10,
//            DEFAULT_APP_HYPERPARAMS(PROPS)};

    void InitSvrParameters() {
        //init vec_svr_parameters
         for (const std::string &column: valueColumns)
         {
             std::vector<svr::datamodel::SVRParameters_ptr> vec_svr_parameters;
             svr_parameters.set_input_queue_column_name(column);
             for (size_t swt_level = 0; swt_level <= swtLevels; swt_level++)
             {
                 svr_parameters.set_decon_level(swt_level);
                 vec_svr_parameters.push_back(std::make_shared<svr::datamodel::SVRParameters>(svr_parameters));
             }
             auto key = std::make_pair(testQueue->get_table_name(), column);
             ensembles_svr_parameters[key] = vec_svr_parameters;
         }
    }

    void InitUser(){
        // init user
        testUser = std::make_shared<svr::datamodel::User>(0, userName, userEmail, userPassword, userRealName,
                                                          svr::datamodel::ROLE::USER);
    }

    void InitDataset(){
        testDataset = std::make_shared<svr::datamodel::Dataset>(0, datasetName, userName, testQueue, std::deque<datamodel::InputQueue_ptr>{},
                                                   priority, "description", 1, C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), swtLevels, swtWaveletName,
                                                   max_lookback_time_gap, std::deque<datamodel::Ensemble_ptr>(), is_active);
        // testDataset->set_ensemble_svr_parameters(ensembles_svr_parameters);
    }

    void InitInputQueueData(){
        // initialize InputQueue object
        testQueue = std::make_shared<svr::datamodel::InputQueue>("", queueName, userName, "description", resolution,legalTimeDeviation,
                timezone, valueColumns);

        // init data generator
        InputQueueRowDataGenerator dataGenerator(aci.input_queue_service, testQueue,
                                                 valueColumns.size(), testDataNumberToGenerate);

        // generate some random data
        //PROFILE_EXEC_TIME( { while ( !dataGenerator.isDone() ); do {
        //    datamodel::DataRow_ptr row = dataGenerator();
        //    aci.input_queue_service.add_row(testQueue, row);
        //} };, "Generating " << testDataNumberToGenerate << " InputQueue rows");
        // should save all the data
    }

    void InitDeconQueueData(){
        bpt::ptime startTime = testQueue->begin()->get()->get_value_time();
        bpt::ptime endTime = testQueue->get_data().rbegin()->get()->get_value_time();

        ASSERT_FALSE(startTime.is_special());
        ASSERT_FALSE(endTime.is_special());

        LOG4_TRACE("Deconstructing test queue");
        //deconQueues = aci.decon_queue_service.deconstruct(testQueue, testDataset);
        for (datamodel::DeconQueue_ptr p_decon : deconQueues) {
            LOG4_DEBUG("Deconstructed Data Queue: " << p_decon->metadata_to_string());
        }
    }

    void InitEnsemble(){
        //ensembles = aci.ensemble_service.init_ensembles_from_dataset(testDataset, deconQueues);
        testDataset->set_ensembles(ensembles);

        ASSERT_EQ(ensembles.size(), deconQueues.size());
    }

    void saveDbData()
    {
        LOG4_TRACE("Saving test user");
        ASSERT_TRUE(aci.user_service.save(testUser) == 1);

        LOG4_TRACE("Saving test queue");
        ASSERT_TRUE(aci.input_queue_service.save(testQueue) > 0);

        LOG4_INFO(testQueue->get_table_name() << " is having data: "
                 << bpt::to_simple_string(testQueue->begin()->get()->get_value_time()) << " - "
                 << bpt::to_simple_string(testQueue->get_data().rbegin()->get()->get_value_time()));

        LOG4_TRACE("Saving test dataset");
        ASSERT_TRUE(aci.dataset_service.save(testDataset) == 1);

    }

    void removeDbData()
    {
        aci.dataset_service.remove(testDataset);
        aci.input_queue_service.remove(testQueue);
        aci.user_service.remove(testUser);
    }

    virtual void SetUp() override
    {
        InitUser();
        InitInputQueueData();
        InitSvrParameters();
        InitDataset();
        InitDeconQueueData();
        InitEnsemble();
        saveDbData();
    }

    virtual void TearDown() override {
        removeDbData();
        trx = nullptr;
    }

};

TEST_F(EnsembleIntegrationTests, testEnsembleCRUD)
{
    ASSERT_EQ(testDataset->get_ensembles().size(), size_t(aci.ensemble_service.remove_by_dataset_id(testDataset->get_id())));
#if 0
    for (size_t i = 0; i < ensembles.size(); ++i) {
        datamodel::Ensemble_ptr &p_ensemble = ensembles.at(i);
         PROFILE_EXEC_TIME(
                aci.ensemble_service.train(
                    testDataset, p_ensemble, testDataset->get_ensemble_svr_parameters()[p_ensemble->get_key_pair()],
                    testDataset->get_max_lookback_time_gap(), 0),
                "Retraining all models");
    }
#endif
    for (const datamodel::Ensemble_ptr &p_ensemble : ensembles) {
        // for each decon level should be there a model
        ASSERT_EQ(p_ensemble->get_models().size(), testDataset->get_transformation_levels() + 1);
    }

    PROFILE_EXEC_TIME(aci.ensemble_service.save_ensembles(ensembles, true), "Persisting ensemble models");

    for (const datamodel::Ensemble_ptr &p_ensemble : ensembles)
    {
        ASSERT_NE(0UL, p_ensemble->get_id());
    }
}
