#include "include/DaoTestFixture.h"
#include <model/User.hpp>
#include "include/InputQueueRowDataGenerator.hpp"
#include "../../OnlineSVR/test/test_harness.hpp"
#include "online_emd.hpp"

using namespace svr;

TEST_F(DaoTestFixture, DeconQueueWorkflow)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>
            (bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser",
             svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5),
            "UTC",
            std::deque<std::string>{"up", "down", "left", "right"});
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>
            (0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1, C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), 4, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    datamodel::DeconQueue_ptr dq = std::make_shared<svr::datamodel::DeconQueue>("DeconQueuetableName", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());

    // The decon queue is saved with saving the dataset
    //aci.decon_queue_service.save(dq);

    aci.dataset_service.remove(ds);
    aci.decon_queue_service.remove(dq);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

#define TEST_SAVE_DECON
//#define TEST_INPUT_ONLY
#define TEST_DECON_LEVELS 64
#define TEST_LAG_COUNT 400
#define TEST_MAX_RECON_DIFF 1e-7

#define ONLINE_BATCH_LEN 3600
#define ONLINE_LEN 5

#define CORRUPTION_TEST // disables time invariance test
#define TIME_INVARIANCE_OFFSET 150

#define TEST_FIRST_BATCH_PADDING 1000
#define TEST_RESIDUALS_COUNT(p_dataset) ((p_dataset)->get_residuals_length("__DUMMY__")/2)
#define TEST_FIRST_BATCH_SIZE(p_dataset) (TEST_RESIDUALS_COUNT(p_dataset) + TEST_FIRST_BATCH_PADDING)
#define TEST_INPUT_QUEUE_LIMIT(p_dataset) (TEST_FIRST_BATCH_SIZE(p_dataset) + ONLINE_BATCH_LEN + ONLINE_LEN)

void
load_input_queue(const datamodel::Dataset_ptr &p_dataset, const std::string &file_name, datamodel::InputQueue_ptr &p_input_queue)
{
    LOG4_DEBUG("Loading input queue from " << file_name);
    std::ifstream ifstream(file_name);
    if (not ifstream or not ifstream.good()) LOG4_THROW("Input stream not ready.");
    std::string line;
    size_t ctr = 0;
#if !defined(TEST_INPUT_ONLY) and !defined(TEST_SAVE_DECON)
    const auto limit = TEST_INPUT_QUEUE_LIMIT(p_dataset);
#endif
    while (not ifstream.eof()) {
        std::getline(ifstream, line);
        std::vector<std::string> fields;
        if (line.empty()) continue;
        svr::common::split(line, '\t', fields);
        if (fields.size() != 4) continue;
        const auto row_time = bpt::time_from_string(fields[0]);
        const auto p_row = std::make_shared<svr::datamodel::DataRow>(
                row_time,
                boost::posix_time::second_clock::local_time(),
                std::atof(fields[2].c_str()),
                std::vector<double>(1, std::atof(fields[3].c_str())));
        p_input_queue->get_data().push_back(p_row);
        ++ctr;
#if !defined(TEST_INPUT_ONLY) and !defined(TEST_SAVE_DECON)
        if (ctr >= limit / p_input_queue->get_resolution().total_seconds()) break;
#endif
    }
    LOG4_DEBUG("Loaded " << ctr << " rows from " << file_name);
}


TEST_F(DaoTestFixture, testDeconRecon)
{
    User_ptr p_user = std::make_shared<svr::datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High) ;
    // Test of Save and Load to database is disabled for now, TODO enable in future
    aci.user_service.save(p_user);

    datamodel::InputQueue_ptr p_all_data_inputqueue = std::make_shared<svr::datamodel::InputQueue>(
            "EURUSD1S", "EURUSD1S", p_user->get_name(), "EURUSD1S", bpt::seconds(1), bpt::seconds(0), "Europe/Zurich", std::deque<std::string>{"eurusd_avg_bid"} );
    datamodel::InputQueue_ptr p_all_data_inputqueue_1h = std::make_shared<svr::datamodel::InputQueue>(
            "EURUSD1H", "EURUSD1H", p_user->get_name(), "EURUSD1H", bpt::seconds(3600), bpt::seconds(5), "Europe/Zurich", std::deque<std::string>{"eurusd_avg_bid"} );
    aci.input_queue_service.save(p_all_data_inputqueue);
    aci.input_queue_service.save(p_all_data_inputqueue_1h);
    auto p_inputq = p_all_data_inputqueue->clone_empty();
    datamodel::Dataset_ptr p_dataset = std::make_shared<svr::datamodel::Dataset>(
            bigint(0), "DeconQueueTestDataset", p_user->get_user_name(), p_inputq, std::deque<datamodel::InputQueue_ptr>{p_all_data_inputqueue_1h}, svr::datamodel::Priority::Normal, "dsDescription", 1, C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), TEST_DECON_LEVELS, "cvmd");
    APP.ensemble_service.init_default_ensembles(p_dataset);
    p_dataset->set_is_active(true);
    aci.dataset_service.save(p_dataset);

    load_input_queue(p_dataset, "../SVRRoot/OnlineSVR/test/online_emd_test_data/0.98_eurusd_avg_1_test_data.sql", p_all_data_inputqueue);
    load_input_queue(p_dataset, "../SVRRoot/OnlineSVR/test/online_emd_test_data/0.98_eurusd_avg_3600_test_data.sql", p_all_data_inputqueue_1h);
#ifdef TEST_DECON_SAVE
    svr::datamodel::DeconQueue_ptr p_1h_decon_queue;
    PROFILE_EXEC_TIME(p_1h_decon_queue = aci.decon_queue_service.deconstruct(p_all_data_inputqueue_1h, p_dataset, false).at(0),
                      "Deconstruction of 1H single column containing " << p_all_data_inputqueue_1h->size() << " rows.");
    APP.decon_queue_service.save(p_1h_decon_queue);
    return;
#endif

    double total_input_diff = 0;
    size_t compared_ct = 0;
    for (const auto &row_1h: p_all_data_inputqueue_1h->get_data()) {
        const auto it_row_1s = svr::lower_bound_before(p_all_data_inputqueue->get_data(), row_1h->get_value_time());
        // const auto it_row_1s = p_all_data_inputqueue->get_data().find(row_1h->get_value_time());
        if (it_row_1s == p_all_data_inputqueue->end()) {
            LOG4_DEBUG("Row not found in hires data for " << row_1h->get_value_time());
            continue;
        }
        const auto twap_1h = svr::calc_twap(
                it_row_1s, p_all_data_inputqueue->end(),
                row_1h->get_value_time(), row_1h->get_value_time() + p_all_data_inputqueue_1h->get_resolution(),
                p_all_data_inputqueue->get_resolution(), 0);
        if (std::isnan(twap_1h)) continue;
        const auto diff_twap = std::abs(twap_1h - row_1h.get()->get_value(0));
        total_input_diff += diff_twap;
        ++compared_ct;
        if (diff_twap > std::numeric_limits<double>::epsilon())
            LOG4_WARN("Input data inconsistent at " << row_1h->get_value_time() << " difference " << diff_twap << " 1h price " << row_1h.get()->get_value(0) << " mean 1s " << twap_1h << " last 1s iter time " << it_row_1s->get()->get_value_time() << " 1s prices");
        else
            LOG4_TRACE("Data is OK at " << row_1h->get_value_time() << " difference " << diff_twap);
    }
    LOG4_DEBUG("Average input diff " << (total_input_diff/double(compared_ct)) << " out of " << compared_ct << " comparisons.");
#ifdef TEST_INPUT_ONLY
    return;
#endif

    p_inputq->get_data().insert(p_inputq->end(),
            p_all_data_inputqueue->begin(),
            TEST_FIRST_BATCH_SIZE(p_dataset) == INT_MAX ? p_all_data_inputqueue->end() : std::next(p_all_data_inputqueue->begin(), TEST_FIRST_BATCH_SIZE(p_dataset)));

    ASSERT_EQ(p_inputq->size(), (size_t) aci.input_queue_service.save(p_inputq));
    bpt::ptime start_time = aci.input_queue_service.find_oldest_record(p_inputq)->get_value_time();
    bpt::ptime end_time = aci.input_queue_service.find_newest_record(p_inputq)->get_value_time();
    ASSERT_FALSE(start_time.is_special());
    ASSERT_FALSE(end_time.is_special());
    APP.ensemble_service.init_default_ensembles(p_dataset);
    datamodel::DeconQueue_ptr p_online_decon_queue;
    const auto residuals_length = p_dataset->get_residuals_length();
    LOG4_DEBUG("Residuals length " << residuals_length);
    APP.input_queue_service.clear(p_inputq);
    APP.input_queue_service.save(p_inputq);
    PROFILE_EXEC_TIME(p_online_decon_queue = aci.decon_queue_service.deconstruct(p_dataset, p_inputq, "EURUSD1S"),
                      "Deconstruction of single column containing " << p_inputq->size() << " rows.");

    ASSERT_EQ(p_online_decon_queue->size(), p_inputq->size() % 2 ? p_inputq->size() - 1 : p_inputq->size()) << "InputQueue size and deconstracted data size aren't equal";
    for (auto &it : p_online_decon_queue->get_data())
    {
        ASSERT_EQ(it.get()->get_values().size(), (size_t) TEST_DECON_LEVELS) << "Decon queue data at " << it->get_value_time() << " have wrong deconstructed levels";
    }

    svr::business::EnsembleService::update_ensemble_decon_queues(p_dataset->get_ensembles(), {p_online_decon_queue});
    size_t ctr = 0;
    for (auto row_iter = std::next(p_all_data_inputqueue->begin(), TEST_FIRST_BATCH_SIZE(p_dataset));
            row_iter != p_all_data_inputqueue->end(); ++row_iter) {
        p_inputq->get_data().push_back(*row_iter);
        if (++ctr < ONLINE_BATCH_LEN) continue;
        PROFILE_EXEC_TIME(p_online_decon_queue = aci.decon_queue_service.deconstruct(p_dataset, p_inputq, "EURUSD1S"),
                          "Deconstruction of single column containing " << (ctr == ONLINE_BATCH_LEN ? ONLINE_BATCH_LEN : 1) << " rows.");
        svr::business::EnsembleService::update_ensemble_decon_queues(p_dataset->get_ensembles(), {p_online_decon_queue});
    }
    p_online_decon_queue = p_dataset->get_ensemble(0)->get_decon_queue()->clone();
    p_dataset->get_ensemble(0)->get_decon_queue()->get_data().clear();
// Corruption test
#ifndef CORRUPTION_TEST
// Time invariance test
    p_inputq->get_data().erase(p_inputq->begin(), std::next(p_inputq->begin(), TIME_INVARIANCE_OFFSET));
    APP.decon_queue_service.test_start_cvmd_pos = TIME_INVARIANCE_OFFSET;
#endif
    APP.input_queue_service.clear(p_inputq);
    APP.input_queue_service.save(p_inputq);
    auto p_batch_decon_queue = aci.decon_queue_service.deconstruct(p_dataset, p_inputq, "bid");
    size_t row_ix = 0;
    double time_variance = 0;
    auto batch_decon_data = p_batch_decon_queue->get_data();
    for (auto batch_row_iter = std::next(batch_decon_data.begin(), TEST_RESIDUALS_COUNT(p_dataset));
        batch_row_iter != batch_decon_data.end();
        ++batch_row_iter) {

        const auto online_row_iter = svr::find(p_online_decon_queue->get_data(), batch_row_iter->get()->get_value_time());
        if (online_row_iter == p_online_decon_queue->end()) {
            LOG4_WARN("Batch time " << batch_row_iter->get()->get_value_time() << " row " << row_ix <<  " not found in online decon queue.");
            continue;
        }
        double total_diff = 0;
        for (size_t col_ix = 0; col_ix < batch_row_iter->get()->get_values().size(); ++col_ix) {
            const auto diff = std::abs(batch_row_iter->get()->get_value(col_ix) - online_row_iter->get()->get_value(col_ix));
            total_diff += diff;
            if (diff > std::numeric_limits<double>::epsilon())
                LOG4_WARN("Row " << row_ix << " at " << batch_row_iter->get()->get_value_time() << " col " << col_ix << " value difference " << diff);
        }
        time_variance += total_diff;
        if (total_diff > std::numeric_limits<double>::epsilon())
            LOG4_WARN("Row " << row_ix << " at " << batch_row_iter->get()->get_value_time() << " col ix total value difference " << total_diff);
        ++row_ix;
    }
    time_variance /= double(row_ix);
    LOG4_INFO("Average total time variance " << time_variance);
    aci.input_queue_service.clear(p_inputq);
    aci.input_queue_service.save(p_inputq);

    datamodel::InputQueue_ptr recon_queue = p_inputq->clone_empty();
    PROFILE_EXEC_TIME(APP.decon_queue_service.reconstruct(
            *p_online_decon_queue,
            svr::business::recon_type_e::ADDITIVE,
            recon_queue->get_data()),
                      "Reconstruction of " << TEST_DECON_LEVELS << " levels containing " << p_inputq->size() << " rows.");

    double total_diff = 0;
    size_t diff_ct = 0;
    for (const auto &p_iq_row: p_inputq->get_data()) {
        const auto i_rq_row = svr::find(recon_queue->get_data(), p_iq_row->get_value_time());
        if (i_rq_row == recon_queue->end()) {
            LOG4_WARN("Row with time " << p_iq_row->get_value_time() << " not found in recon queue!");
            continue;
        }
        const auto diff = std::abs(i_rq_row->get()->get_value(0) - p_iq_row.get()->get_value(0));

        ++diff_ct;
        if (diff_ct <= residuals_length) continue;

        total_diff += diff;
        if (diff > TEST_MAX_RECON_DIFF)
            LOG4_WARN("Difference of " << diff << " at " << diff_ct << ": " << p_iq_row->get_value_time() << " input price " << p_iq_row.get()->get_value(0) << " recon price " << i_rq_row->get()->get_value(0) << " is too big!");
    }
    const double avg_diff = total_diff / double(diff_ct - residuals_length);

    LOG4_DEBUG("Average diff is " << avg_diff << " total diff is " << total_diff << " compared rows count " << diff_ct - residuals_length);
    ASSERT_LE(avg_diff, TEST_MAX_RECON_DIFF);
    LOG4_DEBUG("p_undelta_online_dq->begin()->second->get_values().size() " << p_online_decon_queue->begin()->get()->get_values().size());
    aci.decon_queue_service.save(p_online_decon_queue);
    datamodel::DeconQueue_ptr p_deconqueue2 = aci.decon_queue_service.get_by_table_name(p_online_decon_queue->get_table_name());

    aci.decon_queue_service.load(p_deconqueue2);

    ASSERT_EQ(*p_online_decon_queue, *p_deconqueue2);

    aci.dataset_service.remove(p_dataset);

    aci.decon_queue_service.remove(p_online_decon_queue);

    aci.input_queue_service.remove(p_inputq);
    aci.user_service.remove(p_user);
}


TEST_F(DaoTestFixture, TestSaveDQIntegrity)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "JamesBond", "JamesBond@email", "JamesBond", "JamesBond", svr::datamodel::ROLE::ADMIN,
            svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "SomeInputQueue", "SomeInputQueue", user1->get_name(), "SomeInputQueue", bpt::seconds(60), bpt::seconds(5),
            "UTC", std::deque<std::string>{"up", "down", "left", "right"});
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "SomeTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1, C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), 2, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    datamodel::DeconQueue_ptr dq = std::make_shared<svr::datamodel::DeconQueue>("SomeDeconQueuetableName", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());

    bpt::ptime nw = bpt::second_clock::local_time();

    datamodel::DataRow_ptr row = std::make_shared<datamodel::DataRow>(nw, bpt::second_clock::local_time(), 1, 1);
    row->set_values({0, 1, 2});
    dq->get_data().push_back(row);

    aci.decon_queue_service.save(dq);
    aci.decon_queue_service.save(dq);

    datamodel::DataRow_ptr row1 = std::make_shared<datamodel::DataRow>(nw + bpt::seconds(1), bpt::second_clock::local_time(), 1, 1);
    row1->set_values({0, 1, 2});
    dq->get_data().push_back(row1);

    datamodel::DeconQueue_ptr dq1 = std::make_shared<svr::datamodel::DeconQueue>("SomeDeconQueuetableName", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());

    aci.decon_queue_service.load(dq1, nw - bpt::hours(1), nw + bpt::hours(1), 1000);

    ASSERT_EQ(1UL, dq1->size());

    aci.dataset_service.remove(ds);

    aci.decon_queue_service.remove(dq);

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, TestDQUpdates)
{
    User_ptr user1 = std::make_shared<svr::datamodel::User>(
            bigint(), "WarrenBuffett", "WarrenBuffett@email", "WarrenBuffett", "WarrenBuffett",
            svr::datamodel::ROLE::ADMIN, svr::datamodel::Priority::High);

    aci.user_service.save(user1);

    datamodel::InputQueue_ptr iq = std::make_shared<svr::datamodel::InputQueue>(
            "GatesFoundationIQ", "GatesFoundationIQ", user1->get_name(), "GatesFoundationIQ", bpt::seconds(60),
            bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"});
    aci.input_queue_service.save(iq);

    datamodel::Dataset_ptr ds = std::make_shared<svr::datamodel::Dataset>(0, "GatesFoundationDS", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}, svr::datamodel::Priority::Normal, "", 1, C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), 2, "sym7");
    ds->set_is_active(true);
    aci.dataset_service.save(ds);

    datamodel::DeconQueue_ptr dq = std::make_shared<svr::datamodel::DeconQueue>("GatesFoundationDQ", iq->get_table_name(), "up", ds->get_id(), ds->get_transformation_levels());

    bpt::ptime nw = bpt::second_clock::local_time();

    datamodel::DataRow_ptr row = std::make_shared<svr::datamodel::DataRow>(nw, bpt::second_clock::local_time(), 1, 1);
    row->set_values({0, 1, 2});
    dq->get_data().push_back(row);

    aci.decon_queue_service.save(dq);

    ///////////// No decon queue table recreation
    dq->get_data().clear();

    datamodel::DataRow_ptr row1 = std::make_shared<datamodel::DataRow>(nw + bpt::seconds(60), bpt::second_clock::local_time(), 1, 1);
    row1->set_values({0, 1, 2});
    dq->get_data().push_back(row1);
    aci.decon_queue_service.save(dq);

    datamodel::DeconQueue_ptr dq_test1 = aci.decon_queue_service.get_by_table_name(dq->get_table_name());
    aci.decon_queue_service.load(dq_test1, nw, nw + bpt::seconds(61), 10);

    ASSERT_EQ(2UL, dq_test1->size());
    ASSERT_EQ(3UL, dq_test1->begin()->get()->get_values().size());

    ///////////// Table should be recreated;

    dq->get_data().clear();

    datamodel::DataRow_ptr row2 = std::make_shared<datamodel::DataRow>(nw + bpt::seconds(2 * 60), bpt::second_clock::local_time(), 1, 1);
    row2->set_values({3, 4, 5});
    dq->get_data().push_back(row2);

    aci.decon_queue_service.save(dq);

    datamodel::DeconQueue_ptr dq_test2 = aci.decon_queue_service.get_by_table_name(dq->get_table_name());
    aci.decon_queue_service.load(dq_test2, nw, nw + bpt::seconds(2 * 60 + 1), 10);

    ASSERT_EQ(3UL, dq_test2->size());
    ASSERT_EQ(3UL, dq_test2->begin()->get()->get_values().size() );

    aci.dataset_service.remove(ds);

    aci.decon_queue_service.remove(dq);

    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);

}
