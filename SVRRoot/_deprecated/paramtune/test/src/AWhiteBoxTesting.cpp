#include "test-config.hpp"
#include "appcontext.hpp"
#include "gtest/gtest.h"
#include <model/User.hpp>
#include <model/InputQueue.hpp>
#include <model/Dataset.hpp>

using namespace svr::datamodel;
using namespace svr::dao;
using namespace svr::business;
using namespace svr::context;
using namespace svr::common;
using namespace bpt;
using namespace std;

namespace {

class WhiteBoxTests : public ::testing::Test {
protected:

    ScopedTransaction_ptr trx;


    InputQueue_ptr ethalon_input_queue;
    std::vector<DeconQueue_ptr> ethalon_decon_queues;
    std::vector<std::vector<double>> ethalon_train_matrix;
    std::vector<double> ethalon_response_vector;
    Dataset_ptr testDataset;

    string input_queue_table_name = "q_svrwave_eurusd_60";

    string user_name = "svrwave";

    //InputQueue details
    const string logical_name = "eurusd";
    const bpt::time_duration resolution = bpt::minutes(1);
    const bpt::time_duration legal_time_deviation = bpt::seconds(5);
    const std::vector<std::string> value_columns = {"open", "high", "low", "close"};


    // dataset details
    string dataset_name = "test dataset";
    Priority priority = Priority::Normal;
    size_t transformation_levels = 4;
    std::string transformation_name = "coif1";
    bpt::time_duration max_lookback_time_gap = bpt::hours(23);
    bool is_active = true;

    // train_data details
    size_t column_index = 0;
    size_t lookback_rows = 5;
    size_t model_number = 2;
    double adjacent_levels_ratio = 0.5;
    size_t decremental_offset = 10;

    //vec_svr_parameters details
//    SVRParameters svr_parameters = SVRParameters(0, 0, 0, 10, 0.035, 19, 0.11, 1, 0.5, kernel_type::RBF, 10);

    //times for data
    time_period times{time_from_string("2015-03-20 22:00:00"), time_from_string("2015-03-23 09:59:00")};

    void loadEthalonData()
    {
        // set ethalon InputQueue data
        ifstream f("../SVRRoot/paramtune/test/input_data.txt");
        ASSERT_TRUE(f.is_open());
        string line;
        DataRow::Container ethalon_input_data;
        while (std::getline(f, line))
        {
            stringstream ss(line);
            //2015-03-20 22:00:00	2015-11-17 12:54:50	93	1.08088999999999991	1.0811599999999999	1.08088999999999991	1.08102999999999994
            string time[4];
            double val[5];
            ss >> time[0] >> time[1] >> time[2] >> time[3];
            ss >> val[0] >> val[1] >> val[2] >> val[3] >> val[4];
            DataRow row = DataRow(
                    bpt::time_from_string(time[0] + " " + time[1]),
                                        bpt::time_from_string(time[2] + " " + time[3]),
                                        val[0], {val[1], val[2], val[3], val[4]});

            ethalon_input_data[row.get_value_time()] = make_shared<DataRow>(row);
        }
        f.close();
        ethalon_input_queue = std::make_shared<InputQueue>(string(), logical_name, user_name, string(), resolution,
                                                        legal_time_deviation, time_duration(), value_columns);
        ethalon_input_queue->set_data(ethalon_input_data);

        // set ethalon DeconQueue data
        f.open("../SVRRoot/paramtune/test/deconstracted_data.txt");
        ASSERT_TRUE(f.is_open());
        std::vector<DataRow::Container> ethalon_decon_data = std::vector<DataRow::Container>(4);
        while (getline(f, line))
        {
            stringstream ss(line);
            //2015-Mar-20 22:04:00 55 4.32389 0.000851253 0.000261426 6.09629e-05 2.10399e-05 4.32457 0.000870668
//            0.000512203 0.000141853 -1.99657e-06 4.32348 0.000815653 0.000242096 -7.46687e-06 4.02276e-05 4.32414
//            0.000816966 0.000489874 0.000132351 1.33068e-05
            string time[2];
            double val[21];
            ss >> time[0] >> time[1];
            for (int i = 0; i < 21 ; ++i)
            {
                ss >> val[i];
            }
            for (int col = 0; col < 4; ++col)
            {
                size_t index = col * 4;
                DataRow row = DataRow(bpt::time_from_string(time[0] + " " + time[1]),
                                        bpt::second_clock::local_time(),
                                        val[0], {val[index + 1], val[index+ 2], val[index + 3], val[index + 4], val[index + 5]});
                ethalon_decon_data[col][row.get_value_time()] = make_shared<DataRow>(row);
            }
        }
        f.close();
        for (size_t i = 0; i < value_columns.size(); ++i)
        {
            DeconQueue_ptr decon = make_shared<DeconQueue>("", input_queue_table_name,value_columns[i],
                                                           testDataset->get_id());
            decon->set_data(ethalon_decon_data[i]);
            ethalon_decon_queues.push_back(decon);
        }

        f.open("../SVRRoot/paramtune/test/train_data.txt");
        ASSERT_TRUE(f.is_open());
        while(getline(f, line))
        {
            stringstream ss(line);
            double y;
            ss >> y;
            string val;
            std::vector<double> x;
            while (ss >> val)
            {
                x.push_back(stod(val.substr(val.find_first_of(":") + 1)));
            }
            ethalon_response_vector.push_back(y);
            ethalon_train_matrix.push_back(x);
        }
        f.close();
    }




    virtual void SetUp() override
    {
        loadEthalonData();
        testDataset = std::make_shared<Dataset>(0, dataset_name, user_name, ethalon_input_queue, std::vector<InputQueue_ptr>(),
                                            priority, "", transformation_levels, transformation_name, max_lookback_time_gap);
    }

    virtual void TearDown() override {
//        removeDbData();
        trx = nullptr;
    }

};


TEST_F(WhiteBoxTests, checkEthalonData)
{

    // inputQueue checking
    const DataRowContainer loaded_queue_data = AppContext::get_instance().input_queue_service.get_queue_data(input_queue_table_name,
                                                                                 times.begin(), times.end());
    const DataRowContainer& ethalon_queue_data = ethalon_input_queue->get_data();
    ASSERT_EQ(loaded_queue_data.size(), ethalon_queue_data.size());
    for (DataRowContainer::const_iterator it = loaded_queue_data.begin(); it != loaded_queue_data.end(); ++it)
    {
        ASSERT_EQ(*(it->second), *(ethalon_queue_data.at(it->first)));
    }

    // deconQueue checking
    std::vector<DeconQueue_ptr> decon_queues = AppContext::get_instance().decon_queue_service.deconstruct(ethalon_input_queue, testDataset);
    ASSERT_EQ(decon_queues.size(), ethalon_decon_queues.size());
    for (size_t i = 0; i < decon_queues.size(); ++i)
    {
        const DataRowContainer &real_data = decon_queues[i]->get_data();
        const DataRowContainer &ethalon_data = ethalon_decon_queues[i]->get_data();
        ASSERT_EQ(real_data.size(), ethalon_data.size());
        for (DataRowContainer::const_iterator it = real_data.begin(); it != real_data.end(); ++it)
        {
            DataRow tmp1 = *ethalon_data.at(it->first);
            DataRow tmp2 = *(it->second);
            ASSERT_TRUE(*(it->second) == *ethalon_data.at(it->first));
        }
    }

    // train matrix checking
    const std::vector<size_t> adjacent_levels = get_adjacent_indexes(
            model_number, adjacent_levels_ratio, transformation_levels + 1);

    /* TODO add auxilliary decon queue tests */
    auto training_data = AppContext::get_instance().model_service.get_training_data(
            datarow_range(decon_queues[column_index]->get_data().begin(), decon_queues[column_index]->get_data().end(), decon_queues[column_index]->get_data()),
            lookback_rows,
            adjacent_levels,
            max_lookback_time_gap,
            model_number);

    std::shared_ptr<Matrix<double>> training_matrix = std::make_shared<Matrix<double>>(*training_data.training_matrix);
    std::shared_ptr<Vector<double>> response_vector = std::make_shared<Vector<double>>(*training_data.reference_vector);
    ASSERT_EQ(training_matrix->GetLengthRows(), response_vector->GetLength())
                << "Training matrix row length [" << training_matrix->GetLengthRows()
                << "] and response vector size [" << response_vector->GetLength() << "] are not the same";

    for (int row_numb = 0; row_numb < training_matrix->GetLengthRows(); ++row_numb)
    {
        ASSERT_EQ(response_vector->GetValue(row_numb), ethalon_response_vector.at(row_numb))
                << "Corrupt response vector index " << row_numb << " values mismatch current "
                << response_vector->GetValue(row_numb) << " expected "
                << ethalon_response_vector.at(row_numb);

        Vector<double> *train_row = training_matrix->GetRowRef(row_numb);
        const std::vector<double>& ethalon_train_row = ethalon_train_matrix.at(row_numb);
        ASSERT_EQ(size_t(train_row->GetLength()), ethalon_train_row.size())
                << "Corrupt training vector index [" << row_numb <<"]: column length mismatch, expected ["
                << ethalon_train_matrix.at(row_numb).size() << "]";

        for (int col_numb = 0; col_numb < train_row->GetLength(); ++col_numb)
        {
            ASSERT_EQ(train_row->GetValue(col_numb), ethalon_train_row.at(col_numb))
                    << "Corrupt train vector index " << row_numb << ", " << col_numb << " values mismatch. Current "
                    << train_row->GetValue(col_numb) << ", expected " << ethalon_train_row.at(col_numb);
        }
    }
}


} // anon namespace
