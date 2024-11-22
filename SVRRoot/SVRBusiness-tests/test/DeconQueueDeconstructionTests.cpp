#include "include/DaoTestFixture.h"
#include "util/math_utils.hpp"
#include "common/constants.hpp"
#include <iostream>
#include <memory>

using namespace svr;

namespace
{
    std::string new_datarow_container = "new_datarow_container";
}

void save_datarow_containers(std::vector<datamodel::DeconQueue_ptr> &decon_queues, std::string path);
std::deque<svr::datamodel::DataRow::container> load_datarow_containers(const std::string &file_path);
void compare_datarow_containers(svr::datamodel::DataRow::container const & lhs, svr::datamodel::DataRow::container const & rhs);

TEST_F(DaoTestFixture, BasicDeconstructionTest)
{
    bpt::time_period times{bpt::time_from_string("2017-05-02 22:00:00"), bpt::time_from_string("2017-05-05 05:59:00")};
    std::string const test_iq_name = "q_svrwave_eurusd_60";

    datamodel::InputQueue_ptr test_iq = aci.input_queue_service.get_queue_metadata(test_iq_name);

    const svr::data_row_container test_iq_data
        = aci.input_queue_service.load(test_iq_name, times.begin(), times.end());

    test_iq->set_data(test_iq_data);

 datamodel::Dataset_ptr dataset_100 = aci.dataset_service.get_user_dataset("svrwave", "eurusd");

    ////////////////////////////////////////////////////////////////////////////
    //
    // Test deconstruction
    //

    std::deque<datamodel::DeconQueue_ptr> decon_queues;// = aci.decon_queue_service.deconstruct(test_iq, dataset_100);

//    save_datarow_containers(decon_queues, "../SVRRoot/SVRBusiness-tests/test/test_data/etalon_decon_output.txt");

    std::string etalon_decon_output_path = exec("find ../SVRRoot/SVRBusiness-tests/test/test_data -name etalon_decon_output.txt");
    erase_after(etalon_decon_output_path, '\n');

    auto etalon_decon_containers = load_datarow_containers(etalon_decon_output_path);

    ASSERT_EQ(etalon_decon_containers.size(), decon_queues.size());

    auto ri = decon_queues.begin();

    for(auto ei = etalon_decon_containers.begin(); ri != decon_queues.end(); ++ei, ++ri)
        compare_datarow_containers(*ei, (*ri)->get_data());

    ////////////////////////////////////////////////////////////////////////////
    //
    // Test training data
    //

//    size_t column_index = 0;
    size_t lookback_rows = 5;
    size_t model_number = 2;
    double adjacent_levels_ratio = 0.5;
    size_t transformation_levels = 4;
//    bpt::time_duration max_lookback_time_gap = bpt::hours(23);

    const auto adjacent_levels = business::SVRParametersService::get_adjacent_indexes(model_number, adjacent_levels_ratio, transformation_levels + 1);

    /* TODO add auxilliary decon queue tests */
    arma::mat features, labels;
    arma::vec last_knowns;
    std::deque<bpt::ptime> label_times;
#if 0 // TODO Fix for new implementation
    aci.model_service.get_training_data(
            features, labels, last_knowns, label_times,
            datamodel::datarow_range{decon_queues[column_index]->get_data()},
            datamodel::datarow_range{decon_queues[column_index]->get_data()},
            {datamodel::datarow_range{decon_queues[column_index]->get_data()}},
            lookback_rows,
            adjacent_levels,
            max_lookback_time_gap,
            model_number,
            test_iq->get_resolution(),
            bpt::min_date_time,
            test_iq->get_resolution(),
            1);
#endif
    const auto params = std::make_shared<svr::datamodel::SVRParameters>(
            0,
            dataset_100->get_id(),
            test_iq_name,
            "eurusd",
            1, 0, 0, 0, 0,
            datamodel::C_default_svrparam_svr_cost,
            datamodel::C_default_svrparam_svr_epsilon,
            datamodel::C_default_svrparam_kernel_param1,
            datamodel::C_default_svrparam_kernel_param2,
            datamodel::C_default_svrparam_decrement_distance,
            adjacent_levels_ratio,
            datamodel::C_default_svrparam_kernel_type,
            lookback_rows);

    auto real_training_matrix = std::make_shared<arma::mat>(features);
    auto real_response_matrix = std::make_shared<arma::mat>(labels);

    arma::vec real_response_vektor(real_response_matrix->n_rows);
    for(size_t i = 0; i < real_response_matrix->n_rows; ++i)
        real_response_vektor[i] = (*real_response_matrix)(i, 0);

    std::string etalon_response_vector_output_path = exec("find ../SVRRoot/SVRBusiness-tests/test/test_data -name etalon_response_vector_output.txt");
    erase_after(etalon_response_vector_output_path, '\n');
}

void compare_datarow_containers(svr::datamodel::DataRow::container const & lhs, svr::datamodel::DataRow::container const & rhs)
{
    ASSERT_EQ(lhs.size(), rhs.size());

    static double const epsilon = std::numeric_limits<float>::epsilon();

    for(auto li = lhs.begin(), ri = rhs.begin(); li != lhs.end(); ++li, ++ri)
    {
        ASSERT_EQ(li->get()->get_value_time(), ri->get()->get_value_time());
        ASSERT_GT(epsilon, fabs(li->get()->get_tick_volume() - ri->get()->get_tick_volume() ) );
        ASSERT_EQ(li->get()->get_values().size(), ri->get()->get_values().size());

        for(auto ldi = li->get()->get_values().begin(), rdi = ri->get()->get_values().begin(); ldi != li->get()->get_values().end(); ++ldi, ++rdi)
            ASSERT_GT(epsilon, fabs(*ldi - *rdi ) );
    }
}

void save_datarow_containers(std::deque<datamodel::DeconQueue_ptr> &decon_queues, std::string &path)
{
    std::ofstream ofs(path);
    for (auto it = decon_queues.begin(); it != decon_queues.end(); ++it)
    {
        auto queue = *it;
        ofs << queue->data_to_string(66000);
        ofs << std::endl << new_datarow_container << std::endl;
    }
}

std::deque<svr::datamodel::DataRow::container> load_datarow_containers(const std::string &file_path)
{
    std::ifstream ifile(file_path.c_str());
    if(ifile.bad())
        throw std::runtime_error("load_datarow_containers: Error while openinig the file");

    std::deque<svr::datamodel::DataRow::container> result;
    svr::datamodel::DataRow::container current_container;

    size_t valuation_counter = -1;
    static size_t const valuate_every = 100;

    while(!ifile.eof())
    {
        try
        {
            ++valuation_counter;

            std::string line;
            std::getline(ifile, line);
            if(line.empty())
                continue;

            if(line == new_datarow_container)
            {
                result.push_back(current_container);
                current_container.clear();
                continue;
            }

            std::istringstream istr(line);
            std::string str_tmp1, str_tmp2;
            double dbl_tmp;

            //Reading value_time

            istr >> str_tmp1;
            if( ( valuation_counter % valuate_every ) == 0 && str_tmp1 != "ValueTime:")
                throw std::runtime_error("load_datarow_containers: Wrong file format");

            istr >> str_tmp1 >> str_tmp2;

            if(!str_tmp2.empty())
                str_tmp2.resize(str_tmp2.size()-1);

            svr::datamodel::DataRow_ptr row = std::make_shared<svr::datamodel::DataRow>();
            row->set_value_time(bpt::time_from_string(str_tmp1 + " " + str_tmp2));

            //Skipping update_time

            istr >> str_tmp1;
            if( ( valuation_counter % valuate_every ) == 0 && str_tmp1 != "UpdateTime:")
                throw std::runtime_error("load_datarow_containers: Wrong file format");

            istr >> str_tmp1 >> str_tmp2;

            //Reading data

            istr >> dbl_tmp;
            row->set_tick_volume(dbl_tmp);

            istr >> str_tmp1;
            if( ( valuation_counter % valuate_every ) == 0 && str_tmp1 != ",")
                throw std::runtime_error("load_datarow_containers: Wrong file format");

            std::vector<double> values;

            while(!istr.eof())
            {
                istr >> dbl_tmp;
                values.push_back(dbl_tmp);

                istr >> str_tmp1;
                if( ( valuation_counter % valuate_every ) == 0 && str_tmp1 != ",")
                    throw std::runtime_error("load_datarow_containers: Wrong file format");
            }
            row->set_values(values);

            current_container.push_back(row);
        }
        catch (...)
        {
            std::cerr << "load_datarow_containers: exception thrown while processing line #" << valuation_counter << " of file " << file_path << "\n";
            std::rethrow_exception(std::current_exception());
        }
    }
    return result;

}
