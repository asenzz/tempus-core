#include "include/DaoTestFixture.h"
#include <memory>

#include <model/User.hpp>
#include <model/Request.hpp>
#include <model/Dataset.hpp>
#include <model/InputQueue.hpp>

using namespace svr;
using datamodel::MultivalRequest;

TEST_F(DaoTestFixture, BasicWhiteboxTest)
{

    tdb.prepareSvrConfig(tdb.TestDbUserName, tdb.dao_type, 1);
    // Latest data in the IQ    open high low close
    // 2017-05-05 06:20:00     2017-05-05 13:20:55     20       1.09573000000000        1.09574000000000        1.09564000000000        1.09571000000000

    std::string const default_user_name("svrwave");
    bigint const default_dataset_id = 100;
    bpt::ptime const nw = bpt::second_clock::local_time();
    bpt::ptime const default_request_time = bpt::from_iso_string("20170503T1808000");
    bpt::time_duration const default_resolution = bpt::seconds(60);

    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full || DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::first_half)
    {

        datamodel::MultivalRequest_ptr request = datamodel::MultivalRequest_ptr( new
            MultivalRequest(bigint(0), default_user_name, default_dataset_id, nw
            , default_request_time, default_request_time + default_resolution * 15, default_resolution.total_seconds()
            , "{open,high,low,close}")
        );

        aci.request_service.save(request);

        aci.flush_dao_buffers();
    }

    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full) {
        EXPECT_TRUE(tdb.run_daemon());
    }

    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full || DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::second_half)
    {
        auto results = aci.request_service.get_multival_results
        (
              default_user_name
            , default_dataset_id
            , default_request_time
            , default_request_time + default_resolution * 15
            , default_resolution.total_seconds()
        );

        ASSERT_NE(0UL, results.size());

        const auto& fis = bpt::from_iso_string;

        typedef std::map<bpt::ptime, std::map<std::string, double>> result_map;

        // Predicted values as of May 10, 2017
        result_map const reference_data
        {
            {fis("20170503T180800"), {{"open", 1.08827}, {"high", 1.08832}, {"low", 1.08821}, {"close", 1.08829}}},
            {fis("20170503T180900"), {{"open", 1.08826}, {"high", 1.08846}, {"low", 1.08826}, {"close", 1.08846}}},
            {fis("20170503T181000"), {{"open", 1.08849}, {"high", 1.08851}, {"low", 1.08843}, {"close", 1.08843}}},
            {fis("20170503T181100"), {{"open", 1.08845}, {"high", 1.08856}, {"low", 1.08845}, {"close", 1.08856}}},
            {fis("20170503T181200"), {{"open", 1.08859}, {"high", 1.08864}, {"low", 1.08859}, {"close", 1.08864}}},
            {fis("20170503T181300"), {{"open", 1.08864}, {"high", 1.08868}, {"low", 1.08864}, {"close", 1.08868}}},
            {fis("20170503T181400"), {{"open", 1.0887},  {"high", 1.08872}, {"low", 1.0887},  {"close", 1.08872}}},
            {fis("20170503T181500"), {{"open", 1.08874}, {"high", 1.08874}, {"low", 1.08848}, {"close", 1.08856}}},
            {fis("20170503T181600"), {{"open", 1.08859}, {"high", 1.08859}, {"low", 1.08859}, {"close", 1.08859}}},
            {fis("20170503T181700"), {{"open", 1.08861}, {"high", 1.08861}, {"low", 1.0886},  {"close", 1.0886}}},
            {fis("20170503T181800"), {{"open", 1.08861}, {"high", 1.08869}, {"low", 1.08861}, {"close", 1.08869}}},
            {fis("20170503T181900"), {{"open", 1.08871}, {"high", 1.08871}, {"low", 1.08871}, {"close", 1.08871}}},
            {fis("20170503T182000"), {{"open", 1.08873}, {"high", 1.08875}, {"low", 1.08873}, {"close", 1.08875}}},
            {fis("20170503T182100"), {{"open", 1.08872}, {"high", 1.08872}, {"low", 1.08871}, {"close", 1.08871}}},
            {fis("20170503T182200"), {{"open", 1.08869}, {"high", 1.08869}, {"low", 1.08865}, {"close", 1.08866}}}
        };

	result_map const expected_forecast_data  // Should be recomputed in case of change of algorithm!
        {
            {fis("20170503T180800"), {{"open", 1.08842997848231}, {"high", 1.08843996581386}, {"low", 1.08841997578984},
 {"close", 1.08841997157165}}},
            {fis("20170503T180900"), {{"open", 1.08844565621272}, {"high", 1.0884503323314}, {"low", 1.08844316296392}, 
{"close", 1.08847828683604}}},
            {fis("20170503T181000"), {{"open", 1.08848200064562}, {"high", 1.08850258785858}, {"low",1.08850313895059}, 
{"close", 1.08846472270109}}},
            {fis("20170503T181100"), {{"open", 1.08858035540768}, {"high", 1.08857788055797}, {"low", 1.08857188296975}, 
{"close", 1.08859721453949}}},
            {fis("20170503T181200"), {{"open", 1.08864380047681}, {"high", 1.08865523378237}, {"low", 1.08863073766052}, 
{"close", 1.08866800502746}}},
            {fis("20170503T181300"), {{"open", 1.08869196440743}, {"high", 1.0887013612698}, {"low", 1.08868782780928}, 
{"close", 1.08871047575622}}},
            {fis("20170503T181400"), {{"open", 1.08872727774431}, {"high", 1.08873923080705}, {"low", 1.08870326466312}, 
 {"close", 1.08875204838602}}},
            {fis("20170503T181500"), {{"open", 1.08859744847954}, {"high", 1.08861243409821}, {"low", 1.08859272290097}, 
{"close",1.08861190943045}}},
            {fis("20170503T181600"), {{"open", 1.08864364291444}, {"high", 1.08864597340924}, {"low",1.08863074079777}, 
{"close", 1.08866111549107}}},
            {fis("20170503T181700"), {{"open", 1.08865026872777}, {"high", 1.08865344793065}, {"low",1.08863714569918},  
{"close", 1.08867302561911}}},
            {fis("20170503T181800"), {{"open", 1.08872402055873}, {"high", 1.0887243139895}, {"low",1.08870885568022}, 
{"close", 1.08874675985403}}},
            {fis("20170503T181900"), {{"open", 1.08873961617096}, {"high", 1.08874666684142}, {"low",1.08872109148164}, 
{"close", 1.08875939609418}}},
            {fis("20170503T182000"), {{"open", 1.08878168066868}, {"high", 1.08877920803634}, {"low",1.08876020353101}, 
{"close", 1.0887938698591}}},
            {fis("20170503T182100"), {{"open", 1.08874466655044}, {"high", 1.08875127092474}, {"low",1.08873508927205}, 
{"close", 1.08875418388603}}},
            {fis("20170503T182200"), {{"open", 1.08871391308303}, {"high", 1.08870718860254}, {"low",1.08869754937872}, 
{"close", 1.08871084943184}}}
        };




        result_map actual_data;

        for(auto const & result : results)
            actual_data[result->value_time][result->value_column] = result->value;

        for(auto &actd : actual_data)
        {
            std::cout << bpt::to_simple_string(actd.first);
            for(auto vls : actd.second)
                std::cout << "\t" << vls.first << " " << std::setprecision (15) << vls.second;
            std::cout << "\n";
        }

        //ASSERT_EQ(reference_data.size(), actual_data.size());

        double max_diff = 0;
        double rmse = 0;
        double me = 0;
        int num_predictions = 0;

        for(auto const & ref_pairs : reference_data)
            for(auto const & ref_val : ref_pairs.second)
            {
                auto act_iter = actual_data[ref_pairs.first].find(ref_val.first);
                ASSERT_NE(act_iter, actual_data[ref_pairs.first].end());

                double diff = fabs(ref_val.second - act_iter->second) / ref_val.second;
                rmse += diff*diff;
                me += diff;
                ++num_predictions;

                max_diff = std::max(diff, max_diff);
            }
        rmse = sqrt(rmse / num_predictions);
        me /= num_predictions;
	double variation_from_expected_forecast_data = 0.;
        
	for(auto const & ref_pairs : expected_forecast_data )
            for(auto const & ref_val : ref_pairs.second)
            {
                auto act_iter = actual_data[ref_pairs.first].find(ref_val.first);
                ASSERT_NE(act_iter, actual_data[ref_pairs.first].end());

                double diff = fabs(ref_val.second - act_iter->second) / ref_val.second;
                variation_from_expected_forecast_data += diff*diff;
            }
        variation_from_expected_forecast_data = sqrt(variation_from_expected_forecast_data / num_predictions);


        std::cout << "BasicWhiteboxTest:: max value diff: " << max_diff << std::endl;
        std::cout << "BasicWhiteboxTest:: rmse: " << rmse << std::endl;
        std::cout << "BasicWhiteboxTest:: variation_from_expected_forecast_data: " << variation_from_expected_forecast_data << std::endl;
        std::cout << "BasicWhiteboxTest:: me (bias): " << me << std::endl;

        ASSERT_LT(max_diff, 0.1L);
    }
}

//TEST_F(DaoTestFixture, modwt_BasicWhiteboxTest){
//
//     tdb.prepareSvrConfig(tdb.TestDbUserName, tdb.dao_type, 1);
//    // Latest data in the IQ    open high low close
//    // 2017-05-05 06:20:00     2017-05-05 13:20:55     20       1.09573000000000        1.09574000000000        1.09564000000000        1.09571000000000
//
//    std::string const default_user_name("svrwave");
//    bigint const default_dataset_id = 100;
//    bpt::ptime const nw = bpt::second_clock::local_time();
//    bpt::ptime const default_request_time = bpt::from_iso_string("20170503T1808000");
//    bpt::time_duration const default_resolution = bpt::seconds(60);
//    u_int64_t default_decremental_distance = 2000;
//
//    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full || DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::first_half)
//    {
//        std::map<std::pair<std::string, std::string>, std::vector<datamodel::SVRParameters_ptr>> parameters_pairs
//                = aci.svr_parameters_service.get_all_by_dataset_id(default_dataset_id);
//        for(auto parameters_pair : parameters_pairs)
//            for(auto parameters : parameters_pair.second)
//            {
//                parameters->set_svr_decremental_distance(default_decremental_distance);
//                aci.svr_parameters_service.save(parameters);
//            }
//
//        datamodel::MultivalRequest_ptr request = datamodel::MultivalRequest_ptr(
//           new MultivalRequest( bigint(0),
//                                default_user_name,
//                                default_dataset_id,
//                                nw,
//                                default_request_time,
//                                default_request_time + default_resolution * 15,
//                                default_resolution.total_seconds(),
//                                "{open,high,low,close}" )
//        );
//
//        aci.request_service.save(request);
//
//        aci.flush_dao_buffers();
//    }
//
//    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full)
//        EXPECT_TRUE(tdb.run_daemon());
//
//    if(DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::full || DaoTestFixture::DoTestPart == DaoTestFixture::test_completeness::second_half)
//    {
//        std::vector<datamodel::MultivalResponse_ptr>  results = aci.request_service.get_multival_results
//        (
//             default_user_name,
//             default_dataset_id,
//             default_request_time,
//             default_request_time + default_resolution * 15,
//             default_resolution.total_seconds()
//        );
//
//        ASSERT_NE(0UL, results.size());
//
//}
//}
