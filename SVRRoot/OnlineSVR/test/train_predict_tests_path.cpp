/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "kernel_basic_integration_test.hpp"
#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif
#include <atomic>
#include <rapidcsv.h>
#include <arpa/inet.h>
#include <chrono>
#include <regex>

#define DEVICE_ID_LEN 64
#define VALIDATION_LENGTH 7000

using namespace svr;

arma::rowvec string_to_features(const std::string &str, const size_t output_row_size)
{
    arma::rowvec result(output_row_size, arma::fill::zeros);
    for (size_t i = 0; i < output_row_size && i < str.size(); ++i) result[i] = double(str[i]);
    return result;
}


arma::colvec bool_strings(const std::vector<std::string> &fields_str)
{
    static const std::regex true_pattern("true", std::regex_constants::icase);
    static const std::regex false_pattern("false", std::regex_constants::icase);
    arma::colvec res(fields_str.size());
#pragma omp parallel for
    for (size_t i = 0; i < fields_str.size(); ++i)
        if (std::regex_search(fields_str[i], true_pattern))
            res[i] = 1;
        else if (std::regex_search(fields_str[i], false_pattern))
            res[i] = 0;
        else
            res[i] = std::numeric_limits<double>::quiet_NaN();
    return res;
}


arma::colvec hash_strings(const std::vector<std::string> &fields_str)
{
    static const std::hash<std::string> str_hasher;
    arma::colvec res(fields_str.size());
#pragma omp parallel for
    for (size_t i = 0; i < fields_str.size(); ++i)
        res[i] = fields_str[i].empty() ? std::numeric_limits<double>::quiet_NaN() : double(str_hasher(fields_str[i]));
    return res;
}

arma::colvec date_strings(const std::vector<std::string> &dates_str)
{
    arma::colvec res(dates_str.size());
#pragma omp parallel for
    for (size_t i = 0; i < dates_str.size(); ++i) {
#if 0
        click_date[i] = click_date_str[i].empty() ? std::numeric_limits<double>::quiet_NaN() : double(str_hasher(click_date_str[i]));
        //click_date.row(i) = string_to_features(click_date_str[i], 19);
#else
        std::tm row_tm = {};
        std::stringstream(dates_str[i]) >> std::get_time(&row_tm, "%Y-%b-%d %H:%M:%S");
        res[i] = double(std::mktime(&row_tm));
#endif
    }
    return res;
}

typedef struct sockaddr_in6 sockaddr_in6_t;

arma::colvec ip_strings(const std::vector<std::string> &ips_str)
{
    arma::colvec res(ips_str.size());
#pragma omp parallel for
    for (size_t i = 0; i < ips_str.size(); ++i) {
#if 0
        //        ip_addresses.row(i) = string_to_features(ip_addresses_str[i], 39);
        ip_addresses[i] = ip_addresses_str[i].empty() ? std::numeric_limits<double>::quiet_NaN() : double(str_hasher(ip_addresses_str[i]));
#else
        struct sockaddr_in tmp_ipv4;
        struct sockaddr_in6 tmp_ipv6;
        if (inet_pton(AF_INET, ips_str[i].c_str(), (void *) &tmp_ipv4.sin_addr)) {
            res[i] = double(tmp_ipv4.sin_addr.s_addr);
        } else if (inet_pton(AF_INET6, ips_str[i].c_str(), (void *) &tmp_ipv6.sin6_addr)) {
            const static struct in6_addr *in6 = &tmp_ipv6.sin6_addr;
            res[i] = 0;
            for (size_t i = 0; i < 16; ++i) res[i] += double(in6->s6_addr[i] << i * sizeof(in6->s6_addr[i]) * 8);
        } else
            res[i] = std::numeric_limits<double>::quiet_NaN();
#endif
    }
    return res;
}

TEST(path_train_predict, basic_integration)
{
    LOG4_BEGIN();
    svr::IKernel<double>::IKernelInit();

    OnlineMIMOSVR_ptr model, model2;
    SVRParameters_ptr p_svr_parameters = std::make_shared<SVRParameters>(0, 0, "test", "test", 0, 0, 0, 1e5, 0, 0, 0, CHUNK_SIZE, 1, kernel_type_e::PATH, 1);
    rapidcsv::Document doc(
            "/var/tmp/devicetracker_export_2023_04_27-08_54_03.csv", // "/var/tmp/processed_csv_devicetracker.csv",
            rapidcsv::LabelParams(),
            rapidcsv::SeparatorParams(',', true),
            rapidcsv::ConverterParams(true, std::numeric_limits<double>::quiet_NaN(), svr::common::C_int_nan));

    std::shared_ptr<arma::mat> p_labels = std::make_shared<arma::mat>(hash_strings(doc.GetColumn<std::string>("userID")));

    const arma::colvec ip_addresses = ip_strings(doc.GetColumn<std::string>("IP Address"));
    const arma::colvec devices_id = hash_strings(doc.GetColumn<std::string>("Device ID"));
    const arma::colvec country = hash_strings(doc.GetColumn<std::string>("Country"));
    const arma::colvec fraud_score = arma::conv_to<arma::colvec>::from(doc.GetColumn<double>("Fraud Score"));
    const arma::colvec operating_system = hash_strings(doc.GetColumn<std::string>("Operating System"));
    const arma::colvec browser = hash_strings(doc.GetColumn<std::string>("Browser"));
    const arma::colvec proxy = bool_strings(doc.GetColumn<std::string>("Proxy"));
    const arma::colvec isp = hash_strings(doc.GetColumn<std::string>("ISP"));
    const arma::colvec connection_type = hash_strings(doc.GetColumn<std::string>("Connection Type"));
    const arma::colvec recent_abuse = bool_strings(doc.GetColumn<std::string>("Recent Abuse"));
    const arma::colvec bot_status = bool_strings(doc.GetColumn<std::string>("Bot Status"));
    const arma::colvec click_date = date_strings(doc.GetColumn<std::string>("Click Date"));
    const arma::colvec fraud_reasons = hash_strings(doc.GetColumn<std::string>("Fraud Reasons"));
    const arma::colvec request_id = hash_strings(doc.GetColumn<std::string>("Request ID"));
    const arma::colvec active_vpn = hash_strings(doc.GetColumn<std::string>("Active VPN"));
    const arma::colvec active_tor = hash_strings(doc.GetColumn<std::string>("Active TOR"));
    const arma::colvec risk_score = doc.GetColumn<double>("Risk Score");
    const arma::colvec guid = hash_strings(doc.GetColumn<std::string>("GUID"));
    const arma::colvec region = hash_strings(doc.GetColumn<std::string>("Region"));
    const arma::colvec city = hash_strings(doc.GetColumn<std::string>("City"));
    const arma::colvec page_views = doc.GetColumn<double>("Page Views");
    const arma::colvec unique = hash_strings(doc.GetColumn<std::string>("Unique"));
    const arma::colvec brand = hash_strings(doc.GetColumn<std::string>("Brand"));
    const arma::colvec phone_model = hash_strings(doc.GetColumn<std::string>("Model"));
    const arma::colvec mobile = hash_strings(doc.GetColumn<std::string>("Mobile"));
    const arma::colvec vpn = bool_strings(doc.GetColumn<std::string>("VPN"));
    const arma::colvec tor = bool_strings(doc.GetColumn<std::string>("TOR"));
    const arma::colvec abuse_velocity = hash_strings(doc.GetColumn<std::string>("Abuse Velocity"));
    const arma::colvec user_agent = hash_strings(doc.GetColumn<std::string>("User Agent"));
    const arma::colvec last_seen = date_strings(doc.GetColumn<std::string>("Last Seen"));
    const arma::colvec fraudulent_behaviour = doc.GetColumn<double>("Fraudulent Behavior");
    const arma::colvec leaked_user_data = doc.GetColumn<double>("Leaked User Data");

    LOG4_DEBUG(
            "Got matrixes of size " << arma::size(*p_labels) << " " << arma::size(ip_addresses) << " " << arma::size(devices_id) << " " << arma::size(country) << " " << arma::size(operating_system) << " " << arma::size(click_date) << " " <<
            arma::size(browser) << " " << arma::size(proxy) << " " << arma::size(isp) << " " << arma::size(connection_type) << " " << arma::size(recent_abuse) << " " << arma::size(bot_status) << " " << arma::size(click_date));


    std::shared_ptr<arma::mat> p_features = std::make_shared<arma::mat>(
         svr::common::join_rows<double>(
                 31, &ip_addresses, &devices_id, &country, &fraud_score, &operating_system, &browser, &proxy, &isp, &connection_type, &recent_abuse, &bot_status, &click_date, &fraud_reasons, &request_id,
                 &active_vpn, &active_tor, &risk_score, &guid, &region, &city, &page_views, &unique, &brand, &phone_model, &mobile, &vpn, &tor, &abuse_velocity, &user_agent, &last_seen, &fraudulent_behaviour, &leaked_user_data));
    std::vector<size_t> invalid_rows;
    for (size_t i = 0; i < p_features->n_rows; ++i)
        if (p_features->row(i).has_nan() || p_labels->row(i).has_nan())
            invalid_rows.emplace_back(i);

    p_features->shed_rows(arma::conv_to<arma::uvec>::from(invalid_rows));
    p_labels->shed_rows(arma::conv_to<arma::uvec>::from(invalid_rows));
    p_features->save("/var/tmp/iqpay_features.csv", arma::csv_ascii);
    p_labels->save("/var/tmp/iqpay_labels.csv", arma::csv_ascii);
    const auto labels_scaling_factor = arma::median(arma::abs(arma::vectorise(*p_labels))) / svr::common::C_input_obseg_labels;
    LOG4_DEBUG("Labels " << arma::size(*p_labels) << ", labels scaling factor " << labels_scaling_factor << ", invalid rows " << invalid_rows.size());
    *p_labels /= labels_scaling_factor;
    const arma::rowvec col_divisor = arma::mean(arma::abs(*p_features)) / svr::common::C_input_obseg_features;
    for (size_t i = 0; i < p_features->n_cols; ++i) p_features->col(i) /= arma::as_scalar(col_divisor[i]);
    LOG4_DEBUG("Got matrixes " << arma::size(*p_features) << ", " << arma::size(*p_labels) << ", ");
    if (p_labels->n_rows != p_features->n_rows)
        LOG4_THROW("Sizes not appropriate, labels " << arma::size(*p_labels) << ", features " << arma::size(*p_features));
    const size_t half_train_len = (p_features->n_rows - VALIDATION_LENGTH) / 2;
    PROFILE_EXEC_TIME(model = std::make_shared<svr::OnlineMIMOSVR>(
            p_svr_parameters,
            std::make_shared<arma::mat>(p_features->rows(0, half_train_len - 1)),
            std::make_shared<arma::mat>(p_labels->rows(0, half_train_len - 1))),
                      "Model cold start, training and tuning on real labels");
    arma::mat predictions;
    PROFILE_EXEC_TIME(predictions = model->chunk_predict(p_features->rows(half_train_len, p_features->n_rows - VALIDATION_LENGTH - 1), bpt::special_values::not_special),
                      "Model predict");

    arma::mat residuals = p_labels->rows(half_train_len, p_features->n_rows - VALIDATION_LENGTH - 1) - predictions;
    const double res_scale_factor = arma::median(arma::abs(arma::vectorise(residuals))) / common::C_input_obseg_labels;
    residuals /= res_scale_factor;
    PROFILE_EXEC_TIME(model2 = std::make_shared<svr::OnlineMIMOSVR>(
            p_svr_parameters,
            std::make_shared<arma::mat>(p_features->rows(half_train_len, p_features->n_rows - VALIDATION_LENGTH - 1)),
            std::make_shared<arma::mat>(residuals)),
                      "Model cold start, training and tuning on residuals");
    arma::mat residual_predictions, final_predictions;
    PROFILE_EXEC_TIME(residual_predictions = model2->chunk_predict(p_features->rows(p_features->n_rows - VALIDATION_LENGTH - 1, p_features->n_rows - 1), bpt::special_values::not_special),
                      "Model predict residuals");
    residual_predictions *= res_scale_factor;
    PROFILE_EXEC_TIME(final_predictions = model->chunk_predict(p_features->rows(p_features->n_rows - VALIDATION_LENGTH - 1, p_features->n_rows - 1), bpt::special_values::not_special),
                      "Model predict final");

    final_predictions += residual_predictions;

    *p_labels *= labels_scaling_factor / svr::common::C_input_obseg_labels;
    final_predictions *= labels_scaling_factor / svr::common::C_input_obseg_labels;
    const auto meanabs_labels = common::meanabs<double>(*p_labels);
    const auto mae = common::meanabs<double>(p_labels->rows(p_labels->n_rows - VALIDATION_LENGTH - 1, p_labels->n_rows - 1) - final_predictions);
    const double mape = 100. * mae / meanabs_labels;
    for (size_t i = 0; i < final_predictions.n_rows; ++i) {
        static double cml_mae;
        cml_mae += std::abs<double>(p_labels->at(p_labels->n_rows - VALIDATION_LENGTH + i, 0) - final_predictions(i, 0));
        LOG4_DEBUG("Cumulative " << i << " MAE " << cml_mae / double(i + 1) << ", MAPE " << 100. * cml_mae / (meanabs_labels * double(i + 1)));
    }

    LOG4_INFO("Validation on last 1000 rows, MAE " << mae << ", MAPE " << mape << " pct.");
    LOG4_END();
}
