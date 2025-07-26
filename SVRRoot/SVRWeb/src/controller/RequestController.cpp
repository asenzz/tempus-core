#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <regex>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <unordered_set>
#include <cppcms/json.h>
#include "controller/RequestController.hpp"
#include "controller/MainController.hpp"
#include "appcontext.hpp"
#include "RequestService.hpp"
#include "DAO/RequestDAO.hpp"
#include "model/Request.hpp"
#include "model/Dataset.hpp"
#include "util/validation_utils.hpp"

namespace svr {
namespace web {
void RequestView::makeMultivalRequest(cppcms::json::object data)
{
    constexpr char user[] = DEFAULT_WEB_USER;
    // const auto user = session()["user"]; // TODO Fix a bug with CPPCMS session
    const auto ival_cols = data.find("value_columns");
    if (ival_cols == data.cend()) {
        constexpr char error[] = "No value_columns specified!";
        return_error(error);
        return;
    }
    static const std::regex comma_regex(",");
    const std::string value_columns = "{\"" + std::regex_replace(ival_cols->second.str(), comma_regex, "\",\"") + "\"}";
    REJECT_EMPTY_1(data["dataset"].str());
    const auto &dataset_id_str = data["dataset"].str();
#ifdef VERIFY_REQUESTS
    const auto dataset_id = boost::lexical_cast<bigint>(dataset_id_str);
    if (!APP.dataset_service.exists(dataset_id)) {
        const string error = "No dataset with id " + data["dataset"].str();
        LOG4_ERROR(error);
        return_error(error);
        return;
    }
#endif
    REJECT_EMPTY_1(data["value_time_start"].str());
    REJECT_EMPTY_1(data["value_time_end"].str());

    const auto &value_time_start_str = data["value_time_start"].str();
    const auto &value_time_end_str = data["value_time_end"].str();
    const auto &resolution_str = data["resolution"].str();
    const auto request_id = APP.request_service.make_request(user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str, value_columns);
    if (request_id < 1) {
        const std::string msg = common::formatter() << "Failed making forecast request " << user << ", " << dataset_id_str << ", " << value_time_start_str << ", " << value_time_end_str
                                << ", " << resolution_str << " " << value_columns;
        LOG4_ERROR(msg);
        return_error(msg);
        return;
    }

    json::value response;
    response["request_id"] = request_id;
    return_result(response);
}

void RequestView::getMultivalResults(json::object data)
{
    REJECT_EMPTY_1(data["resolution"].str());
    REJECT_EMPTY_1(data["dataset"].str());
    REJECT_EMPTY_1(data["value_time_start"].str());
    REJECT_EMPTY_1(data["value_time_end"].str());

    json::value response;
    constexpr char user[] = DEFAULT_WEB_USER;
    // const string user = session()["user"];

#ifdef VERIFY_REQUESTS

    const auto resolution = seconds(boost::lexical_cast<uint32_t>(data["resolution"].str()));
    const auto value_time_start = bpt::time_from_string(data["value_time_start"].str());
    const auto value_time_end = bpt::time_from_string(data["value_time_end"].str());

    REJECT_NOT_A_DATE_TIME(value_time_start);
    REJECT_NOT_A_DATE_TIME(value_time_end);

    const auto dataset_id = boost::lexical_cast<bigint>(data["dataset"].str());

    const auto request = AppContext::get().request_service.get_multival_request(user, dataset_id, value_time_start, value_time_end);
    if (!request) {
        const std::string error = "No such request has been submitted: user: " + std::string(user) + " dataset_id: " + to_string(dataset_id)
                                  + " start time: " + to_simple_string(value_time_start) + " end time: " + to_simple_string(value_time_end);
        LOG4_ERROR(error);
        return_error(error);
        return;
    }

    const auto request_columns = from_sql_array(request->value_columns);
    const uint32_t expected_rows_ct = request_columns.size() * ((value_time_end - value_time_start) / resolution);
    const auto responses = AppContext::get().request_service.get_multival_results(
            user, dataset_id, value_time_start, value_time_end, resolution.total_seconds());
    if (responses.empty() || expected_rows_ct > responses.size()) {
        const std::string error = common::formatter() << "Couldn't find enough responses " << responses.size() << " matching the requested " << expected_rows_ct << " for " << request->to_string();
        LOG4_WARN(error);
        return_error(error);
        return;
    }

#else

    const auto &dataset_id_str = data["dataset"].str();
    const auto &resolution_str = data["resolution"].str();
    const auto &value_time_start_str = data["value_time_start"].str();
    const auto &value_time_end_str = data["value_time_end"].str();
    const auto responses = APP.request_service.get_multival_results(user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str);
    if (responses.empty()) {
        const std::string error =
                common::formatter() << "Couldn't find any responses for user " << user << ", dataset " << dataset_id_str << ", starting " << value_time_start_str <<
                ", until " << value_time_end_str << ", resolution " << resolution_str;
        LOG4_WARN(error);
        return_error(error);
        return;
    }

#endif

    json::array json_array;
    for (const auto &aresponse: responses) {
        auto it_jsonarray = std::find_if(C_default_exec_policy, json_array.begin(), json_array.end(), [&aresponse](const auto &json_row) {
            return bpt::time_from_string(json_row["tm"].str()) == aresponse->value_time;
        });
        if (it_jsonarray == json_array.cend()) {
            json::value new_row;
            new_row.set("tm", common::to_mql_date(aresponse->value_time));
            LOG4_TRACE("New row at time " << aresponse->value_time);
            it_jsonarray = json_array.insert(json_array.cend(), new_row);
        }
        it_jsonarray->set(aresponse->value_column, common::to_string(aresponse->value));

        if (common::AppConfig::S_log_threshold > boost::log::trivial::severity_level::trace) continue;
        std::stringstream s;
        it_jsonarray->save(s, json::compact);
        LOG4_TRACE("Added column " << aresponse->value_column << ", value " << aresponse->value << " to row " << s.str());
    }

    std::atomic<bool> json_array_correct = true;

#ifdef VERIFY_REQUESTS

    OMP_FOR(json_array.size())
    for (const auto &json_object: json_array) {
        if (json_object.array().size() != request_columns.size() + 1) {
            json_array_correct = false;
            LOG4_ERROR("Response array has incorrect number of columns " << json_object.array().size() << ", should be " << request_columns.size() + 1);
            continue;
        }
        for (const auto &column: request_columns) {
            if (json_object[column].is_null() || json_object[column].is_undefined()) {
                json_array_correct = false;
                LOG4_ERROR("Response array has no value for column " << column);
                continue;
            }
        }
    }

#endif

    if (json_array.empty() || !json_array_correct) {
        constexpr char error[] = "No good response is ready yet.";
        LOG4_WARN(error);
        return_error(error);
        return;
    }

    LOG4_TRACE("Returning " << json_array.size() << " rows.");
    response.array(json_array);
    return_result(response);
}

void RequestView::main(std::string string)
{
    json_rpc_server::main(string);
}
}
}
