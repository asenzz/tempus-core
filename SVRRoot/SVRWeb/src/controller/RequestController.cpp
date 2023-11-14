#include <cppcms/json.h>
#include <cppcms/session_interface.h>
#include "controller/RequestController.hpp"
#include "controller/MainController.hpp"
#include "appcontext.hpp"
#include "DAO/RequestDAO.hpp"

#include <model/Request.hpp>
#include <model/Dataset.hpp>

#include <util/ValidationUtils.hpp>

using namespace svr::datamodel;
using namespace svr::context;
using namespace svr::common;
using namespace bpt;
using namespace std;
using namespace cppcms;

namespace svr {
namespace web {

void RequestView::makeMultivalRequest(cppcms::json::object data)
{
    reject_empty(data["dataset"].str());
    reject_empty(data["value_time_start"].str());
    reject_empty(data["value_time_end"].str());
    //if "value_columns" is empty, all of HLOC are requested

    ptime valueTimeStart = bpt::time_from_string(data["value_time_start"].str());
    ptime valueTimeEnd = bpt::time_from_string(data["value_time_end"].str());
    reject_not_a_date_time(valueTimeStart);
    reject_not_a_date_time(valueTimeEnd);
    string user = DEFAULT_WEB_USER; // session()["user"];
    bigint dataset_id = boost::lexical_cast<bigint>(data["dataset"].str());
    auto const ivalCols = data.find("value_columns");
    std::string vctemp = ivalCols == data.end() ? std::string() : ivalCols->second.str();
    string const valueColumns = std::string("{") + (vctemp.empty() ? "high,low,open,close" : vctemp) + "}";

    Dataset_ptr dataset = AppContext::get_instance().dataset_service.get(dataset_id);
    if (!dataset) {
        string error = "No dataset with id " + data["dataset"].str();
        LOG4_ERROR(error);
        return_error(error);
        return;
    }

    time_duration resolution = dataset->get_input_queue()->get_resolution();
    MultivalRequest_ptr request = AppContext::get_instance().request_service.get_multival_request(user, dataset_id, valueTimeStart, valueTimeEnd, resolution.total_seconds(), valueColumns);
    if (request.get() != nullptr) {
        string msg = "Cannot make forecast request because it already exists! [" + request->to_string() + "]";
        LOG4_ERROR(msg);
        return_error(msg);
        return;
    }

    request = std::make_shared<MultivalRequest>(0, user, dataset_id, second_clock::local_time(), valueTimeStart, valueTimeEnd, resolution.total_seconds(), valueColumns);
    if (!AppContext::get_instance().request_service.save(request)) {
        string error = "Cannot create forecast request!";
        LOG4_ERROR(error);
        return_error(error);
        return;
    }

    json::value response;

    response["request_id"] = request->get_id();
    return_result(response);
}

void RequestView::getMultivalResults(json::object data)
{
    reject_empty(data["resolution"].str());
    reject_empty(data["dataset"].str());
    reject_empty(data["value_time_start"].str());
    reject_empty(data["value_time_end"].str());

    json::value response;

    time_duration resolution = seconds(boost::lexical_cast<long>(data["resolution"].str()));
    ptime value_time_start = bpt::time_from_string(data["value_time_start"].str());
    ptime value_time_end = bpt::time_from_string(data["value_time_end"].str());
    reject_not_a_date_time(value_time_start);
    reject_not_a_date_time(value_time_end);
    string user = DEFAULT_WEB_USER; // session()["user"];
    bigint dataset_id = boost::lexical_cast<bigint>(data["dataset"].str());
    auto const iintervals = data.find("intervals");
    size_t const intervals = iintervals == data.end() ? 1 : boost::lexical_cast<size_t>(iintervals->second.str());
    reject_not_positive(intervals);

    MultivalRequest_ptr request = AppContext::get_instance().request_service.get_multival_request(user, dataset_id, value_time_start, value_time_end);
    if (!request) {
        std::string error = "No such request has been submitted: user: " + user + " dataset_id: " + to_string(dataset_id)
                            + " start time: " + to_simple_string(value_time_start) + " end time: " + to_simple_string(value_time_end);
        LOG4_ERROR(error);
        return_error(error);
        return;
    }

    std::vector<MultivalResponse_ptr> responses = AppContext::get_instance().request_service.get_multival_results(
            user, dataset_id, value_time_start, value_time_end, resolution.total_seconds());
    if (responses.empty()) {
        string error = "Response is not ready yet.";
        LOG4_WARN(error);
        return_error(error);
        return;
    }

    struct mt4bar
    {
        std::map<std::string, double> values;
        std::set<std::string> columns;

        bool good() const
        {
            if (columns.empty())
                return false;
            if (columns.size() != values.size())
                return false;

            std::set<std::string>::const_iterator icol = columns.begin();
            std::map<std::string, double>::const_iterator ival = values.begin();

            for (; icol != columns.end(); ++icol, ++ival)
                if (*icol != ival->first || fabs(ival->second) < std::numeric_limits<double>::epsilon())
                    return false;
            return true;
        }
    };

    using mt4bars = std::map<bpt::ptime, mt4bar>;
    mt4bars bars;

    json::array tmp_array;

    for (auto &&aresponse: responses) {
        auto &bar = bars[aresponse->value_time];
        bar.values[aresponse->value_column] = aresponse->value;
    }

    for (auto me: bars) {
        for (auto const &clm: from_sql_array(request->value_columns))
            me.second.columns.insert(clm);

        if (!me.second.good())
            continue;

        json::value tmp;

        tmp["tm"] = to_mt4_date(me.first);

        for (std::map<std::string, double>::const_iterator iter = me.second.values.begin(); iter != me.second.values.end(); ++iter)
            tmp[std::string(iter->first.begin(), iter->first.begin() + 1)] = iter->second;

        tmp_array.push_back(tmp);
    }

    if (tmp_array.empty()) {
        string error = "Response is not ready yet.";
        LOG4_WARN(error);
        return_error(error);
        return;
    }

    response.array(tmp_array);

    return_result(response);
}

void RequestView::main(std::string string)
{
    json_rpc_server::main(string);
}

}
}

