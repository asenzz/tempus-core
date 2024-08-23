#include <controller/InputQueueController.hpp>
#include <view/InputQueueView.hpp>
#include "appcontext.hpp"
#include "controller/MainController.hpp"
#include "util/validation_utils.hpp"

// TODO Implement time zone conversion at time_from_string() calls

using namespace svr::datamodel;
using namespace svr::context;
using namespace svr::common;
using namespace bpt;
using namespace std;


namespace svr {
namespace web {

InputQueueController::InputQueueController(cppcms::service &svc) : application(svc)
{
    dispatcher().assign("/", &InputQueueController::showAll, this);
    mapper().assign("showall", "/");

    dispatcher().assign("/show/(\\w+)", &InputQueueController::show, this, 1);
    mapper().assign("show", "/show/{1}");

    dispatcher().assign("/create", &InputQueueController::create, this);
    mapper().assign("create", "/create");

    attach(new InputQueueView(svc), "inputQueue_ajaxview", "/ajax{1}", "/ajax(/(.*))?", 1);
}

void InputQueueController::show(std::string queueName)
{
    session()["user"] = DEFAULT_WEB_USER;
    content::InputQueue model;
    model.pageTitle = "InputQueue Details";

    datamodel::InputQueue_ptr queue = AppContext::get_instance().input_queue_service.get_queue_metadata(queueName);

    if (queue.get() != nullptr && queue->get_owner_user_name() == DEFAULT_WEB_USER /* session()["user"] */ ) {
        model.object = queue;
    } else {
        model.pageError = "No queue named \"" + queueName + "\" was found!";
    }

    render("ShowInputQueue", model);
}

void InputQueueController::showAll()
{
    content::Main main;
    main.pageTitle = "Input Queues";
    render("InputQueues", main);
}

void InputQueueController::create()
{

    if (request().request_method() == "POST") {
        handle_create_post();
    } else {
        handle_create_get();
    }
}

void InputQueueView::getAllInputQueues()
{
    session()["user"] = DEFAULT_WEB_USER;
    std::vector<datamodel::InputQueue_ptr> queue_vector;
    const auto queues = APP.input_queue_service.get_all_user_queues(DEFAULT_WEB_USER /* session()["user"] */ );
    std::copy(C_default_exec_policy, queues.begin(), queues.end(), std::back_inserter(queue_vector));
    return_result(queue_vector);
}

void InputQueueView::main(std::string string)
{
    json_rpc_server::main(string);
}

void InputQueueController::handle_create_get()
{

    LOG4_DEBUG("Handling Create Dataset GET request");
    content::InputQueue queue;
    queue.pageTitle = "Create InputQueue";
    render("CreateInputQueue", queue);
}

void InputQueueController::handle_create_post()
{
    LOG4_DEBUG("Handling Create Dataset POST request");
    content::InputQueue model;
    model.form.load(context());
    session()["user"] = DEFAULT_WEB_USER;
    if (model.form.validate()) {
        model.load_form_data();
        if (AppContext::get_instance().input_queue_service.exists(DEFAULT_WEB_USER /* session()["user"] */, model.object->get_logical_name(),
                                                                  model.object->get_resolution())) {
            model.form.logical_name.valid(false);
            model.form.resolution.valid(false);
            model.pageError = "Queue with name " + model.object->get_logical_name()
                              + " and resolution " + bpt::to_simple_string(model.object->get_resolution())
                              + " is already created!";
        } else {
            model.object->set_owner_user_name(DEFAULT_WEB_USER /* session()["user"] */);
            if (!AppContext::get_instance().input_queue_service.save(model.object)) {
                model.pageError = "Error while saving inputQueue! Please try again later!";
            } else {
                std::stringstream url;
                mapper().map(url, "/queue/show", model.object->get_table_name());
                response().set_redirect_header(url.str());
                return;
            }
        }
    }
    model.pageTitle = "Create InputQueue";
    render("CreateInputQueue", model);
}

InputQueueView::InputQueueView(cppcms::service &srv) : cppcms::rpc::json_rpc_server(srv)
{
    bind("getAllInputQueues", cppcms::rpc::json_method(&InputQueueView::getAllInputQueues, this), method_role);
    bind("showInputQueue", cppcms::rpc::json_method(&InputQueueView::showInputQueue, this), method_role);
    bind("getValueColumnsModel", cppcms::rpc::json_method(&InputQueueView::getValueColumnsModel, this), method_role);
    bind("sendTick", cppcms::rpc::json_method(&InputQueueView::sendTick, this), method_role);
    bind("getNextTimeRangeToBeSent", cppcms::rpc::json_method(&InputQueueView::getNextTimeRangeToBeSent, this),
         method_role);
    bind("historyData", cppcms::rpc::json_method(&InputQueueView::historyData, this), method_role);
    bind("reconcileHistoricalData", cppcms::rpc::json_method(&InputQueueView::reconcileHistoricalData, this),
         method_role);
}

void InputQueueView::showInputQueue(std::string queueTableName)
{
    return_result(AppContext::get_instance().input_queue_service.load(queueTableName, bpt::min_date_time,
                                                                      bpt::max_date_time, 0));
}

void InputQueueView::getValueColumnsModel(std::string queueTableName)
{
    datamodel::InputQueue_ptr queue = AppContext::get_instance().input_queue_service.get_queue_metadata(queueTableName);
    if (queue.get() == nullptr) {
        return_error("Queue was not found!");
        return;
    }
    std::vector<content::JqgridColModel> colModels;
    int columnNo = 0;
    for (const std::string &columnName : queue->get_value_columns()) {
        content::JqgridColModel colModel;
        colModel.label = columnName;
        colModel.name = "value_" + std::to_string(columnNo++);
        colModel.width = 50;
        colModels.push_back(colModel);
    }
    return_result(colModels);
}


namespace {
svr::dao::OptionalTimeRange
parseTimeRange(session_interface &session, cppcms::json::object obj, string &logicalName, time_duration &resolution, datamodel::InputQueue_ptr &queue)
{
    logicalName = obj["symbol"].str();
    std::transform(C_default_exec_policy, logicalName.begin(), logicalName.end(), logicalName.begin(), ::tolower);
    resolution = seconds(boost::lexical_cast<long>(obj["period"].str()));

    session["user"] = DEFAULT_WEB_USER;
    if (!AppContext::get_instance().input_queue_service.exists(session["user"], logicalName, resolution)) {
        return {};
    }
    queue = AppContext::get_instance().input_queue_service.get_queue_metadata(session["user"], logicalName, resolution);
    ptime offeredTimeFrom = time_from_string(obj["offeredTimeFrom"].str());
    ptime offeredTimeTo = time_from_string(obj["offeredTimeTo"].str());

    LOG4_DEBUG("User " << session["user"] << ", symbol " << obj["symbol"].str() << ", period " << obj["period"].str() <<
        ", time from " << offeredTimeFrom << ", time to " << offeredTimeTo);

    return {std::make_pair(offeredTimeFrom, offeredTimeTo)};
}
}

void InputQueueView::getNextTimeRangeToBeSent(cppcms::json::object obj)
{
    string logicalName;
    time_duration resolution;
    datamodel::InputQueue_ptr queue;

    session()["user"] = DEFAULT_WEB_USER;
    auto timeRange = parseTimeRange(session(), obj, logicalName, resolution, queue);
    if (!timeRange) {
        return_error("No Input Queue has been setup for symbol " + logicalName + " and resolution " + to_simple_string(resolution));
        return;
    }

    datamodel::DataRow_ptr newestRow = AppContext::get_instance().input_queue_service.find_newest_record(queue);
    if (newestRow) timeRange->first = newestRow->get_value_time();
    if (timeRange->first >= timeRange->second) {
        return_error("No data needed for queue " + queue->get_table_name() + " already have up to " + to_simple_string(timeRange->first));
        return;
    }
    //timeRange->second += queue->get_resolution();

    LOG4_DEBUG("Next time range from " << timeRange->first << " to " << timeRange->second);

    json::value response;
    response["timeFrom"] = to_mt4_date(timeRange->first);
    response["timeTo"] = to_mt4_date(timeRange->second);
    LOG4_DEBUG("User " << DEFAULT_WEB_USER /* session()["user"] */ << ", symbol " << obj["symbol"].str() << ", period " << obj["period"].str());

    return_result(response);
}

void InputQueueView::reconcileHistoricalData(cppcms::json::object obj)
{

    string logicalName;
    time_duration resolution;
    datamodel::InputQueue_ptr queue;
    session()["user"] = DEFAULT_WEB_USER;
    auto timeRange = parseTimeRange(session(), obj, logicalName, resolution, queue);
    if (!timeRange) {
        return_error("No Input Queue has been setup for symbol " + logicalName + " and resolution " + to_simple_string(resolution));
        return;
    }

    timeRange = AppContext::get_instance().input_queue_service.get_missing_hours(queue, timeRange.get());

    json::value response;

    if (timeRange) {
        response["timeFrom"] = to_mt4_date(timeRange->first);
        response["timeTo"] = to_mt4_date(timeRange->second);
        return_result(response);
        return;
    }

    AppContext::get_instance().input_queue_service.purge_missing_hours(queue);

    response["message"] = "Reconciliation done";
    return_result(response);
}

namespace {
using parsed_header_element = std::pair<string, string>;
using parsed_header = std::vector<parsed_header_element>;

parsed_header parseMT4DataHeader(const std::string &rawHeader,
                                 const std::string &delimiter)
{
    reject_empty(rawHeader, "Header must not be empty!");
    reject_empty(delimiter, "Delimiter must not be empty!");

    auto headers = split(rawHeader, delimiter);

    parsed_header headersWithType;

    for (string &headerItem : headers) {
        auto tok = split(headerItem, ":");
        if (tok.size() != 2) {
            throw invalid_argument("Cannot get parameter name and type from " + headerItem);
        }
        headersWithType.push_back({tok[0], tok[1]});
    }
    return headersWithType;
}

using db_columns = std::deque<std::string>;

struct header_to_column_mapper
{
    struct builder
    {
        virtual double build(std::deque<std::string> const &) const = 0;

        virtual ~builder() = default;
    };

    struct defaulter : builder
    {
        double build(std::deque<std::string> const &) const
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };

    struct reader : builder
    {
        size_t header_idx;

        reader(size_t header_idx) : header_idx(header_idx)
        {}

        double build(std::deque<std::string> const &string_values) const
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            return std::stod(string_values[header_idx]);
#pragma GCC diagnostic pop
        }
    };

    std::deque<builder const *> builders;

    header_to_column_mapper(parsed_header const &ph, db_columns const &dbc)
            : builders(dbc.size(), nullptr)
    {
        size_t index = 0;
        for (auto const &column : dbc) {

            parsed_header::const_iterator const it_header =
                    std::find_if(ph.begin(), ph.end(), [&column](parsed_header_element const &v) {
                        return ignore_case_equals(v.first, column);
                    });

            if (it_header != ph.end())
                builders[index++] = new reader(it_header - ph.begin());
            else
                builders[index++] = new defaulter();
        }
    }

    ~header_to_column_mapper()
    {
        for (auto bldr : builders)
            delete bldr;
    }

    std::vector<double> map_web_to_db(const std::deque<std::string> &string_values)
    {
        std::vector<double> result;

        for (auto bldr : builders)
            result.push_back(bldr->build(string_values));

        return result;
    }
};

}

void InputQueueView::historyData(cppcms::json::object data)
{
    json::value response;
    string symbol = data["symbol"].str();
    session()["user"] = DEFAULT_WEB_USER;
    std::transform(C_default_exec_policy, symbol.begin(), symbol.end(), symbol.begin(), ::tolower);
    time_duration resolution = seconds(boost::lexical_cast<long>(data["period"].str()));
    LOG4_DEBUG("Resolution " << resolution);// << " data " << data);
    datamodel::InputQueue_ptr queue = AppContext::get_instance().input_queue_service.get_queue_metadata(DEFAULT_WEB_USER /* session()["user"] */, symbol, resolution);
    if (!queue) {
        const std::string err_msg = "Queue for " + session()["user"] + " " + symbol + " " + bpt::to_simple_string(resolution) + " not found!";
        LOG4_ERROR(err_msg);
        return_error(err_msg);
        return;
    }
    LOG4_DEBUG("Found queue " << queue->to_string());

    if (data.find("barFrom") != data.end()) {
        const size_t barFrom = boost::lexical_cast<size_t>(data["barFrom"].str());
        const size_t barTo = boost::lexical_cast<size_t>(data["barTo"].str());

        const string header = data["header"].str();
        const string delimiter = data["delimiter"].str();
        parsed_header headersWithType;

        try {
            headersWithType = parseMT4DataHeader(header, delimiter);
        } catch (const exception &ex) {
            LOG4_ERROR(ex.what());
            return_error(ex.what());
            return;
        }

        size_t const timeIndex = std::find_if(headersWithType.begin(), headersWithType.end(),
                                              [&](pair<string, string> const &v) {
                                                  return ignore_case_equals(v.first, "Time");
                                              }) - headersWithType.begin();
        size_t const volumeIndex = std::find_if(headersWithType.begin(), headersWithType.end(),
                                                [&](pair<string, string> const &v) {
                                                    return ignore_case_equals(v.first, "Volume");
                                                }) - headersWithType.begin();

        header_to_column_mapper mapper(headersWithType, APP.input_queue_service.get_db_table_column_names(queue));

        LOG4_DEBUG("Loaded history queue " << barFrom << " " << barTo);
        for (size_t barIx = barFrom; barIx < barTo; ++barIx) {
            const string row = data[to_string(barIx)].str();
            const auto rowFields = split(row, delimiter);

            if (rowFields.size() != headersWithType.size()) {
                const std::string errMsg = "Invalid data, the row " + row + " doesn't match the supplied header: " + header;
                LOG4_ERROR(errMsg);
                return_error(errMsg);
                return;
            }

            const datamodel::DataRow_ptr queueRow = make_shared<DataRow>( // TODO Implement time zone parsing
                        time_from_string(rowFields[timeIndex]), bpt::second_clock::local_time(),
                        boost::lexical_cast<double>(rowFields[volumeIndex]), mapper.map_web_to_db(rowFields)
                    );
            AppContext::get_instance().input_queue_service.add_row(queue, queueRow);
        }
        LOG4_DEBUG("Saving " << queue->size() << " rows");
        ssize_t savedRows;
        if ((savedRows = AppContext::get_instance().input_queue_service.save(queue)) > 0) {
            response["message"] = "OK";
            response["savedRows"] = savedRows;
            LOG4_DEBUG("Saved " << savedRows << " rows.");
            return_result(response);
        } else {
            const std::string errMsg = "Cannot save the data!";
            LOG4_ERROR(errMsg);
            response["message"] = errMsg;
            return_error(response);
        }
    } else {
        LOG4_DEBUG("Missed hours detected for queue " << queue->get_table_name());
        response["message"] = "OK";
        return_result(response);
    }
}

void InputQueueView::sendTick(cppcms::json::object obj)
{

    string logicalName = obj["symbol"].str();
    std::transform(C_default_exec_policy, logicalName.begin(), logicalName.end(), logicalName.begin(), ::tolower);
    time_duration resolution = seconds(boost::lexical_cast<long>(obj["period"].str()));
    session()["user"] = DEFAULT_WEB_USER;
    if (!AppContext::get_instance().input_queue_service.exists(session()["user"], logicalName, resolution)) {
        return_error("No Input Queue has been setup for symbol " + logicalName + " and resolution " + to_simple_string(resolution));
        return;
    }
    datamodel::InputQueue_ptr queue = AppContext::get_instance().input_queue_service.get_queue_metadata(session()["user"], logicalName, resolution);
    const auto db_columns = AppContext::get_instance().input_queue_service.get_db_table_column_names(queue);
    std::vector<double> values;
    values.reserve(db_columns.size());
    for (const auto &column: db_columns) {
        auto iter = obj.find(column);
        const double value = iter == obj.end() ? std::numeric_limits<double>::quiet_NaN() : std::stod(iter->second.str());
        values.push_back(value);
    }

    double tick_volume = 0;
    {
        auto iter = obj.find("Volume");
        if (iter != obj.end()) tick_volume = std::stod(iter->second.str());
    }

    bpt::ptime valueTime = bpt::time_from_string(obj["time"].str());
    datamodel::DataRow_ptr row = ptr<DataRow>(valueTime, second_clock::local_time(), tick_volume, values);

    LOG4_DEBUG(row->to_string());

    json::value response;
    AppContext::get_instance().input_queue_service.add_row(queue, row);
    if ((AppContext::get_instance().input_queue_service.save(queue) > 0)) {
        response["message"] = "OK";
        response["savedRows"] = queue->size();
        return_result(response);
    } else {
        response["message"] = "Cannot save the data!";
        return_error(response);
    }
}


}
}

