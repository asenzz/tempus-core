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

void InputQueueController::show(std::string queue_name)
{
    session()["user"] = DEFAULT_WEB_USER;
    content::InputQueue model;
    model.pageTitle = "InputQueue Details";
    const auto queue = AppContext::get().input_queue_service.get_queue_metadata(queue_name);
    if (queue.get() != nullptr && queue->get_owner_user_name() == DEFAULT_WEB_USER /* session()["user"] */ )
        model.object = queue;
    else
        model.pageError = "No queue named \"" + queue_name + "\" was found!";

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
    if (request().request_method() == "POST") handle_create_post();
    else handle_create_get();
}

void InputQueueView::getAllInputQueues()
{
    session()["user"] = DEFAULT_WEB_USER;
    std::vector<datamodel::InputQueue_ptr> queue_vector;
    const auto queues = APP.input_queue_service.get_all_user_queues(DEFAULT_WEB_USER /* session()["user"] */ );
    std::copy(C_default_exec_policy, queues.cbegin(), queues.cend(), std::back_inserter(queue_vector));
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
        if (AppContext::get().input_queue_service.exists(DEFAULT_WEB_USER /* session()["user"] */, model.object->get_logical_name(), model.object->get_resolution())) {
            model.form.logical_name.valid(false);
            model.form.resolution.valid(false);
            model.pageError = "Queue with name " + model.object->get_logical_name() + " and resolution " + bpt::to_simple_string(model.object->get_resolution())
                              + " is already created!";
        } else {
            model.object->set_owner_user_name(DEFAULT_WEB_USER /* session()["user"] */);
            if (!AppContext::get().input_queue_service.save(model.object)) {
                model.pageError = "Error while saving p_input_queue! Please try again later!";
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
    bind("getNextTimeRangeToBeSent", cppcms::rpc::json_method(&InputQueueView::getNextTimeRangeToBeSent, this), method_role);
    bind("historyData", cppcms::rpc::json_method(&InputQueueView::historyData, this), method_role);
    bind("reconcileHistoricalData", cppcms::rpc::json_method(&InputQueueView::reconcileHistoricalData, this), method_role);
}

void InputQueueView::showInputQueue(std::string queueTableName)
{
    return_result(AppContext::get().input_queue_service.load(queueTableName, bpt::min_date_time, bpt::max_date_time, 0));
}

void InputQueueView::getValueColumnsModel(std::string queue_table_name)
{
    const auto p_queue = AppContext::get().input_queue_service.get_queue_metadata(queue_table_name);
    if (!p_queue) {
        return_error("Queue was not found!");
        return;
    }
    const auto &value_columns = p_queue->get_value_columns();
    std::vector<content::JqgridColModel> col_models(value_columns.size());
    OMP_FOR_i(value_columns.size()) {
        auto &col_model = col_models[i];
        col_model.label = value_columns[i];
        col_model.name = "value_" + std::to_string(i);
        col_model.width = 50;
    }
    return_result(col_models);
}


namespace {

svr::dao::OptionalTimeRange
parse_time_range(session_interface &session, cppcms::json::object obj, string &logical_name, time_duration &resolution, datamodel::InputQueue_ptr &queue)
{
    LOG4_BEGIN();

    const auto offered_time_from = time_from_string(obj["offeredTimeFrom"].str());
    const auto offered_time_to = time_from_string(obj["offeredTimeTo"].str());

    logical_name = obj["symbol"].str();
    resolution = seconds(boost::lexical_cast<long>(obj["period"].str()));

    session["user"] = DEFAULT_WEB_USER;
    queue = AppContext::get().input_queue_service.get_queue_metadata(DEFAULT_WEB_USER, logical_name, resolution);
    if (!queue) {
        LOG4_ERROR("No Input Queue has been setup for symbol " << logical_name << " and resolution " << to_simple_string(resolution));
        return {};
    }
    LOG4_DEBUG("User " << session["user"] << ", symbol " << obj["symbol"].str() << ", period " << obj["period"].str() <<
                       ", time from " << offered_time_from << ", time to " << offered_time_to);
    LOG4_END();
    return {std::make_pair(offered_time_from, offered_time_to)};
}

}

/* TODO Implement time zone conversion at time_from_string() calls, thorough range checking for missing values according to forex market hours and comparison with other queues
 * by adding time zone information to the client POST request and a 'thorough' flag.
 */
void InputQueueView::getNextTimeRangeToBeSent(cppcms::json::object obj)
{
    LOG4_BEGIN();

    string logical_name;
    time_duration resolution;
    datamodel::InputQueue_ptr queue;

    session()["user"] = DEFAULT_WEB_USER;
    auto time_range = parse_time_range(session(), obj, logical_name, resolution, queue);
    if (!time_range) {
        return_error("No Input Queue has been setup for symbol " + logical_name + " and resolution " + to_simple_string(resolution));
        return;
    }
    LOG4_TRACE("Offered time range from " << time_range->first << " to " << time_range->second);
    const auto newest_row = AppContext::get().input_queue_service.find_newest_record(queue);
    if (newest_row) time_range->first = newest_row->get_value_time();
    if (time_range->first >= time_range->second) {
        const string msg = "No data needed for queue " + queue->get_table_name() + " already have up to " + to_simple_string(time_range->first);
        LOG4_DEBUG(msg);
        return_error(msg);
        return;
    }
    //timeRange->second += queue->get_resolution();

    LOG4_DEBUG("Next time range from " << time_range->first << " to " << time_range->second);

    json::value response;
    response["timeFrom"] = to_mql_date(time_range->first);
    response["timeTo"] = to_mql_date(time_range->second);
    LOG4_DEBUG("User " << DEFAULT_WEB_USER /* session()["user"] */ << ", symbol " << obj["symbol"].str() << ", period " << obj["period"].str());
    return_result(response);
}

void InputQueueView::reconcileHistoricalData(cppcms::json::object obj)
{

    string logical_name;
    time_duration resolution;
    datamodel::InputQueue_ptr queue;
    session()["user"] = DEFAULT_WEB_USER;
    auto time_range = parse_time_range(session(), obj, logical_name, resolution, queue);
    if (!time_range) {
        const string msg = "No Input Queue has been setup for symbol " + logical_name + " and resolution " + to_simple_string(resolution);
        LOG4_ERROR(msg);
        return_error(msg);
        return;
    }

    time_range = AppContext::get().input_queue_service.get_missing_hours(queue, time_range.get());

    json::value response;

    if (time_range) {
        response["timeFrom"] = to_mql_date(time_range->first);
        response["timeTo"] = to_mql_date(time_range->second);
        return_result(response);
        return;
    }

    AppContext::get().input_queue_service.purge_missing_hours(queue);

    response["message"] = "Reconciliation done";
    return_result(response);
}

namespace {

using parsed_header_element = std::pair<string, string>;
using parsed_header = std::vector<parsed_header_element>;

parsed_header parseMT4DataHeader(const std::string &raw_header, const std::string &delimiter)
{
    REJECT_EMPTY(raw_header, "Header must not be empty!");
    REJECT_EMPTY(delimiter, "Delimiter must not be empty!");

    const auto headers = split(raw_header, delimiter);
    parsed_header headers_with_type;
    for (const auto &header_item: headers) {
        auto tok = split(header_item, ":");
        if (tok.size() != 2) THROW_EX_FS(invalid_argument, "Cannot get parameter name and type from " + header_item);
        headers_with_type.emplace_back(tok[0], tok[1]);
    }
    return headers_with_type;
}

using db_columns = std::deque<std::string>;

struct header_to_column_mapper {
    struct builder {
        virtual double build(std::deque<std::string> const &) const = 0;

        virtual ~builder() = default;
    };

    struct defaulter : builder {
        double build(std::deque<std::string> const &) const
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };

    struct reader : builder {
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

    header_to_column_mapper(parsed_header const &ph, db_columns const &dbc) : builders(dbc.size(), nullptr)
    {
        OMP_FOR_i(dbc.size()) {
            const auto &column = dbc[i];
            const auto it_header = std::find_if(ph.cbegin(), ph.cend(), [&column](const auto &v) {
                return ignore_case_equals(v.first, column);
            });
            builders[i] = it_header == ph.end() ? new defaulter() : (const header_to_column_mapper::builder *) new reader(it_header - ph.begin());
        }
    }

    ~header_to_column_mapper()
    {
        OMP_FOR(builders.size())
        for (const auto bldr: builders) delete bldr;
    }

    std::vector<double> map_web_to_db(const std::deque<std::string> &string_values)
    {
        std::vector<double> result;
        for (auto bldr: builders) result.emplace_back(bldr->build(string_values));
        return result;
    }
};

}

void InputQueueView::historyData(cppcms::json::object data)
{
    json::value response;
    const auto symbol = data["symbol"].str();
    session()["user"] = DEFAULT_WEB_USER;
    const auto resolution = seconds(boost::lexical_cast<uint32_t>(data["period"].str()));
    LOG4_DEBUG("Resolution " << resolution);// << " data " << data);
    auto p_queue = AppContext::get().input_queue_service.get_queue_metadata(DEFAULT_WEB_USER /* session()["user"] */, symbol, resolution);
    if (!p_queue) {
        const std::string err_msg = "Queue for " + session()["user"] + " " + symbol + " " + bpt::to_simple_string(resolution) + " not found!";
        LOG4_ERROR(err_msg);
        return_error(err_msg);
        return;
    }
    LOG4_DEBUG("Found queue " << p_queue->to_string());

    if (data.find("barFrom") != data.cend()) {
        const auto bar_from = boost::lexical_cast<uint32_t>(data["barFrom"].str());
        const auto bar_to = boost::lexical_cast<uint32_t>(data["barTo"].str());
        const auto header = data["header"].str();
        const auto delimiter = data["delimiter"].str();
        parsed_header headers_with_type;

        try {
            headers_with_type = parseMT4DataHeader(header, delimiter);
        } catch (const exception &ex) {
            const std::string msg = common::formatter() << "Error parsing headers " << header << ", " << ex.what();
            LOG4_ERROR(msg);
            return_error(msg);
            return;
        }

        const uint32_t time_index = std::find_if(C_default_exec_policy, headers_with_type.cbegin(), headers_with_type.cend(),
                                                 [](const auto &v) { return ignore_case_equals(v.first, "Time"); }) - headers_with_type.cbegin();
        if (time_index >= headers_with_type.size()) {
            constexpr char err_msg[] = "Time column not found!";
            LOG4_ERROR(err_msg);
            return_error(err_msg);
            return;
        }
        const uint32_t volume_index = std::find_if(C_default_exec_policy, headers_with_type.cbegin(), headers_with_type.cend(),
                                                   [](const auto &v) { return ignore_case_equals(v.first, "Volume"); }) - headers_with_type.cbegin();
        if (volume_index >= headers_with_type.size()) {
            constexpr char err_msg[] = "Volume column not found!";
            LOG4_ERROR(err_msg);
            return_error(err_msg);
            return;
        }
        header_to_column_mapper mapper(headers_with_type, APP.input_queue_service.get_db_table_column_names(p_queue));
        const auto local_time = second_clock::local_time();
        LOG4_DEBUG("Loaded history queue " << bar_from << " " << bar_to);
        bool break_loop = false;
        OMP_FOR_(bar_to - bar_from, ordered)
        for (uint32_t bar_ix = bar_from; bar_ix < bar_to; ++bar_ix) {
            if (break_loop) continue;
            const auto &row_str = data[std::to_string(bar_ix)].str();
            const auto row_fields = split(row_str, delimiter);
            if (row_fields.size() != headers_with_type.size()) {
                const std::string err_msg = "Invalid data, the row " + row_str + " doesn't match the supplied header: " + header;
                LOG4_ERROR(err_msg);
                return_error(err_msg);
#pragma omp atomic write
                break_loop = true;
            }
            const auto p_row = ptr<DataRow>( // TODO Implement time zone parsing
                    bpt::time_from_string(row_fields[time_index]), local_time, boost::lexical_cast<double>(row_fields[volume_index]), mapper.map_web_to_db(row_fields)
            );
#pragma omp ordered
            business::InputQueueService::add_row(*p_queue, p_row);
        }
        if (break_loop) return;

        LOG4_DEBUG("Saving " << p_queue->size() << " rows");
        size_t saved_rows;
        if ((saved_rows = AppContext::get().input_queue_service.save(p_queue)) > 0) {
            response["message"] = "OK";
            response["savedRows"] = saved_rows;
            LOG4_DEBUG("Saved " << saved_rows << " rows.");
            return_result(response);
        } else {
            constexpr char err_msg[] = "Cannot save the data!";
            LOG4_ERROR(err_msg);
            response["message"] = err_msg;
            return_error(response);
        }
    } else {
        LOG4_DEBUG("Missed hours detected for queue " << p_queue->get_table_name());
        response["message"] = "OK";
        return_result(response);
    }
}

void InputQueueView::sendTick(cppcms::json::object obj)
{
    LOG4_BEGIN();

    const auto logical_name = obj["symbol"].str();
    const auto resolution = seconds(boost::lexical_cast<uint32_t>(obj["period"].str()));
    // session()["user"] = DEFAULT_WEB_USER;
    // TODO Replace with specialized stored procedure
    const auto p_queue = AppContext::get().input_queue_service.get_queue_metadata(DEFAULT_WEB_USER, logical_name, resolution);
    if (!p_queue) {
        const std::string msg = "Queue for " DEFAULT_WEB_USER /* session()["user"] */ " " + logical_name + " " + to_simple_string(resolution) + " not found!";
        LOG4_ERROR(msg);
        return_error(msg);
        return;
    }
    const auto &db_columns = p_queue->get_value_columns();
    auto values = (const char ** const) malloc(db_columns.size() * sizeof(char *));
    std::deque<std::string> value_column_str(db_columns.size());
    LOG4_TRACE("Input queue has " << db_columns.size() << " columns");
    uint16_t i = 0;
    for (const auto &column: db_columns) { // Ordered values
        const auto iter = obj.find(column);
        if (iter == obj.cend()) {
            const std::string msg = "Column " + column + " is missing in received data!";
            LOG4_ERROR(msg);
            return_error(msg);
            return;
        }
        value_column_str[i] = "\"" + column + "\":" + iter->second.str();
        values[i] = value_column_str[i].c_str();
        ++i;
        LOG4_TRACE("Column " << i << ", " << column << ", value " << iter->second.str());
    }
    const auto value_time_str = "\"value_time\":\"" + obj["time"].str() + "\"";
    const auto update_time_str = "\"update_time\":\"" + bpt::to_simple_string(second_clock::local_time()) + "\"";
    const auto volume_str = "\"tick_volume\":" + obj["Volume"].str();
    json::value response;
    try {
        APP.input_queue_service.upsert_row_str(p_queue->get_table_name().c_str(), value_time_str.c_str(), update_time_str.c_str(), volume_str.c_str(), values, i);
        response["message"] = "OK";
        response["savedRows"] = p_queue->size();
    } catch (const std::exception &ex) {
        response.set<std::string>("message", common::formatter() << "Cannot save the data!" << ex.what());
        return_error(response);
        return;
    }
    free(values);
    return_result(response);
}


}
}

