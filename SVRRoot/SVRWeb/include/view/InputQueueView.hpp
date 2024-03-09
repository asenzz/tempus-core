#pragma once

#include <cppcms/view.h>
#include <cppcms/form.h>
#include <cppcms/json.h>

#include <view/MainView.hpp>
#include <model/InputQueue.hpp>

namespace content {

struct InputQueueForm : cppcms::form
{

    cppcms::widgets::regex_field logical_name;
    cppcms::widgets::text description;
    cppcms::widgets::text resolution;
    cppcms::widgets::text legal_time_deviation;
    cppcms::widgets::text timezone;
    std::vector<std::unique_ptr<cppcms::widgets::text>> value_columns;
    cppcms::widgets::submit save;

    InputQueueForm();

    virtual bool validate() override;

private:
    bool validate_queue_name();

    bool validate_resolution();

    bool validate_legal_time_deviation();

    bool validate_value_columns();

    const std::string logical_name_pattern = "[a-zA-Z]\\w+";
    const std::string time_duration_pattern = "[-]h[h][:mm][:ss][.fff]";

};

struct InputQueue : public Main
{
    svr::datamodel::InputQueue_ptr object = svr::dtr<svr::datamodel::InputQueue>();
    InputQueueForm form;

    void load_form_data()
    {
        object->set_logical_name(form.logical_name.value());
        object->set_description(form.description.value());
        object->set_resolution(bpt::duration_from_string(form.resolution.value()));
        object->set_legal_time_deviation(bpt::duration_from_string(form.legal_time_deviation.value()));
        object->set_time_zone(form.timezone.value());

        std::deque<std::string> value_columns;
        for (const auto &value_column_field: form.value_columns) {
            if (!value_column_field->value().empty())
                value_columns.emplace_back(value_column_field->value());
        }
        object->set_value_columns(value_columns);
    }
};

struct JqgridColModel
{
    std::string label;
    std::string name;
    int width;
};
}


namespace cppcms {
namespace json {

// We specilize cppcms::json::traits structure to convert
// objects to and from json values

template<>
struct traits<svr::datamodel::InputQueue_ptr>
{
// TODO: unused    static datamodel::InputQueue_ptr get(value const &v)
//    {
//        if(v.type()!=is_object) {
//            throw bad_value_cast();
//        }
//        datamodel::InputQueue_ptr queue;
//
//        queue->set_table_name(v.get<std::string>("table_name"));
//        queue->set_logical_name(v.get<std::string>("queue_name"));
//        queue->set_owner_user_name(v.get<std::string>("owner"));
//        queue->set_description(v.get<std::string>("description"));
//        queue->set_resolution(bpt::duration_from_string(v.get<std::string>("resolution")));
//        queue->set_legal_time_deviation(bpt::duration_from_string(v.get<std::string>("legal_time_deviation")));
//        queue->set_value_columns(v.get<std::vector<std::string> >("data_columns"));
//        queue->set_is_sparse_data(v.get<std::string>("is_sparse") == "TRUE");
//        queue->set_time_zone(v.get<std::string>("timezone"));
//
//        return queue;
//    }
    static void set(value &v, svr::datamodel::InputQueue_ptr const &in)
    {
        v.set("table_name", in->get_table_name());
        v.set("queue_name", in->get_logical_name());
        v.set("owner", in->get_owner_user_name());
        v.set("description", in->get_description());
        v.set("resolution", bpt::to_simple_string(in->get_resolution()));
        v.set("legal_time_deviation", bpt::to_simple_string(in->get_legal_time_deviation()));
        std::vector<std::string> value_columns;
        const auto res_value_columns = in->get_value_columns();
        value_columns.insert(value_columns.end(), res_value_columns.begin(), res_value_columns.end());
        v.set("data_columns", value_columns);
        v.set("timezone", in->get_time_zone());
    }
};


template<>
struct traits<content::JqgridColModel>
{
    static content::JqgridColModel get(value const &v)
    {
        return content::JqgridColModel();
    }

    static void set(value &v, content::JqgridColModel const &in)
    {
        v.set("label", in.label);
        v.set("name", in.name);
        v.set("width", in.width);
    }
};

template<>
struct traits<svr::datamodel::DataRow_ptr>
{

    static svr::datamodel::DataRow_ptr get(value const &v)
    {
        return svr::ptr<svr::datamodel::DataRow>();
    }

    static void set(value &v, svr::datamodel::DataRow_ptr const &in)
    {
        v.set("value_time", bpt::to_simple_string(in->get_value_time()));
        v.set("update_time", bpt::to_simple_string(in->get_update_time()));
        v.set("tick_volume", in->get_tick_volume());
        int valueNo = 0;
        for (const double &value: in->get_values()) {
            v.set("value_" + std::to_string(valueNo++), value);
        }
    }
};

template<>
struct traits<svr::datamodel::DataRow::container>
{

    static svr::datamodel::DataRow::container get(value const &v)
    {
        return svr::datamodel::DataRow::container();
    }

    static void set(value &v, svr::datamodel::DataRow::container const &in)
    {
        std::vector<cppcms::json::value> json_rows;
        for (auto &row: in) {
            json_rows.emplace_back(row);
        }
        v.array(json_rows);
    }
};
} // json
} // cppcms
