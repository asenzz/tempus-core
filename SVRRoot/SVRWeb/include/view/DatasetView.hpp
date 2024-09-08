#pragma once

#include <cppcms/json.h>

#include <view/MainView.hpp>
#include "model/Dataset.hpp"

using namespace svr;
namespace content {

struct DatasetForm : cppcms::form
{

    cppcms::widgets::regex_field name;
    cppcms::widgets::text description;
    cppcms::widgets::text lookback_time;
    cppcms::widgets::select priority;
    cppcms::widgets::numeric<double> svr_C;
    cppcms::widgets::numeric<double> svr_epsilon;
    cppcms::widgets::numeric<double> svr_kernel_param;
    cppcms::widgets::numeric<double> svr_kernel_param2;
    cppcms::widgets::numeric<size_t> svr_decremental_distance;
    cppcms::widgets::numeric<double> svr_adjacent_levels_ratio;
    cppcms::widgets::text svr_kernel_type;
    cppcms::widgets::numeric<size_t> transformation_levels;
    cppcms::widgets::numeric<size_t> gradients;
    cppcms::widgets::numeric<size_t> chunk_size;
    cppcms::widgets::select transformation_wavelet;
    cppcms::widgets::submit submit;

    DatasetForm();

    virtual bool validate() override;

private:

    std::vector<std::pair<std::string, std::string>> get_wavelet_filters();

    bool validate_lookback_time();

    const std::string time_duration_pattern = "[-]h[h][:mm][:ss][.fff]";

};

struct Dataset : public Main
{
    std::string dataset_name;
    std::string user_name;
    std::string priority;
    std::string description;
    std::string gradients;
    std::string chunk_size;
    std::string multiout;
    std::string transformation_levels;
    std::string transformation_wavelet;
    std::string lookback_time;
    std::string svr_c;
    std::string svr_epsilon;
    std::string svr_kernel_param;
    std::string svr_kernel_param2;
    std::string svr_decremental_distance;
    std::string svr_adjacent_levels_ratio;
};

struct DatasetWithForm : public Main
{

    datamodel::Dataset_ptr object;
    DatasetForm form;

    void load_form_data();
};
}


namespace cppcms {
namespace json {

// We specilize cppcms::json::traits structure to convert
// objects to and from json values

template<>
struct traits<datamodel::Dataset_ptr>
{
    static datamodel::Dataset_ptr get(value const &v)
    {
        if (v.type() != is_object) {
            throw bad_value_cast();
        }
        datamodel::Dataset_ptr dataset;

        dataset->set_id(v.get<bigint>("dataset_id"));
        dataset->set_dataset_name(v.get<std::string>("dataset_name"));
        dataset->set_user_name(v.get<std::string>("user_name"));
        dataset->set_priority(svr::datamodel::get_priority_from_string(v.get<std::string>("priority")));
        dataset->set_description(v.get<std::string>("description"));
        dataset->set_gradients(v.get<size_t>("gradients"));
        dataset->set_chunk_size(v.get<size_t>("max_chunk_size"));
        dataset->set_multistep(v.get<size_t>("multiout"));
        dataset->set_spectrum_levels(v.get<size_t>("transformation_levels"));
        dataset->set_transformation_name(v.get<std::string>("transformation_name"));

        return dataset;
    }

    static void set(value &v, datamodel::Dataset_ptr const &in)
    {

        v.set("dataset_id", in->get_id());
        v.set("dataset_name", in->get_dataset_name());
        v.set("user_name", in->get_user_name());
        v.set("priority", svr::datamodel::to_string(in->get_priority()));
        v.set("description", in->get_description());
        v.set("gradients", in->get_spectral_levels());
        v.set("max_chunk_size", in->get_spectral_levels());
        v.set("multiout", in->get_spectral_levels());
        v.set("transformation_levels", in->get_spectral_levels());
        v.set("transformation_name", in->get_transformation_name());
    }
};
} // json
} // cppcms
