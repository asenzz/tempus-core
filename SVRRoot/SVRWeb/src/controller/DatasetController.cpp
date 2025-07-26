#include "appcontext.hpp"
#include "controller/DatasetController.hpp"
#include "controller/MainController.hpp"
#include "DatasetService.hpp"
#include "view/DatasetView.hpp"

namespace svr {
namespace web {
void DatasetController::show(const std::string dataset_name)
{
    content::Dataset model;
    model.pageTitle = "Dataset";
    const auto p_dataset = APP.dataset_service.get_user_dataset(DEFAULT_WEB_USER /* session()["user"] */, dataset_name);

    if (p_dataset.get() == nullptr) {
        model.pageError = "No such dataset exists!";
    } else {
        model.dataset_name = p_dataset->get_dataset_name();
        model.description = p_dataset->get_description();
        model.user_name = p_dataset->get_user_name();
        model.lookback_time = bpt::to_simple_string(p_dataset->get_max_lookback_time_gap());
        model.priority = datamodel::to_string(p_dataset->get_priority());
        model.gradients = std::to_string(p_dataset->get_gradient_count());
        model.chunk_size = std::to_string(p_dataset->get_max_chunk_size());
        model.multiout = std::to_string(p_dataset->get_multistep());
        model.transformation_levels = std::to_string(p_dataset->get_spectral_levels());
        model.transformation_wavelet = p_dataset->get_transformation_name();
    }
    render("ShowDataset", model);
}

void DatasetController::showAll()
{
    content::Main main;
    main.pageTitle = "Datasets";
    render("Datasets", main);
}

void DatasetController::create()
{
    if (request().request_method() == "POST") {
        handle_create_post();
    } else {
        handle_create_get();
    }
}

void DatasetView::getAllDatasets()
{
    const auto r = APP.dataset_service.find_all_user_datasets(DEFAULT_WEB_USER /* session()["user"] */);
    return_result(std::vector<datamodel::Dataset_ptr>{r.begin(), r.end()});
}

void DatasetController::handle_create_get()
{
    content::DatasetWithForm dataset;
    dataset.pageTitle = "Create Dataset";
    render("CreateDataset", dataset);
}

void DatasetController::handle_create_post()
{
    content::DatasetWithForm model;
    model.form.load(context());

    if (model.form.validate()) {
        model.load_form_data();
        if (APP.dataset_service.exists(DEFAULT_WEB_USER /* session()["user"] */, model.object->get_dataset_name())) {
            model.form.name.valid(false);
            model.form.name.error_message("The Dataset with name " + model.object->get_dataset_name() + " already created!");
        } else {
            model.object->set_user_name(DEFAULT_WEB_USER /* session()["user"] */);
            if (!APP.dataset_service.save(model.object)) {
                model.pageError = "Error saving dataset: Please try again later!";
            } else {
                std::stringstream url;
                mapper().map(url, "/dataset/show", model.object->get_dataset_name());
                response().set_redirect_header(url.str());
                return;
            }
        }
    }
    model.pageTitle = "Create Dataset";
    render("CreateDataset", model);
}
}
}
