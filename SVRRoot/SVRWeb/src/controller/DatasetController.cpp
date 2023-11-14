#include "controller/DatasetController.hpp"
#include "view/DatasetView.hpp"
#include "appcontext.hpp"

#include "controller/MainController.hpp"

using namespace svr::datamodel;
using namespace svr::context;

namespace svr{
namespace web {

void DatasetController::show(std::string datasetName) {
    content::Dataset model;
    model.pageTitle = "Dataset";
    Dataset_ptr dataset = AppContext::get_instance().dataset_service.get_user_dataset(DEFAULT_WEB_USER /* session()["user"] */, datasetName);

    if(dataset.get() == nullptr){
        model.pageError = "No such dataset exists!";
    }else{
        model.dataset_name = dataset->get_dataset_name();
        model.description = dataset->get_description();
        model.user_name = dataset->get_user_name();
        //!!TODO fix!!!
//        model.lookback_time = bpt::to_simple_string(dataset->get_lookback_time());
        model.priority = svr::datamodel::to_string(dataset->get_priority());
        model.transformation_levels = std::to_string(dataset->get_transformation_levels());
        model.transformation_wavelet = dataset->get_transformation_name();
    }
    render("ShowDataset", model);
}

void DatasetController::showAll() {
    content::Main main;
    main.pageTitle = "Datasets";
    render("Datasets", main);
}

void DatasetController::create() {
    if(request().request_method() == "POST"){
        handle_create_post();
    } else {
        handle_create_get();
    }
}

void DatasetView::getAllDatasets() {
    return_result(AppContext::get_instance().dataset_service.find_all_user_datasets(DEFAULT_WEB_USER /* session()["user"] */));
}

void DatasetController::handle_create_get() {
    content::DatasetWithForm dataset;
    dataset.pageTitle = "Create Dataset";
    render("CreateDataset", dataset);
}

void DatasetController::handle_create_post() {
    content::DatasetWithForm model;
    model.form.load(context());

    if(model.form.validate()){
        model.load_form_data();
        if(AppContext::get_instance().dataset_service.exists(DEFAULT_WEB_USER /* session()["user"] */, model.object->get_dataset_name())){
            model.form.name.valid(false);
            model.form.name.error_message("The Dataset with name " + model.object->get_dataset_name() + " already created!");
        } else{
            model.object->set_user_name(DEFAULT_WEB_USER /* session()["user"] */);
            if(!AppContext::get_instance().dataset_service.save(model.object)){
                model.pageError = "Error saving dataset: Please try again later!";
            } else{
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
