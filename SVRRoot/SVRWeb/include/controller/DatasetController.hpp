#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>
#include <cppcms/rpc_json.h>

namespace svr{
namespace web{

class DatasetView : public cppcms::rpc::json_rpc_server{
public:
    DatasetView(cppcms::service &srv) : cppcms::rpc::json_rpc_server(srv) {
        bind("getAllDatasets", cppcms::rpc::json_method(&DatasetView::getAllDatasets, this), method_role);
    }

    void getAllDatasets();

};

class DatasetController : public cppcms::application{

    void handle_create_get();
    void handle_create_post();

public:
    DatasetController(cppcms::service &svc): application(svc){
        dispatcher().assign("/", &DatasetController::showAll, this);
        mapper().assign("showall", "/");

        dispatcher().assign("/show/(\\w+)", &DatasetController::show, this, 1);
        mapper().assign("show", "/show/{1}");

        dispatcher().assign("/create", &DatasetController::create, this);
        mapper().assign("create", "/create");

        attach(new DatasetView(svc), "dataset_ajaxview", "/ajax{1}", "/ajax(/(.*))?", 1);
    }

    void show(std::string datasetName);
    void showAll();
    void create();

};


}
}
