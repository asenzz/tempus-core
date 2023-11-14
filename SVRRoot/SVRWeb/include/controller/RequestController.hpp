//
// Created by vg on 27.6.15.
//

#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>
#include <cppcms/rpc_json.h>

namespace svr {
namespace web {

class RequestView : public cppcms::rpc::json_rpc_server{

public:
    RequestView(cppcms::service &svc) : cppcms::rpc::json_rpc_server(svc){
        bind("makeMultivalRequest", cppcms::rpc::json_method(&RequestView::makeMultivalRequest, this), method_role);
        bind("getMultivalResults", cppcms::rpc::json_method(&RequestView::getMultivalResults, this), method_role);
    }

private:
    virtual void main(std::string) override;
    void makeMultivalRequest(cppcms::json::object data);
    void getMultivalResults(cppcms::json::object data);
};

class RequestController : public cppcms::application{
public:
    RequestController(cppcms::service &svc): application(svc){
        attach(new RequestView(svc), "request_ajaxview", "/ajax{1}", "/ajax(/(.*))?", 1);
    }
};
}
}