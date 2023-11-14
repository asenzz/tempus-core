#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>

#include <fstream>
#include "appcontext.hpp"
#include "controller/MT4LoginController.hpp"

using namespace cppcms;

namespace svr {
namespace web {
class MT4Controller : public application {
    json::value json;
public:
    MT4Controller(cppcms::service &svc) : application(svc) {

        attach(new MT4LoginController(svc), "login", "/login{1}", "/login(/(.*))?", 1);
        dispatcher().assign("/hello", &MT4Controller::hello, this);

        mapper().root(rootPath);

    }

    void hello(){
        LOG4_DEBUG("Hello from cppcms!");
        response().out() << "Hello from cppcms!";
    }
    const std::string rootPath = "/mt4";
};
}
}