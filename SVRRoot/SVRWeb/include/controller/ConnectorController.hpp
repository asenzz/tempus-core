#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>

#include <fstream>
#include "appcontext.hpp"
#include "controller/ConnectorLoginController.hpp"

using namespace cppcms;

namespace svr {
namespace web {

class ConnectorController : public application {
    json::value json;

public:
    ConnectorController(cppcms::service &svc) : application(svc)
    {
        attach(new ConnectorLoginController(svc), "login", "/login{1}", "/login(/(.*))?", 1);
        mapper().root(root_path);
    }

    const std::string root_path = "/tempus";
};

}
}
