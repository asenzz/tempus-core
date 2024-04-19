#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>

#include "controller/LoginController.hpp"
#include "controller/InputQueueController.hpp"
#include "controller/DatasetController.hpp"
#include "controller/UserController.hpp"
#include "controller/RequestController.hpp"
#include <fstream>
#include <common/logging.hpp>
#include <view/MainView.hpp>

using namespace cppcms;

#define DEFAULT_WEB_USER "svrwave"

namespace svr {
namespace web {
class MainController : public application {

public:
    MainController(cppcms::service &svc) : application(svc) {
        LOG4_BEGIN();
        dispatcher().assign("/", &MainController::index, this);
        mapper().assign("index", "/");

        mapper().root(rootPath);

        attach(new LoginController(svc), "login", "/login{1}", "/login(/(.*))?", 1);
        attach(new InputQueueController(svc), "queue", "/queue{1}", "/queue(/(.*))?", 1);
        attach(new DatasetController(svc), "dataset", "/dataset{1}", "/dataset(/(.*))?", 1);
        attach(new UserController(svc), "user", "/user{1}", "/user(/(.*))?", 1);
        attach(new RequestController(svc), "request", "/request{1}", "/request(/(.*))?", 1);
        LOG4_END();
    }
    void index();

    virtual void main(std::string) override;

    bool logged_in();
private:
    const std::string rootPath = "/web";
    const std::string loginPath = "/login/";
    const std::string loginUrl = rootPath + loginPath;
};
}
}