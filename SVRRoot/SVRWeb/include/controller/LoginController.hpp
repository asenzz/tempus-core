#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_request.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>
#include <cppcms/session_interface.h>
#include <common/Logging.hpp>

namespace svr{
namespace web{
class LoginController : public cppcms::application{

    void handle_login_post();
    void handle_login_get();


public:
    LoginController(cppcms::service &svc): application(svc){
        dispatcher().assign("/?", &LoginController::login, this);
        mapper().assign("login", "");
    }

    void main(std::string url) override{
        application::main(url);
    }

    void login();
    void logout();
};


}
}
