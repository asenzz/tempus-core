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
class MT4LoginController : public cppcms::application{

    void handle_login_post();
    void handle_login_get();


public:
    MT4LoginController(cppcms::service &svc): application(svc){
        dispatcher().assign("/?", &MT4LoginController::login, this);
        mapper().assign("login", "");
    }

    void login();
    void logout();
};


}
}
