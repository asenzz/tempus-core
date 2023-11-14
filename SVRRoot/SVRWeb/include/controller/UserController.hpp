#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>
#include <cppcms/rpc_json.h>
#include <model/User.hpp>
#include <cppcms/session_interface.h>

namespace svr{
namespace web{

class UserView : public cppcms::rpc::json_rpc_server{
public:
    UserView(cppcms::service &srv) : cppcms::rpc::json_rpc_server(srv) {
        bind("getAllUsers", cppcms::rpc::json_method(&UserView::getAllUsers, this),method_role);
    }

    void getAllUsers();

};

class UserController : public cppcms::application{

    void handle_create_get();
    void handle_create_post();

public:
    UserController(cppcms::service &svc): cppcms::application(svc){
        dispatcher().assign("/", &UserController::showAll, this);
        mapper().assign("showall", "/");

        dispatcher().assign("/show/(\\w+)", &UserController::show, this, 1);
        mapper().assign("show", "/show/{1}");

        dispatcher().assign("/create", &UserController::create, this);
        mapper().assign("create", "/create");

        attach(new UserView(svc), "user_ajaxview", "/ajax{1}", "/ajax(/(.*))?", 1);

    }

    void show(std::string userName);

    void create();

    void showAll();

};

}
}

