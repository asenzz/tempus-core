#include <view/LoginView.hpp>
#include "controller/LoginController.hpp"
#include "AuthenticationProvider.hpp"
#include "appcontext.hpp"
#include <model/User.hpp>

using namespace svr::datamodel;
using namespace svr::context;

namespace svr {
namespace web {

void LoginController::login()
{
    LOG4_DEBUG("LoginController login " << request().request_method() << " request received.");

    if(request().request_method() == "POST"){
        handle_login_post();
    }else{
        logout();
        handle_login_get();
    }
    return;
}

void LoginController::logout() {
    LOG4_DEBUG("LoginController logout request received");
    session().clear();
}

void LoginController::handle_login_post() {
    LOG4_DEBUG("Login POST Request received");
    std::string username = request().post("username");
    std::string password = request().post("password");
    if(AppContext::get().authentication_provider.login(username, password)){
        User_ptr user = AppContext::get().user_service.get_user_by_user_name(username);
        session()["user"] = username;
        switch(user->get_role()){
            case svr::datamodel::ROLE::ADMIN:
                session()["role"] = "ADMIN";
                break;
            default:
                session()["role"] = "USER";
        }
        response().set_redirect_header("/web/");
    }else{
        session().clear();
        response().set_redirect_header("/web/login/");
    }
}

void LoginController::handle_login_get() {
    LOG4_DEBUG("Login GET Request received");
    content::Login login;
    login.pageTitle = "Login";
    render("Login", login);
}
}
}