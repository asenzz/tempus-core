#include "view/LoginView.hpp"
#include "controller/LoginController.hpp"
#include "UserService.hpp"
#include "appcontext.hpp"
#include "AuthenticationProvider.hpp"
#include "model/User.hpp"

namespace svr {
namespace web {
void LoginController::login()
{
    LOG4_DEBUG("Login " << request().request_method() << " request received.");
    if (request().request_method() == "POST") {
        handle_login_post();
    } else {
        logout();
        handle_login_get();
    }
    LOG4_END();
}

void LoginController::logout()
{
    LOG4_DEBUG("Logout request received");
    session().clear();
}

void LoginController::handle_login_post()
{
    LOG4_DEBUG("Login POST Request received");
    const auto username = request().post("username");
    const auto password = request().post("password");
    if (APP.authentication_provider.login(username, password)) {
        const auto user = APP.user_service.get_user_by_user_name(username);
        session()["user"] = username;
        switch (user->get_role()) {
            case datamodel::ROLE::ADMIN:
                session()["role"] = "ADMIN";
                break;
            default:
                session()["role"] = "USER";
        }
        response().set_redirect_header("/web/");
    } else {
        session().clear();
        response().set_redirect_header("/web/login/");
    }
}

void LoginController::handle_login_get()
{
    LOG4_DEBUG("Login GET Request received");
    content::Login login;
    login.pageTitle = "Login";
    render("Login", login);
}
}
}
