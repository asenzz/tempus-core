#include "view/LoginView.hpp"
#include "controller/ConnectorLoginController.hpp"
#include "AuthenticationProvider.hpp"
#include "UserService.hpp"
#include "appcontext.hpp"
#include "model/User.hpp"

namespace svr {
namespace web {
void ConnectorLoginController::login()
{
    LOG4_BEGIN();

    if (request().request_method() == "POST") {
        handle_login_post();
    } else {
        logout();
        handle_login_get();
    }

    LOG4_END();
}

void ConnectorLoginController::logout()
{
    LOG4_DEBUG("Login logout request received");
    session().clear();
}

void ConnectorLoginController::handle_login_post()
{
    const auto username = request().post("username");
    const auto password = request().post("password");
    LOG4_DEBUG("Login POST Request received " << username << ":" << password);
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
    } else
        response().status(cppcms::http::response::unauthorized);
}

void ConnectorLoginController::handle_login_get()
{
    LOG4_DEBUG("Login GET Request received");
}
}
}
