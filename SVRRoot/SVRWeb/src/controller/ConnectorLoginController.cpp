#include <view/LoginView.hpp>
#include "controller/ConnectorLoginController.hpp"
#include "appcontext.hpp"
#include <model/User.hpp>

using namespace svr::datamodel;
using namespace svr::context;

namespace svr{
namespace web {

void ConnectorLoginController::login() {
    LOG4_BEGIN();

    if(request().request_method() == "POST"){
        handle_login_post();
    }else{
        logout();
        handle_login_get();
    }
    return;
}

void ConnectorLoginController::logout() {
    LOG4_DEBUG("MT4Login logout request received");
    session().clear();
}

void ConnectorLoginController::handle_login_post()
{
    std::string username = request().post("username");
    std::string password = request().post("password");
    LOG4_DEBUG("MT4Login POST Request received " << username << ":" << password);
    if(AppContext::get().authentication_provider.login(username, password)) {
        User_ptr user = AppContext::get().user_service.get_user_by_user_name(username);
        session()["user"] = username;
        switch (user->get_role()) {
            case svr::datamodel::ROLE::ADMIN:
                session()["role"] = "ADMIN";
                break;
            default:
                session()["role"] = "USER";
        }
    } else {
        response().status(cppcms::http::response::unauthorized);
    }
}

void ConnectorLoginController::handle_login_get() {
    LOG4_DEBUG("MT4Login GET Request received");
}
}
}
