#include "controller/MainController.hpp"

namespace svr{
namespace web{

void MainController::index()
{
    LOG4_BEGIN();
    content::Main data;
    data.pageTitle = "Welcome to SVR, " + session()["user"];
    render("Index", data);
}

bool MainController::logged_in()
{
    LOG4_BEGIN();
    return true;
    return session().is_set("user") && session()["user"] != "";
}

void MainController::main(std::string url)
{
    LOG4_TRACE("Login interceptor checks: " << request().path_info());

    if(request().path_info() != loginPath && !logged_in()){
        LOG4_TRACE("Request for url: " << rootPath + request().path_info() << " rejected: Not Logged in.");
        response().set_redirect_header(loginUrl);
    } else {
        application::main(url);
    }
}
}
}