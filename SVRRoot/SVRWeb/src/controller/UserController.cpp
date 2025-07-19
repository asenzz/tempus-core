#include <chrono>
#include <thread>
#include "controller/UserController.hpp"
#include "view/UserView.hpp"
#include "appcontext.hpp"
#include "controller/MainController.hpp"

using namespace svr::datamodel;
using namespace svr::context;

namespace svr {
namespace web {
void UserController::show(std::string user_name) {

    content::User user;
    user.pageTitle = "User Details";

    User_ptr user_ = AppContext::get().user_service.get_user_by_user_name(user_name);

    if(user_.get() == nullptr){
        user.pageError = "No user with such id exists!";
    }else{
        user.object = user_;
    }

    render("ShowUser", user);
}

void UserController::create() {

    if(request().request_method() == "POST"){
        handle_create_post();
    } else {
        handle_create_get();
    }
}

void UserController::showAll() {

    content::Main main;
    main.pageTitle = "Users";
    render("Users", main);
}

void UserView::getAllUsers() {
    return_result(AppContext::get().user_service.get_all_users());
}

void UserController::handle_create_get() {
    content::UserWithForm user;
    user.pageTitle = "Create User";
    render("CreateUser", user);
}

void UserController::handle_create_post() {
    content::UserWithForm user;
    user.form.load(context());

    if(user.form.validate()){
        user.load_form_data();
        if(AppContext::get().user_service.exists(user.object->get_user_name())){
            user.form.username.error_message("The username is already taken!");
            user.form.username.valid(false);
        }
        else if(AppContext::get().user_service.save(user.object) == 0){
            user.pageError = "Cannot save user! Please try again later or contact the administrator.";
        } else{
            LOG4_DEBUG("Created successfully user with id " << user.object->get_id());
            std::stringstream url;
            mapper().map(url, "/user/show", user.object->get_user_name());
            response().set_redirect_header(url.str());
        }
    } else{
        user.pageTitle = "Create User";
        render("CreateUser", user);
    }
}
}
}