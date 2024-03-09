#pragma once

#include <cppcms/view.h>
#include <cppcms/form.h>
#include <cppcms/json.h>

#include <view/MainView.hpp>
#include "model/User.hpp"

namespace content {

struct UserForm : cppcms::form
{

    cppcms::widgets::text username;
    cppcms::widgets::text name;
    cppcms::widgets::password password;
    cppcms::widgets::password password2;
    cppcms::widgets::email email;
    cppcms::widgets::radio role;
    cppcms::widgets::submit submit;

    UserForm()
    {

        username.name("username");
        username.message("Your username");
        username.limits(6, 20);
        username.id("username");
        username.error_message("allowed size is 6 - 20 chars");
        add(username);

        name.name("name");
        name.message("Your name");
        name.limits(6, 20);
        name.id("name");
        name.error_message("allowed size is 6 - 20 chars");
        add(name);

        password.message("Password");
        password.name("password");
        password.id("password");
        password.limits(6, 15);
        password.error_message("Password doesn't meet required complexity");
        add(password);

        password2.message("Password (again)");
        password2.name("password2");
        password2.id("password2");
        password2.limits(6, 15);
        password.error_message("Password doesn't meet required complexity");
        add(password2);

        email.message("E-Mail");
        email.name("email");
        email.id("email");
        email.error_message("Invalid e-mail");
        add(email);

        role.add("Admin", "ADMIN");
        role.add("User", "USER");
        role.name("role");
        role.id("role");
        role.message("System role");
        add(role);

        submit.id("submit");
        submit.name("submit");
        submit.value("Save");
        add(submit);
    }

    virtual bool validate() override
    {
        if (!form::validate()) {
            return false;
        }
        if (password.value() != password2.value()) {
            password2.valid(false);
            password2.error_message("Passwords mismatch!");
            return false;
        }
        if (role.selected() == -1) {
            role.valid(false);
            role.error_message("Please select role");
            return false;
        }
        return true;
    }

};

struct UserWithForm : public Main
{
    User_ptr object;
    UserForm form;

    void load_form_data()
    {
        object = svr::ptr<svr::datamodel::User>();
        object->set_user_name(form.username.value());
        object->set_name(form.name.value());
        object->set_email(form.email.value());
        object->set_password(form.password.value());
        object->set_role(form.role.selected_id() == "ADMIN" ? svr::datamodel::ROLE::ADMIN : svr::datamodel::ROLE::USER);
    }
};

struct User : public Main
{
    User_ptr object;
};

}


namespace cppcms {
namespace json {

// We specilize cppcms::json::traits structure to convert
// objects to and from json values

template<>
struct traits<User_ptr>
{
    static void set(value &v, User_ptr const &in)
    {
        v.set("id", in->get_id());
        v.set("user_name", in->get_user_name());
        v.set("email", in->get_email());
        v.set("name", in->get_name());
    }
};
} // json
} // cppcms