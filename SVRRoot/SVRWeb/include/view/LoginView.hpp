#pragma once


#include <cppcms/view.h>
#include <cppcms/form.h>

#include <view/MainView.hpp>

namespace content{

struct LoginForm : cppcms::form{

    cppcms::widgets::text       username;
    cppcms::widgets::password   password;
    cppcms::widgets::submit     submit;

    LoginForm(){

        username.name("username");
        username.message("Your username:");
        username.limits(6, 20);
        username.id("username");
        add(username);

        password.message("Password:");
        password.name("password");
        password.id("password");
        password.limits(6, 15);
        add(password);

        submit.id("login");
        submit.name("login");
        submit.value("Login");
        add(submit);
    }

};

struct Login : public Main{
    LoginForm form;
};

}