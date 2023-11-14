#pragma once

#include "UserService.hpp"
#include "AuthenticationProvider.hpp"

namespace svr {
namespace business {

class LocalAuthenticationProvider : public AuthenticationProvider{

    svr::business::UserService & user_service;
public:
    LocalAuthenticationProvider(svr::business::UserService &user_service) : user_service(user_service){}

    virtual bool login(const std::string &user_name, const std::string &password) override;
};

}
}

using LocalAuthenticationProvider_ptr = std::shared_ptr<svr::business::LocalAuthenticationProvider>;