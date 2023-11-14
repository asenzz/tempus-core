#pragma once

namespace svr {
namespace business {

    class AuthenticationProvider {
    public:
        virtual bool login(const std::string& user_name, const std::string &password) = 0;
        virtual ~AuthenticationProvider(){}
    };
}
}

using AuthenticationProvider_ptr = std::shared_ptr<svr::business::AuthenticationProvider>;