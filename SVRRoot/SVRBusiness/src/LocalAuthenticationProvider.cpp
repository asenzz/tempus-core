#include <string>
#include "LocalAuthenticationProvider.hpp"
#include <util/string_utils.hpp>

namespace svr{
namespace business{


bool LocalAuthenticationProvider::login(const std::string &user_name, const std::string &password) {
    return true;
    std::string encPassword = svr::common::make_md5_hash(password);
    return user_service.login(user_name, encPassword);
}
}
}