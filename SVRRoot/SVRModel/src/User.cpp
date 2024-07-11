#include <model/User.hpp>
#include <common/logging.hpp>

namespace svr {
namespace datamodel {

void User::init_id()
{
    if (!id) {
        boost::hash_combine(id, user_name);
        boost::hash_combine(id, email);
        boost::hash_combine(id, name);
        boost::hash_combine(id, password);
        boost::hash_combine(id, role);
    }
}

bool User::operator==(User const &other) const
{
    return user_name == other.user_name
           && email == other.email
           && name == other.name
           && password == other.password
           && role == other.role;
}

User::User(const bigint user_id,
           const std::string &username,
           const std::string &email,
           const std::string &password,
           const std::string &name,
           const ROLE &_role,
           Priority _priority)
        : Entity(user_id),
          user_name(username),
          email(email),
          name(name),
          password(password),
          role(_role),
          priority(_priority)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    LOG4_TRACE("User " << username << " created.");
}


void User::set_email(const std::string &email_)
{ this->email = email_; }

void User::set_name(const std::string &name_)
{ this->name = name_; }

void User::set_role(ROLE role_)
{ this->role = role_; }

void User::set_password(const std::string &password_)
{ this->password = password_; }

void User::set_user_name(const std::string &userName)
{ user_name = userName; }

std::string User::get_name() const
{ return name; }

std::string User::get_email() const
{ return email; }

std::string User::get_user_name() const
{ return user_name; }

datamodel::ROLE User::get_role() const
{ return role; }

std::string User::get_password() const
{ return password; }

Priority const &User::get_priority() const
{ return priority; }

void User::set_priority(Priority const &priority_)
{ this->priority = priority_; }

std::string User::to_string() const
{
    std::stringstream ss;
    ss << "User " << get_id()
       << ", name " << get_user_name()
       << ", email " << get_email()
       << ", password " << get_password()
       << ", name " << get_name()
       << ", role " << (get_role() == ROLE::ADMIN ? "Admin" : "User")
       << ", priority " << datamodel::to_string(get_priority());
    return ss.str();
}

std::string User::to_json_string() const
{
    std::stringstream ss;
    ss << "User={EntityId:" << get_id()
       << ",UserName:" << get_user_name()
       << ",email:" << get_email()
       << ",password:" << get_password()
       << ",name:" << get_name()
       << ",role:" << (get_role() == ROLE::ADMIN ? "Admin" : "User")
       << ",priority: " << svr::datamodel::to_string(get_priority())
       << "}";
    return ss.str();
}


}
}



