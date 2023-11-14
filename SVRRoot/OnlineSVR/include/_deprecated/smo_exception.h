#ifndef SVR_SMO_EXCEPTION_H
#define SVR_SMO_EXCEPTION_H

#include <stdexcept>

namespace svr {
namespace smo {


class smo_error: public std::logic_error
{
protected:
public:
    using std::logic_error::logic_error;

    explicit smo_error(const std::string &message) : std::logic_error::logic_error(message) {}
    explicit smo_error(const char *message) : std::logic_error::logic_error(message) {}

    virtual ~smo_error() throw() {}
};


class smo_max_iter_error: public smo_error
{
public:
    explicit smo_max_iter_error(const std::string &message) : smo_error(message) {};
    explicit smo_max_iter_error(const char *message) : smo_error(message) {};

    virtual ~smo_max_iter_error() throw() {}
};


class smo_zero_active_error: public smo_error
{
public:
    explicit smo_zero_active_error(const std::string &message) : smo_error(message) {};
    explicit smo_zero_active_error(const char *message) : smo_error(message) {};

    virtual ~smo_zero_active_error() throw() {}
};


} // namespace svr {
} // namespace smo {

#endif //SVR_SMO_EXCEPTION_H
