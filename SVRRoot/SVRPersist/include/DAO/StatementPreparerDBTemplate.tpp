#ifndef STATEMENTPREPARERDBTEMPLATE_TPP
#define STATEMENTPREPARERDBTEMPLATE_TPP

#include "StatementPreparerDBTemplate.hpp"

namespace svr {
namespace dao {


template<typename T> inline std::string StatementPreparerDBTemplate::escape(T t)
{
    return boost::lexical_cast<std::string>(t);
}

template<typename InputIterator> inline std::deque<std::string> StatementPreparerDBTemplate::escape(InputIterator begin, InputIterator end)
{
    std::deque<std::string> r;
    for (; begin != end; ++begin) {
        std::string tmp = escape(*begin);
        size_t pos = tmp.find_first_of("'");
        if (pos != std::string::npos) {
            tmp.replace(pos, 1, "\"");
            pos = tmp.find_last_of("'");
            if (pos != std::string::npos)
                tmp.replace(pos, 1, "\"");
        }
        r.push_back(tmp);
    }
    return r;
}

template<typename T> inline std::string StatementPreparerDBTemplate::escape(const std::vector<T> &v)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    std::deque<std::string> vals = escape(begin(v), end(v));
    ss << "'{";
    if (!vals.empty()) ss << vals[0];
    for (size_t col_num = 1; col_num < vals.size(); col_num++)
        ss << ", " << vals[col_num];
    ss << "}'";

    return ss.str();
}

template<typename T> inline std::string StatementPreparerDBTemplate::escape(const std::deque<T> &v)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    std::deque<std::string> vals = escape(v.begin(), v.end());
    ss << "'{";
    if (!vals.empty()) ss << vals[0];
    for (size_t col_num = 1; col_num < vals.size(); col_num++)
        ss << ", " << vals[col_num];
    ss << "}'";

    return ss.str();
}

template<typename T> inline std::string StatementPreparerDBTemplate::escape(const std::set<T> &v)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    std::deque<std::string> vals = escape(v.begin(), v.end());
    ss << "'{";
    if (vals.size() > 0)
        ss << vals[0];
    for (size_t col_num = 1; col_num < vals.size(); ++col_num)
        ss << ", " << vals[col_num];
    ss << "}'";

    return ss.str();
}

template<typename T> std::string StatementPreparerDBTemplate::esc(const T &str)
{
    if (p_pg_con) return p_pg_con->esc(str);
    return common::pg_esc(str);
}

template<typename T, typename... Targs> std::string StatementPreparerDBTemplate::prepare_statement(const char *format, T value, Targs... Fargs)
{
    std::string s;
    for (; *format != '\0'; format++) {
        if (*format == '?') { // ? is the argument placeholder which will be replaced with concrete escaped value
            s += (std::string) (escape(value));
            s += (std::string) (prepare_statement(format + 1, Fargs...)); // recursive call
            return s;
        }
        s += *format;
    }
    return s;
}

template<typename T, typename... Targs> std::string StatementPreparerDBTemplate::prepare_statement(const std::string &format, T value, Targs... Fargs)
{
    return prepare_statement(format.c_str(), value, Fargs...);
}

}
}

#endif // STATEMENTPREPARERDBTEMPLATE_TPP
