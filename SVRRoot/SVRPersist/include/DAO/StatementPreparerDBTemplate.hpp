#pragma once

#include <set>
#include "common.hpp"
#include "model/User.hpp"
#include "model/Dataset.hpp"
#include "onlinesvr_persist.tpp"


namespace bpt = boost::posix_time;

namespace svr::dao{

class StatementPreparerDBTemplate
{
    pqxx::connection connection;

    std::string escape(std::nullptr_t);

    std::string escape(const std::shared_ptr<svr::OnlineMIMOSVR> &model);

    std::string escape(const svr::datamodel::Priority &priority);

    std::string escape(const bpt::ptime &);

    std::string escape(const bpt::time_duration &);

    std::string escape(const std::string &);

    std::string escape(const char *s);

    inline static std::string escape(char c)
    {
        std::string result;
        result += "'";
        result += c;
        result += "'";
        return result;
    }

    std::string escape(const bool &);

    static inline std::string escape(const long v)
    { return std::to_string(v); }

    static inline std::string escape(const bigint v)
    { return std::to_string(v); }

    static inline std::string escape(const float v)
    { return common::to_string_with_precision(v); }

    static inline std::string escape(const double v)
    { return common::to_string_with_precision(v); }

    std::string escape(svr::datamodel::ROLE role);

    template<typename T>
    inline std::string escape(T t)
    {
        return boost::lexical_cast<std::string>(t);
    }

    template<typename InputIterator>
    inline std::deque<std::string> escape(InputIterator begin, InputIterator end)
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

    template<typename T>
    inline std::string escape(const std::vector<T> &v)
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

    template<typename T>
    inline std::string escape(const std::deque<T> &v)
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

    template<typename T>
    inline std::string escape(const std::set<T> &v)
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

    std::string prepareStatement(const char *format);

public:

    explicit StatementPreparerDBTemplate(const std::string &connection_string) : connection(connection_string) {}

//    virtual ~StatementPreparerDBTemplate() {}


    template<typename T, typename... Targs>
    std::string prepareStatement(const char *format, T value, Targs... Fargs)
    {
        std::string s;
        for (; *format != '\0'; format++) {
            if (*format == '?') { // ? is the argument placeholder which will be replaced with concrete escaped value
                s += (std::string) (escape(value));
                s += (std::string) (prepareStatement(format + 1, Fargs...)); // recursive call
                return s;
            }
            s += *format;
        }
        return s;
    }

    template<typename T, typename... Targs>
    std::string prepareStatement(const std::string &format, T value, Targs... Fargs)
    {
        return prepareStatement(format.c_str(), value, Fargs...);
    }

    static std::string prepareStatement(const std::string &format)
    {
        return format;
    }
};

} /* namespace svr::dao */

