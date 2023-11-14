#pragma once

#include <ostream>
#include <string>
#include <set>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <vector>
#include "common/types.hpp"

namespace svr::common {

class formatter
{
public:
    formatter() = default;

    ~formatter() = default;

    template<typename Type>
    formatter &operator<<(const Type &value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const
    { return stream_.str(); }

    operator std::string() const
    { return stream_.str(); }

    explicit operator const char *() const
    { return stream_.str().c_str(); }

    enum ConvertToString
    {
        to_str
    };

    std::string operator>>(ConvertToString)
    { return stream_.str(); }

    formatter &operator=(formatter &) = delete;

    formatter(const formatter &) = delete;
private:
    std::stringstream stream_;
};

static const std::string dd_separator{".."};
static const std::string cm_separator{","};

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &v);

std::basic_ostream<char, std::char_traits<char>> &operator<<(std::basic_ostream<char, std::char_traits<char>> &os, const std::vector<uint8_t> &v);

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);

static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](const int c) {return !std::isspace(c);}));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const int c) {return !std::isspace(c);}).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

static inline std::string tolower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

static inline std::string toupper(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}

template<typename T>
std::string deep_to_string(const std::vector<T> &v)
{
    if (v.empty()) return "";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << i << ":" << v[i] << ", ";
    ss << (v.size() - 1) << ":" << v[v.size() - 1];

    return ss.str();
}

template<>
std::string deep_to_string(const std::vector<uint8_t> &v);

template<typename T> std::stringstream
deep_to_tsvs(const std::vector<T> &v, const char sep = '\t')
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &val: v) ss << val << sep;

    return ss;
}

template<> std::stringstream
deep_to_tsvs(const std::vector<uint8_t> &v, const char sep);

template<typename T> std::string
deep_to_tsv(const std::vector<T> &v, const char sep = '\t')
{
    return deep_to_tsvs(v, sep).str();
}


template <typename T> std::string
to_string_with_precision(const T v, const size_t digits_ct = std::numeric_limits<T>::max_digits10)
{
    std::ostringstream out;
    out.precision(digits_ct);
    if (!v || std::isnormal(v))
        out << std::fixed << v;
    else
        out << std::fixed << "'" << v << "'::numeric";
    return out.str();
}

template <typename T>
std::string to_string(const T v)
{
    return to_string_with_precision(v);
}

template <typename T> std::string
to_utf8(const std::basic_string<T, std::char_traits<T>, std::allocator<T>>& source)
{
    std::string result;

    std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
    result = convertor.to_bytes(source);

    return result;
}

template <typename T> void
from_utf8(const std::string& source, std::basic_string<T, std::char_traits<T>, std::allocator<T>>& result)
{
    std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
    result = convertor.from_bytes(source);
}

template<typename T, typename L> inline std::string
deep_to_string(const std::set<std::shared_ptr<T>, L> &v)
{
    if (v.size() == 0) return {};
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    auto v_iter = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++v_iter)
        ss << *v_iter->get() << ", ";
    ss << v.rbegin()->get();
    return ss.str();
}


template<typename T, typename L> std::string
deep_to_string(const std::set<T, L> &v)
{
    if (v.size() == 0) return {};
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    auto v_iter = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++v_iter)
        ss << *v_iter << ", ";
    ss << *v.rbegin();
    return ss.str();
}

void split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &str, const std::string &regex);

std::string gen_random(const size_t len);

std::vector<std::string> from_sql_array(const std::string &array_str);

std::string demangle(const char *mangled);

static inline bool ignoreCaseEquals(const std::string &lhs, const std::string &rhs)
{
    return tolower(lhs) == tolower(rhs);
}

std::string sanitize_db_table_name(std::string toBeIdentifier, char replaceChar = '_');

std::string make_md5_hash(const std::string &in);

std::string to_mt4_date(const bpt::ptime &time);

std::map<std::string, std::string> json_to_map(const std::string &json_str);

std::string map_to_json(const std::map<std::string, std::string> &value);

std::vector<size_t> parse_string_range(const std::string &parameter_string);

std::vector<std::string>
parse_string_range(const std::string &parameter_string, const std::vector<std::string> &set_parameters);

template<typename T>
T fromString(const std::string& s) {
    std::istringstream is(s);
    T t;
    is >> t;
    return t;
}

template<typename T>
std::string toString(const T& t) {
    return to_string<T>(t);
}


} // namespace svr
