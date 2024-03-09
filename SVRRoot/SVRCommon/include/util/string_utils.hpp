#pragma once

#include <cstddef>
#include <execution>
#include <ostream>
#include <string>
#include <set>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <vector>
#include "common/types.hpp"
#include "common/compatibility.hpp"

namespace svr {


static const std::string C_dd_separator{".."};
static const std::string C_cm_separator{","};

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &v);

std::basic_ostream<char, std::char_traits<char>> &operator<<(std::basic_ostream<char, std::char_traits<char>> &os, const std::vector<uint8_t> &v);

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);


template<typename T, typename C> std::basic_ostream<C> &operator<<(std::basic_ostream<C> &r, const std::complex<T> &v)
{
    r << std::real<T>(v) << "+" << std::imag<T>(v) << "i ";
    return r;
}


namespace common {

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
    {
        const std::string &s = stream_.str();
        return s.c_str();
    }

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


static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](const int c) { return !std::isspace(c); }));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

static inline std::string tolower(std::string str)
{
    std::transform(std::execution::par_unseq, str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

static inline std::string toupper(std::string str)
{
    std::transform(std::execution::par_unseq, str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}


template<typename T> std::string
to_binary_string(const std::set<T> &values)
{
    using t_chr = typename std::string::value_type;
    constexpr size_t t_chr_ct = sizeof(size_t) / sizeof(t_chr);
    std::string res;
    res.resize(values.size() * t_chr_ct);
#pragma omp parallel for num_threads(values.size()) collapse(2)
    for (size_t i = 0; i < values.size(); ++i)
        for (size_t j = 0; j < t_chr_ct; ++j)
            res[i * t_chr_ct + j] = reinterpret_cast<const char *>(&*std::next(values.begin(), i))[j];
    return res;
}

template<typename T>
std::string to_string(const std::vector<T> &v)
{
    if (v.empty()) return "";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << i << ":" << v[i] << ", ";
    ss << (v.size() - 1) << ":" << v[v.size() - 1];

    return ss.str();
}


template<typename T>
std::string to_string(const std::deque<T> &v)
{
    if (v.empty()) return "";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << i << ":" << v[i] << ", ";
    ss << (v.size() - 1) << ":" << v[v.size() - 1];

    return ss.str();
}

template<>
std::string to_string(const std::vector<uint8_t> &v);

template<typename T> std::stringstream
to_tsvs(const std::vector<T> &v, const char sep = '\t')
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &val: v) ss << val << sep;

    return ss;
}

template<> std::stringstream
to_tsvs(const std::vector<uint8_t> &v, const char sep);

template<typename T> std::string
to_tsv(const std::vector<T> &v, const char sep = '\t')
{
    return to_tsvs(v, sep).str();
}


template<typename T> std::string
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

template<typename T> std::string
to_utf8(const std::basic_string<T, std::char_traits<T>, std::allocator<T>> &source)
{
    std::string result;

    std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
    result = convertor.to_bytes(source);

    return result;
}

template<typename T> void
from_utf8(const std::string &source, std::basic_string<T, std::char_traits<T>, std::allocator<T>> &result)
{
    std::wstring_convert<std::codecvt_utf8_utf16<T>, T> convertor;
    result = convertor.from_bytes(source);
}

template<typename T, typename L> inline std::string
to_string(const std::set<std::shared_ptr<T>, L> &v)
{
    if (v.empty()) return {};
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << **vit << ", ";
    s << **vit;
    return s.str();
}


template<typename T, typename L> inline std::string
to_string(const std::set<T, L> &v)
{
    if (v.empty()) return {};
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << *vit << ", ";
    s << *vit;
    return s.str();
}

template<typename T, typename L> inline std::string
to_string(const tbb::concurrent_set<std::shared_ptr<T>, L> &v)
{
    if (v.empty()) return {};
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << **vit << ", ";
    s << **vit;
    return s.str();
}


template<typename T, typename L> inline std::string
to_string(const tbb::concurrent_set<T, L> &v)
{
    if (v.empty()) return {};
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << *vit << ", ";
    s << *vit;
    return s.str();
}

template<typename T> inline std::string
to_string(const tbb::concurrent_vector<T> &v)
{
    if (v.empty()) return {};
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << *vit << ", ";
    s << *vit;
    return s.str();
}

template<typename T> inline std::string
to_string(const T v)
{
    return to_string_with_precision(v);
}

}

template<typename T, typename C>
std::basic_ostream<C> &operator<<(std::basic_ostream<C> &s, const tbb::concurrent_vector<T> &v)
{
    return s << common::to_string(v);
}

namespace common {

void split(const std::string &s, char delim, std::vector<std::string> &elems);

std::deque<std::string> split(const std::string &str, const std::string &regex);

std::string gen_random(const size_t len);

std::deque<std::string> from_sql_array(const std::string &array_str);

std::string demangle(const char *mangled);

static inline bool ignore_case_equals(const std::string &lhs, const std::string &rhs)
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

template<typename T> T
from_string(const std::string &s)
{
    std::istringstream is(s);
    T t;
    is >> t;
    return t;
}

template<typename T>
std::string to_string(const T &t)
{
    return to_string<T>(t);
}


} // namespace svr
}