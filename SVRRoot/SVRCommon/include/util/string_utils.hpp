#pragma once

#include <cstddef>
#include <execution>
#include <ostream>
#include <string>
#include <set>
#include <oneapi/tbb/concurrent_set.h>
#include <algorithm>
#include <locale>
#include <codecvt>
#include <vector>
#include "common/types.hpp"
#include "common/compatibility.hpp"

namespace svr {

#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)

constexpr char C_dd_separator[] = "..";
constexpr char C_cm_separator[] = ",";

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &v);

std::basic_ostream<char, std::char_traits<char>> &operator<<(std::basic_ostream<char, std::char_traits<char>> &os, const std::vector<uint8_t> &v);

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);

template<typename T, typename C> std::basic_ostream<C> &operator<<(std::basic_ostream<C> &r, const std::complex<T> &v)
{
    return r << std::real<T>(v) << "+" << std::imag<T>(v) << "i ";
}

namespace common {

char *concat(const char *lhs, const char *rhs);

class formatter {
    enum ConvertToString {
        to_str
    };

    std::stringstream stream_;

public:
    formatter() = default;

    ~formatter() = default;

    template<typename Type> formatter &operator<<(const Type &value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const;

    operator std::string() const;

    explicit operator const char *() const;

    std::string operator>>(ConvertToString);

    formatter &operator=(formatter &) = delete;

    formatter(const formatter &) = delete;
};

std::string &ltrim(std::string &s);

// trim from end
std::string &rtrim(std::string &s);

// trim from both ends
std::string &trim(std::string &s);

std::string tolower(std::string str);

std::string toupper(std::string str);

std::string &lowertrim(std::string &s);

constexpr inline char ctoupper(const char c)
{
    return (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c;
}

template<const size_t N> constexpr auto ctoupper(const char (&input)[N])
{
    std::string result(N - 1, '\0');
#ifdef __clang__
#pragma unroll N - 1
#endif
    for (size_t i = 0; i < result.size(); ++i) result[i] = ctoupper(input[i]);
    return result;
}

#define CTOUPPER(X) common::ctoupper<ARRAYLEN(X)>(X)

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

template<typename T> std::stringstream
to_stringstream(const T *v, const size_t l)
{
    std::stringstream s;
    if (l < 2) {
        s << *v;
        goto __bail;
    }
    for (size_t i = 0; i < l - 1; ++i) s << v[i] << ", ";
    s << v[l - 1];
    __bail:
    return s;
}

template<typename T> inline std::string
to_string(const T *v, const size_t l)
{
    return to_stringstream(v, l).str();
}

template<typename T> inline std::string to_string(const arma::Mat<T> &v, const size_t start_i, const size_t n)
{
    if (v.is_empty()) return "empty";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    const auto last_i = start_i + n - 1;
    for (size_t i = start_i; i < last_i; ++i) ss << i << ": " << v(i) << ", ";
    ss << last_i << ": " << v(last_i);

    return ss.str();
}

template<typename T> inline std::string to_string(const arma::Mat<T> &v, const size_t limit)
{
    return to_string(v, 0, std::min<size_t>(v.n_elem, limit));
}

template<typename T> inline std::string to_string(const std::vector<T> &v)
{
    if (v.empty()) return "empty";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << i << ": " << v[i] << ", ";
    ss << (v.size() - 1) << ": " << v.back();

    return ss.str();
}

template<typename Tx, typename Ty> inline std::string to_string(const std::pair<Tx, Ty> &p)
{
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    s << "first: " << p.first << ", second: " << p.second;
    return s.str();
}

template<typename T> inline std::string to_string(const std::deque<T> &v)
{
    if (v.empty()) return "empty";

    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) s << i << ": " << v[i] << ", ";
    s << v.size() - 1 << ": " << v.back();

    return s.str();
}

template<>
std::string to_string(const std::vector<uint8_t> &v);

template<typename T> inline std::stringstream
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
to_string(const std::deque<std::shared_ptr<T>> &v)
{
    if (v.empty()) return "empty";
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    auto vit = v.begin();
    for (size_t i = 0; i < v.size() - 1; ++i, ++vit)
        s << **vit << ", ";
    s << **vit;
    return s.str();
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

std::string to_mql_date(const bpt::ptime &time);

std::map<std::string, std::string> json_to_map(const std::string &json_str);

std::string map_to_json(const std::map<std::string, std::string> &value);

std::vector<size_t> parse_string_range(const std::string &parameter_string);

std::vector<std::string>
parse_string_range(const std::string &parameter_string, const std::vector<std::string> &set_parameters);

template<typename T> T inline from_string(const std::string &s)
{
    std::istringstream is(s);
    T t;
    is >> t;
    return t;
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true> inline std::string to_string(const T v)
{
    std::ostringstream out;
    out.precision(std::numeric_limits<T>::max_digits10);
    out << v;
    return out.str();
}

} // namespace common
} // namespace svr

namespace std {

template<typename T> std::basic_ostream<char> &
operator<<(std::basic_ostream<char> &s, const std::deque<std::shared_ptr<T>> &v)
{
    return s << svr::common::to_string(v);
}

template<typename C, typename Tr, typename T, typename Less> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<std::shared_ptr<T>, Less> &aset)
{
    if (aset.size() < 2) return s << *aset.begin();
    for_each(aset.begin(), std::next(aset.rbegin()).base(), [&s](const auto &el) { s << *el << ", "; });
    return s << **std::prev(aset.end());
}

template<typename C, typename Tr, typename T> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<std::shared_ptr<T>> &aset)
{
    if (aset.size() < 2) return s << *aset.begin();
    for_each(aset.begin(), std::next(aset.rbegin()).base(), [&s](const auto &el) { s << *el << ", "; });
    return s << **std::prev(aset.end());
}

template<typename T, typename C, typename Tr> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<T> &aset)
{
    if (aset.size() < 2) return s << *aset.begin();
    for_each(aset.begin(), std::next(aset.rbegin()).base(), [&s](const auto &el) { s << el << ", "; });
    return s << *std::prev(aset.end());
}

template<typename T, typename C, typename Ca> std::basic_ostream<C, Ca> &
operator<<(std::basic_ostream<C, Ca> &s, const std::deque<T> &v)
{
    return s << svr::common::to_string(v);
}

#if 0
template<typename Tx, typename Ty, typename C, typename Ca> std::basic_ostream<C, Ca> &
operator<<(std::basic_ostream<C, Ca> &s, const std::pair<Tx, Ty> &p)
{
    return s << svr::common::to_string(p);
}
#endif
namespace {
template<typename TupleT, std::size_t... Is>
std::ostream &printTupleImp(std::ostream &os, const TupleT &tp, std::index_sequence<Is...>)
{
    auto printElem = [&os](const auto &x, size_t id) {
        if (id > 0)
            os << ", ";
        os << id << ": " << x;
    };
    (printElem(std::get<Is>(tp), Is), ...);
    return os;
}
}

template<typename TupleT, std::size_t TupSize = std::tuple_size<TupleT>::value>
std::ostream &operator<<(std::ostream &os, const TupleT &tp)
{
    return printTupleImp(os, tp, std::make_index_sequence<TupSize>{});
}

}
