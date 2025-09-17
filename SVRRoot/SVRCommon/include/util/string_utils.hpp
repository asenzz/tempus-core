#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <set>
#include <oneapi/tbb/concurrent_set.h>
#include <vector>
#include "common/compatibility.hpp"

namespace svr {
#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)

constexpr char C_dd_separator[] = "..";
constexpr char C_cm_separator[] = ",";

std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &v);

std::basic_ostream<char, std::char_traits<char> > &operator<<(std::basic_ostream<char, std::char_traits<char> > &os, const std::vector<uint8_t> &v);

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);

template<typename T, typename C> std::basic_ostream<C> &operator<<(std::basic_ostream<C> &r, const std::complex<T> &v);

namespace common {
char *concat(const char *lhs, const char *rhs);

class formatter
{
    enum ConvertToString
    {
        to_str
    };

    std::stringstream stream_;

public:
    formatter() = default;

    ~formatter() = default;

    template<typename Type> formatter &operator<<(const Type &value);

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

constexpr inline char ctoupper(const char c);

template<const size_t N> constexpr std::string ctoupper(const char *const input);

#define CTOUPPER(X) common::ctoupper<ARRAYLEN(X)>(X)

template<typename T> std::string to_binary_string(const std::set<T> &values);

template<typename T> std::stringstream to_stringstream(const T *v, const size_t l);

template<typename T> inline std::string to_string(const T *v, const size_t l);

template<typename T> inline std::string to_string(const arma::Mat<T> &v, const size_t start_i, const size_t n);

template<typename T> inline std::string to_string(const arma::Mat<T> &v, const size_t limit);

template<typename T> inline std::string to_string(const std::vector<T> &v);

template<typename Tx, typename Ty> inline std::string to_string(const std::pair<Tx, Ty> &p);

template<typename T> inline std::string to_string(const std::deque<T> &v);

template<>
std::string to_string(const std::vector<uint8_t> &v);

template<typename T> inline std::stringstream to_tsvs(const std::vector<T> &v, const char sep = '\t');

template<> std::stringstream to_tsvs(const std::vector<uint8_t> &v, const char sep);

template<typename T> std::string to_tsv(const std::vector<T> &v, const char sep = '\t');

template<typename T> std::string to_string_with_precision(const T v, const size_t digits_ct = std::numeric_limits<T>::max_digits10);

template<typename T> std::string to_utf8(const std::basic_string<T, std::char_traits<T>, std::allocator<T> > &source);

template<typename T> void from_utf8(const std::string &source, std::basic_string<T, std::char_traits<T>, std::allocator<T> > &result);

template<typename T, typename L> inline std::string to_string(const std::set<std::shared_ptr<T>, L> &v);

template<typename T, typename L> inline std::string to_string(const std::set<T, L> &v);

template<typename T, typename L> inline std::string to_string(const tbb::concurrent_set<std::shared_ptr<T>, L> &v);

template<typename T, typename L> inline std::string to_string(const tbb::concurrent_set<T, L> &v);

template<typename T> inline std::string to_string(const tbb::concurrent_vector<T> &v);

template<typename T> inline std::string to_string(const std::deque<std::shared_ptr<T> > &v);
}

template<typename T, typename C> std::basic_ostream<C> &operator<<(std::basic_ostream<C> &s, const tbb::concurrent_vector<T> &v);

namespace common {
void split(const std::string &s, char delim, std::vector<std::string> &elems);

std::deque<std::string> split(const std::string &str, const std::string &regex);

std::string gen_random(const size_t len);

std::deque<std::string> from_sql_array(const std::string &array_str);

std::string demangle(const char *mangled);

static inline bool ignore_case_equals(const std::string &lhs, const std::string &rhs);

std::string sanitize_db_table_name(std::string name, char replace_char = '_');

std::string make_md5_hash(const std::string &in);

std::string to_mql_date(const bpt::ptime &time);

std::map<std::string, std::string> json_to_map(const std::string &json_str);

std::string map_to_json(const std::map<std::string, std::string> &value);

std::vector<size_t> parse_string_range(const std::string &parameter_string);

std::vector<std::string>
parse_string_range(const std::string &parameter_string, const std::vector<std::string> &set_parameters);

template<typename T> T inline from_string(const std::string &s);

template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool>  = true> inline std::string to_string(const T v);

std::string pg_esc(const std::string &input);

std::string pg_esc(const char input[]);

} // namespace common
} // namespace svr

namespace std {
template<typename T> std::basic_ostream<char> &operator<<(std::basic_ostream<char> &s, const std::deque<std::shared_ptr<T> > &v);

template<typename C, typename Tr, typename T, typename Less> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<std::shared_ptr<T>, Less> &aset);

template<typename C, typename Tr, typename T> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<std::shared_ptr<T> > &aset);

template<typename T, typename C, typename Tr> std::basic_ostream<C, Tr> &operator<<(std::basic_ostream<C, Tr> &s, const std::set<T> &aset);

template<typename T, typename C, typename Ca> std::basic_ostream<C, Ca> &operator<<(std::basic_ostream<C, Ca> &s, const std::deque<T> &v);

template<typename TupleT, std::size_t TupSize = std::tuple_size<TupleT>::value> std::ostream &operator<<(std::ostream &os, const TupleT &tp);
}

#include "string_utils.tpp"

#endif // #define STRING_UTILS_HPP
