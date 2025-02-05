#include <boost/date_time/posix_time/ptime.hpp>
#include <cxxabi.h>
#include <iterator>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <openssl/evp.h>

#include "util/string_utils.hpp"
#include "common/logging.hpp"
#include "util/math_utils.hpp"
#include <common/constants.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace svr {

namespace common {

char* concat(const char *lhs, const char *rhs)
{
    // Calculate the lengths of the two strings
    const auto lhs_len = strlen(lhs);
    const auto rhs_len = strlen(rhs);

    // Allocate memory for the concatenated string (+1 for null terminator)
    auto result = new char[lhs_len + rhs_len + 1];

    // Copy the first string
    strcpy(result, lhs);

    // Concatenate the second string
    strcat(result, rhs);

    return result; // Return the dynamically allocated result
}

std::string formatter::str() const
{
    return stream_.str();
}

formatter::operator std::string() const
{
    return stream_.str();
}

formatter::operator const char *() const
{
    const std::string &s = stream_.str();
    return s.c_str();
}

std::string formatter::operator>>(ConvertToString)
{
    return stream_.str();
}

std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](const int c) { return !std::isspace(c); }));
    return s;
}

// trim from end
std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const int c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

// trim from both ends
std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

std::string tolower(std::string str)
{
    std::transform(std::execution::par_unseq, str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

std::string toupper(std::string str)
{
    std::transform(std::execution::par_unseq, str.begin(), str.end(), str.begin(), ::toupper);
    return str;
}


template<>
std::string to_string(const std::vector<uint8_t> &v)
{
    if (v.empty()) return "";

    std::stringstream ss;
    //ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << i << ":" << int(v[i]) << ", ";
    ss << (v.size() - 1) << ":" << int(v[v.size() - 1]);

    return ss.str();
}

template<> std::stringstream
to_tsvs(const std::vector<uint8_t> &v, const char sep)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &val: v) ss << val << sep;

    return ss;
}

std::ostream &
operator<<(std::ostream &os, const std::vector<size_t> &v)
{
    for (auto v_iter = v.begin(); v_iter != std::prev(v.end()); ++v_iter) os << *v_iter << C_cm_separator;
    os << *std::prev(v.end());
    return os;
}

std::basic_ostream<char, std::char_traits<char>> &
operator<<(std::basic_ostream<char, std::char_traits<char>> &os, const std::vector<uint8_t> &v)
{
    for (auto v_iter = v.begin(); v_iter != std::prev(v.end()); ++v_iter) os << *v_iter << C_cm_separator;
    os << *std::prev(v.end());
    return os;
}

std::ostream &
operator<<(std::ostream &os, const std::vector<double> &v)
{
    os.precision(std::numeric_limits< double >::max_digits10);
    for (auto v_iter = v.begin(); v_iter != std::prev(v.end()); ++v_iter) os << *v_iter << C_cm_separator;
    os << *std::prev(v.end());
    return os;
}

std::deque<std::string> from_sql_array(const std::string &array_str)
{
    return split(array_str, ", {}");
}

std::string gen_random(const size_t len)
{
    constexpr char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
    std::string tmp_s;
    tmp_s.reserve(len);
    for (size_t i = 0; i < len; ++i) tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    return tmp_s;
}

void split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
        if (not item.empty())
            elems.push_back(item);
}

std::deque<std::string> split(const std::string &str, const std::string &rex)
{
    std::deque<std::string> list;
    boost::split(list, str, boost::is_any_of(rex), boost::token_compress_off);
    return {list.begin(), std::remove_if(C_default_exec_policy, list.begin(), list.end(), [](const std::string &str){ return str.empty(); })};
}

std::string demangle(const char *mangled)
{
    int status;
    std::unique_ptr<char[], void (*)(void *)> result(
            abi::__cxa_demangle(mangled, 0, 0, &status), std::free);
    return result.get() ? std::string(result.get()) : "::UNKNOWN::";
}

std::string sanitize_db_table_name(std::string where, char replace_char)
{
    std::transform(C_default_exec_policy, where.begin(), where.end(), where.begin(), [replace_char](const char &c)
    {
        if (isalnum(c)) {
            return c;
        }
        return replace_char;
    });

    return where;
}

std::string make_md5_hash(const std::string &in)
{
    auto md5_digest_len = EVP_MD_size(EVP_md5());

    // MD5_Init
    auto mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_md5(), NULL);

    // MD5_Update
    EVP_DigestUpdate(mdctx, in.c_str(), in.length());

    // MD5_Final
    auto md5_digest = (unsigned char *) OPENSSL_malloc(md5_digest_len);
    EVP_DigestFinal_ex(mdctx, md5_digest, (unsigned *) &md5_digest_len);
    EVP_MD_CTX_free(mdctx);
    std::stringstream mds;
    for (DTYPE(md5_digest_len) i = 0; i < md5_digest_len; ++i)
        mds << std::hex << std::setfill('0') << std::setw(sizeof(*md5_digest) * 2) << unsigned(md5_digest[i]);
    return mds.str();
}

std::string to_mql_date(const bpt::ptime &time)
{
    static std::locale loc(std::cout.getloc(), new bpt::time_facet(C_mql_date_time_format));
    std::stringstream s;
    s.imbue(loc);
    s << time;
    LOG4_TRACE("Returning " << s.str());
    return s.str();
}

std::map<std::string, std::string> json_to_map(const std::string &json_str)
{
    LOG4_TRACE("Parsing json string: " << json_str);
    std::map<std::string, std::string> result;

    boost::property_tree::ptree pt;
    std::istringstream is(json_str);
    boost::property_tree::json_parser::read_json(is, pt);

    for (const auto &node:pt) result.insert(std::make_pair(node.first, node.second.get<std::string>("")));

    return result;
}

std::string map_to_json(const std::map<std::string, std::string> &value)
{

    std::string retString = "{";
    for (std::pair<std::string, std::string> val : value) {
        retString += " \"" + val.first + "\": \"" + val.second + "\",";
    }
    retString.pop_back();
    retString += " }";
    return retString;
}

std::vector<size_t> parse_string_range(const std::string &parameter_string)
{
    std::vector<size_t> result;
    size_t ss{0};

    if (parameter_string.find(C_dd_separator) != std::string::npos) {
        const int min = std::stoi(parameter_string, &ss);
        const int max = std::stoi(parameter_string.substr(ss + ARRAYLEN(C_dd_separator)));
        result.resize(max - min + 1);
        std::iota(result.begin(), result.end(), min);
        return result;
    }

    if (parameter_string.find(C_cm_separator) != std::string::npos) {
        while (ss < parameter_string.size()) {
            const size_t ss_accumulator{ss};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            result.push_back(std::stoi(parameter_string.substr(ss_accumulator), &ss));
#pragma GCC diagnostic pop
            ss += ARRAYLEN(C_cm_separator) + ss_accumulator;
        }

        return result;
    }

    result.push_back(std::stoi(parameter_string));
    return result;
}

std::vector<std::string>
parse_string_range(const std::string &parameter_string, const std::vector<std::string> &set_parameters)
{
    std::vector<std::string> result;

    size_t it_found = parameter_string.find(C_dd_separator);

    if (it_found != std::string::npos) {
        std::string token_first(parameter_string, 0, it_found);
        std::string token_second(parameter_string, it_found + ARRAYLEN(C_dd_separator));

        // TODO: Check if improvement needed find here as it is linear complexity (find on small vector)
        auto it_wavelet_first = std::find(set_parameters.begin(), set_parameters.end(), token_first);
        auto it_wavelet_second = std::find(set_parameters.begin(), set_parameters.end(), token_second);

        // Validation for begin..end type parameter string.
        if (it_wavelet_first == set_parameters.end() ||
            it_wavelet_second == set_parameters.end() ||
            std::distance(it_wavelet_first, it_wavelet_second) <= 0) {

            LOG4_ERROR("Wrong parameter string: " << parameter_string);
            return result;
        }

        result.assign(it_wavelet_first, it_wavelet_second);
        result.push_back(*it_wavelet_second);
    } else if (parameter_string.find(C_cm_separator) != std::string::npos) {
        // TODO: There is no check if the found parameters exist in the parameters set.
        boost::split(result, parameter_string, boost::is_any_of(C_cm_separator));
    } else {
        result.push_back(parameter_string);
    }

    return result;
}


} // namespace common
} // namespace svr

