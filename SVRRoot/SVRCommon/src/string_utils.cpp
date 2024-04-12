#include <boost/date_time/posix_time/ptime.hpp>
#include <cxxabi.h>
#include <iterator>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <openssl/evp.h>

#include "util/string_utils.hpp"
#include "common/Logging.hpp"
#include <common/constants.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace std {

std::string to_string(const long double v)
{
    std::ostringstream out;
    out.precision(std::numeric_limits<double>::max_digits10);
    return out.str();
}

std::string to_string(const double v)
{
    std::ostringstream out;
    out.precision(std::numeric_limits<double>::max_digits10);
    return out.str();
}

std::string to_string(const float v)
{
    std::ostringstream out;
    out.precision(std::numeric_limits<float>::max_digits10);
    return out.str();
}
}

namespace svr {


namespace common {

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
    std::list<std::string> list;
    boost::split(list, str, boost::is_any_of(rex), boost::token_compress_off);

    std::deque<std::string> res;
    for_each(list.begin(), list.end(), [&res](const std::string &str)
    { if (str.size() != 0) res.push_back(str); });

    return res;
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
    std::transform(std::execution::par_unseq, where.begin(), where.end(), where.begin(), [replace_char](const char &c)
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
    EVP_DigestFinal_ex(mdctx, md5_digest, (unsigned int *) &md5_digest_len);
    EVP_MD_CTX_free(mdctx);
    std::stringstream mds;
    for (dtype(md5_digest_len) i = 0; i < md5_digest_len; ++i)
        mds << std::hex << std::setfill('0') << std::setw(sizeof(*md5_digest) * 2) << unsigned(md5_digest[i]);
    return mds.str();
}

// const std::string C_mt4_date_time_format {"%Y.%m.%d %H:%M:%S"};
std::string to_mt4_date(const bpt::ptime &time)
{
#if 0
    bpt::time_facet *facet = new bpt::time_facet();

    facet->format(C_mt4_date_time_format.c_str());
    stringstream dateStream;
    dateStream.imbue(std::locale(std::locale::classic(), facet));
#endif
    std::stringstream date_stream;
    date_stream << time.date().year() << '.' << time.date().month().as_number() << '.' << time.date().day().as_number() << ' ' <<
        time.time_of_day().hours() << ':' << time.time_of_day().minutes() << ':' << time.time_of_day().seconds();

    return date_stream.str();
}

std::map<std::string, std::string> json_to_map(const std::string &json_str)
{

    LOG4_TRACE("Parsing json string: " << json_str);
    std::map<std::string, std::string> result;

    boost::property_tree::ptree pt;
    std::istringstream is(json_str);
    boost::property_tree::json_parser::read_json(is, pt);

    for (auto &node:pt)
        result.insert(std::make_pair(node.first, node.second.get<std::string>("")));

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
        const int min{std::stoi(parameter_string, &ss)};
        const int max{std::stoi(parameter_string.substr(ss + C_dd_separator.size()))};

        result.reserve(max - min + 1);
        for (int i = min; i <= max; ++i)
            result.push_back(i);

        return result;
    }

    if (parameter_string.find(C_cm_separator) != std::string::npos) {
        while (ss < parameter_string.size()) {
            const size_t ss_accumulator{ss};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            result.push_back(std::stoi(parameter_string.substr(ss_accumulator), &ss));
#pragma GCC diagnostic pop
            ss += C_cm_separator.size() + ss_accumulator;
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
        std::string token_second(parameter_string, it_found + C_dd_separator.size());

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

