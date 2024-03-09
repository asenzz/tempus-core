#include "common/compatibility.hpp"

#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <cxxabi.h>

#include "common/parallelism.hpp"

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

#ifdef __GNUG__

#include <cxxabi.h>

std::string demangle(const std::string &name) {

    int status = -1;
    std::unique_ptr<char, void(*)(void*)> res {abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
    return !status ? res.get() : name;
}

#else

std::string demangle(const std::string &name) {
    return name;
}

#endif

bool operator<(const arma::SizeMat &lhs, const arma::SizeMat &rhs)
{
    return lhs.n_rows * lhs.n_cols < rhs.n_rows * rhs.n_cols;
}


unsigned adj_threads(const ssize_t iterations)
{
    return std::min<size_t>(iterations, std::thread::hardware_concurrency());
}

namespace common {

// TODO Do a GPU handler and ctx from a queue
bool file_exists(const std::string &filename)
{
    return std::ifstream(filename).good();
}


double drander(t_drand48_data_ptr buffer)
{
    double result;
    drand48_r(buffer, &result);
    return result;
}

t_drand48_data seedme(const size_t thread_id)
{
    unsigned short int seed16v[3];
    seed16v[0] = thread_id;
    seed16v[1] = 91789217 * thread_id + 9821279871;
    seed16v[2] = 128719761 * thread_id + 19827619263;
    t_drand48_data buffer;
    seed48_r(seed16v, &buffer);
    return buffer;
}

viennacl::vector<double> tovcl(const arma::colvec &in)
{
    viennacl::vector<double> res(in.size());
    viennacl::fast_copy(in.mem, in.mem + in.n_elem, res.begin());
    return res;
}


ptimes_set_t
to_multistep_times(
        const ptimes_set_t &prediction_times,
        const bpt::time_duration &resolution, const size_t &multistep_len)
{
    bpt::ptime prev;
    size_t multistep_counter = 0;
    ptimes_set_t multistep_subset;
    for (const auto &prediction_time: prediction_times) {

        if ((prediction_time - resolution != prev) || multistep_counter >= multistep_len) {
            multistep_counter = 0;
            multistep_subset.insert(prediction_time);
        }

        prev = prediction_time;
        ++multistep_counter;
    }
    return multistep_subset;
}

ptimes_set_t
to_times(
        const boost::posix_time::time_period &prediction_range,
        const boost::posix_time::time_duration &resolution)
{
    ptimes_set_t result;
    for (auto prediction_time = prediction_range.begin();
         prediction_time < prediction_range.end();
         prediction_time += resolution)
        result.insert(prediction_time);
    return result;
}

ptimes_set_t
to_times(
        const boost::posix_time::time_period &prediction_range,
        const boost::posix_time::time_duration &resolution,
        const size_t comb_train_ct,
        const size_t comb_validate_ct)
{
    size_t ctr = 0;
    ptimes_set_t result;
    for (auto prediction_time = prediction_range.begin();
         prediction_time < prediction_range.end();
         prediction_time += resolution) {
        if (ctr >= comb_train_ct) {
            ++ctr;
            continue;
        }
        result.insert(prediction_time);
        if (ctr >= comb_train_ct + comb_validate_ct) ctr = 0;
        else ++ctr;
    }
    return result;
}


static const unsigned int max_frames = 63;

/** Print a demangled stack backtrace of the caller function to FILE* out. */
void print_stacktrace()
{
    FILE *out = stdout;
    fprintf(out, "stack trace:\n");

    // storage array for stack trace address data
    void *addrlist[max_frames + 1];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void *));

    if (addrlen == 0) {
        fprintf(out, "  <empty, possibly corrupt>\n");
        return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char **symbollist = backtrace_symbols(addrlist, addrlen);

    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char *funcname = (char *) malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < addrlen; i++) {
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = symbollist[i]; *p; ++p) {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset) {
                end_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset && end_offset
            && begin_name < begin_offset) {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():

            int status;
            char *ret = abi::__cxa_demangle(begin_name,
                                            funcname, &funcnamesize, &status);
            if (status == 0) {
                funcname = ret; // use possibly realloc()-ed string
                fprintf(out, "  %s : %s+%s\n",
                        symbollist[i], funcname, begin_offset);
            } else {
                // demangling failed. Output function name as a C function with
                // no arguments.
                fprintf(out, "  %s : %s()+%s\n",
                        symbollist[i], begin_name, begin_offset);
            }
        } else {
            // couldn't parse the line? print the whole line.
            fprintf(out, "  %s\n", symbollist[i]);
        }
    }

    free(funcname);
    free(symbollist);
}

/*bool Equals(const double& lhs, const double& rhs){
	return(std::fabs(lhs - rhs) < (std::numeric_limits<double>::epsilon() * 1000000));
}*/


bool Equals(const double &lhs, const double &rhs)
{
    return Round(lhs) == Round(rhs);
}


double Round(const double &dbl)
{
    return round(dbl * multi_div) / multi_div;
}

size_t hash_param(const double param_val)
{
    return param_val * 1e5;
}

} // common
} // svr