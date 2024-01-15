#pragma once

#include "common/types.hpp"
#include "common/Logging.hpp"
#include <numeric>
#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#include <viennacl/tools/entry_proxy.hpp>
#include <set>
#include <armadillo>
#include <vector>
#include <any>
#include <cstdarg>


namespace svr {

#if 0
template<typename T> constexpr auto pow = [](const T lhs, const long rhs) -> T
{
    const bool negexp = rhs < 0;
    const unsigned long ct = negexp ? -rhs : rhs;
    T res = 1;
    for (unsigned long i = 0; i < ct; ++i) res *= res;
    return negexp ? 1./ res : res;
};
#endif


template<typename T>
std::vector<T> &operator /= (std::vector<T> &lsh, const T rhs)
{
    for (T &v: lsh) v /= rhs;
    return lsh;
}


template<typename T>
std::atomic<T> &operator += (std::atomic<T> &lhs, const T &rhs)
{
    lhs.store(lhs.load() + rhs);
    return lhs;
}


template<typename T>
std::atomic<T> &operator &= (std::atomic<T> &lhs, const T &rhs)
{
    lhs.store(lhs.load() & rhs);
    return lhs;
}


namespace common {

const size_t C_int_nan = 0xDeadBeef;
const size_t C_ip_max = 0x100000000;

/**
 * Extend division reminder to vectors
 *
 * @param   a       Dividend
 * @param   n       Divisor
 */
template<typename T>
T mod(T a, int n)
{
    return a - floor(a / n) * n;
}

void cholesky_check(viennacl::matrix<double> &A);

template<typename T> bool sane(const arma::Mat<T> &m)
{
    if (m.has_nonfinite() or m.has_nan() or m.empty() or m.has_inf()) return false;
    else return true;
}

template<typename T> bool isnormalz(const T v) { return !v || std::isnormal(v); }

std::vector<double> levy(const size_t d); // Return colvec of columns d
double randouble();
arma::cx_mat matlab_fft(const arma::mat &input);
arma::cx_mat matlab_fft(const arma::cx_mat &input);
arma::cx_mat fftshift(const arma::cx_mat &input);
arma::cx_mat matlab_ifft(const arma::cx_mat &input);
arma::cx_mat ifftshift(const arma::cx_mat &input);


template<typename T> bool
equal_to(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon())
{
    T diff = v1 - v2;
    diff = diff > T(0) ? diff : -diff;
    return diff <= eps;
}

template<typename T> bool
less(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon())
{
    return v1 < v2 - eps;
}

template<typename T> bool
greater(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon())
{
    return v1 > v2 + eps;
}

template<typename T> bool
is_contained(const T v, const T from, const T to, const T eps = std::numeric_limits<T>::epsilon())
{
    return from - eps <= v && v <= to + eps;
}

template<typename T> bool
is_zero(const T v, const T eps = std::numeric_limits<T>::epsilon())
{
    return std::abs(v) < eps;
}


template<typename scalar_t> scalar_t
sum(const std::vector<scalar_t> &v)
{
    scalar_t sum = 0;
    for (size_t t = 0; t < v.size(); ++t) sum += v[t];
    return sum;
}

template<typename scalar_t> scalar_t
sumabs(const std::vector<scalar_t> &v)
{
    scalar_t sum = 0;
    for (size_t t = 0; t < v.size(); ++t) sum += std::abs(v[t]);
    return sum;
}


template<typename scalar_t> void
normalize(std::vector<scalar_t> &v, const double min, const double max)
{
    while (std::abs(sum(v) - max) > std::numeric_limits<float>::epsilon()) {
        const scalar_t sum_ma = sum(v);
        const scalar_t adjust = (max - sum_ma) / scalar_t(v.size());
        for (size_t t = 0; t < v.size(); ++t)
            v[t] = std::max(min, v[t] + adjust);
    }
}

template<typename scalar_t> std::vector<scalar_t>
apply_norma(const std::vector<scalar_t> &values, const size_t num_bins)
{
    std::vector<scalar_t> ma_values(values);
    if (num_bins > values.size()) {
        //LOG4_ERROR("Number of bins " << num_bins << " smaller than input size " << values.size());
        return ma_values;
    }
    const auto sumv = sum(values);
    for (size_t t = num_bins; t < values.size(); ++t) {
        scalar_t ma = 0;
        for (size_t t_ma = t - num_bins; t_ma <= t; ++t_ma)
            ma += values[t_ma];
        ma /= scalar_t(num_bins);
        ma_values[t] = ma;
    }
    normalize(ma_values, 0, sumv);
    const auto sum_out = sum(ma_values);
    if (sum_out != sumv) std::cerr << "Sum output " << sum_out << " differs from sum input " << sumv << std::endl;
    return ma_values;
}

template<typename scalar_t> std::vector<scalar_t>
skew_mask(const std::vector<scalar_t> &values, const double exp, const double mult)
{
    std::vector<scalar_t> result(values);
    for (size_t t = 0; t < values.size(); ++t) {
   //     if (t < values.size() / 2)
   //         result[t] *= mult * std::pow(double(t)/double(values.size()), exp);
   //     else
            result[t] *= mult * std::pow(1. - double(t)/double(values.size()), exp);
    }
    normalize(result, 0, 1);
    return result;
}

template<typename scalar_t> std::vector<scalar_t>
stretch_cut_mask(const std::vector<scalar_t> &values, const double factor, const double size_multiplier)
{
    std::vector<scalar_t> result(values * size_multiplier);
    for (size_t t = 0; t < result.size(); ++t) {
        //     if (t < values.size() / 2)
        result[t] = values[size_t(t / factor)];
        //     else
        //         result[t] *= mult * std::pow(1. - double(t)/double(values.size()), exp);
    }
    normalize(result, 0, 1);
    return result;
}

template<typename T> std::vector<T>
operator +(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
    const size_t output_len = std::min<size_t>(lhs.size(), rhs.size());
    std::vector<T> res(output_len);
#pragma omp parallel for simd
    for (size_t i = 0; i < output_len; ++i)
        res[i] = lhs[i] + rhs[i];
    return res;
}

template<typename T> T
snap_copy(const T v, const std::vector<T> &sorted_possible_values)
{
    if (v <= (sorted_possible_values.front() + *std::next(sorted_possible_values.begin()) ) / 2.)  return sorted_possible_values.front();
    if (v > (sorted_possible_values[sorted_possible_values.size() - 2] + sorted_possible_values.back()) / 2.) return sorted_possible_values.back();
    for (size_t i = 1; i < sorted_possible_values.size() - 1; ++i) {
        if ( v > (sorted_possible_values[i - 1] + sorted_possible_values[i]) / 2. && v < (sorted_possible_values[i] + sorted_possible_values[i + 1]) / 2. )
            return sorted_possible_values[i];
    }
    LOG4_THROW("Cannot snap " << v);
    return std::numeric_limits<T>::quiet_NaN();
}

template<typename T, template<typename, typename> typename Container>  T
snap_inplace(T &v, const Container<T, std::allocator<T>> &sorted_possible_values)
{
    if (v < (sorted_possible_values.front() + *std::next(sorted_possible_values.begin()) ) / 2.)  {
        v = sorted_possible_values.front();
        return v;
    }
    if (v > (sorted_possible_values[sorted_possible_values.size() - 2] + sorted_possible_values.back()) / 2.) {
        v = sorted_possible_values.back();
        return v;
    }
    for (size_t i = 1; i < sorted_possible_values.size() - 1; ++i) {
        if ( v > (sorted_possible_values[i - 1] + sorted_possible_values[i]) / 2. && v < (sorted_possible_values[i] + sorted_possible_values[i + 1]) / 2. ) {
            v = sorted_possible_values[i];
            return v;
        }
    }
    return v;
}


template<typename T> T
min(const T &arg1, const T &arg2, const T &arg3)
{
    T m = arg1;
    if (arg2 < m) m = arg2;
    if (arg3 < m) m = arg3;
    return m;
}


void mirror_tail(std::vector<double> &input, const size_t last_data_ct);

arma::mat fmod(const arma::mat &a, const double n);

std::set<size_t> get_adjacent_indexes(const size_t level, const double ratio, const size_t level_count);

bool is_power_of_two(size_t value);

size_t next_power_of_two(size_t value);

template<typename T> size_t
max_size(const std::vector<std::vector<T>> &c)
{
    size_t max = 0;
    for (const auto &cc: c) if (cc.size() > max) max = cc.size();
    //std::cout << "Max size " << max << std::endl;
    return max;
}

template<typename T> T
max(const std::vector<T> &c)
{
    T max = std::numeric_limits<T>::min();
    for (const T &v: c) if (v > max) max = v;
    return max;
}

template<typename T> arma::Mat<T>
join_rows(const size_t arg_ct...)
{
    va_list args;
    va_start(args, arg_ct);
    arma::Mat<T> res;
    for (size_t i = 0; i < arg_ct; ++i) {
        const arma::Mat<T> *p_mat = va_arg(args, const arma::Mat<T> *);
        if (!p_mat) continue;
        if (res.empty()) res.set_size(p_mat->n_rows, 0);
        res = arma::join_rows(res, *p_mat);
    }
    va_end(args);

    return res;
}


template<typename T> std::string
present(const arma::Mat<T> &m)
{
    std::stringstream res;
    const auto vm = arma::vectorise(m);
    res << "elements " << m.n_elem << ", size " << arma::size(m) << ", mean " << arma::mean(vm) << ", max " << arma::max(vm) << ", min " << arma::min(vm) << ", stddev " << arma::stddev(vm) <<
        ", var " << arma::var(vm) << ", median " << arma::median(vm) <<  ", medianabs " << arma::median(arma::abs(vm)) << ", range " << arma::range(vm) << ", meanabs " << arma::mean(arma::abs(vm));
    return res.str();
}

template<typename T> T
meanabs(const arma::Mat<T> &m)
{
    return arma::mean(arma::abs(arma::vectorise(m)));
}


template<typename T> T
medianabs(const arma::Mat<T> &m)
{
    return arma::median(arma::abs(arma::vectorise(m)));
}


template<typename T> T
meanabs(const std::vector<T> &v)
{
    double res = 0;
#pragma omp parallel for reduction(+:res) shared(v) schedule(dynamic, 32) default(none)
    for (const auto _v: v) res += std::abs(_v);
    return res / double(v.size());
}


template<typename scalar_t> scalar_t
meanabs(const typename std::vector<scalar_t>::const_iterator &begin, const typename std::vector<scalar_t>::const_iterator &end)
{
    double res = 0;
    for (auto iter = begin; iter != end; ++iter) res += std::abs(*iter);
    return res / std::abs(double(std::distance(begin, end)));
}

template<typename T> T
sum(const arma::Mat<T> &m)
{
    return arma::sum(arma::vectorise(m));
}

template<typename T> T
sumabs(const arma::Mat<T> &m)
{
    return arma::sum(arma::abs(arma::vectorise(m)));
}

template<typename T> arma::Mat<T>
extrude_rows(const arma::Mat<T> &m, const size_t ct)
{
    arma::Mat<T> r = m;
    while (r.n_cols < ct) r = arma::join_rows(r, m);
    if (r.n_cols > ct) r.shed_cols(ct, r.n_cols - 1);
    return r;
}

#define INF (9.9e99)

std::vector<std::vector<double>>
transpose_matrix(const std::vector<std::vector<double>> &vvmat);

template<typename scalar_t> scalar_t
log_return(
        const scalar_t x,
        const scalar_t x_1)
{
    const auto res = std::log(x) - std::log(x_1);
#ifndef NDEBUG
    if (std::isnan(res)) throw std::invalid_argument(
                "Log return is NaN of " + std::to_string(x) + " and " + std::to_string(x_1));
#endif
    return res;
}

template<typename scalar_t> scalar_t
diff_return(
        const scalar_t x,
        const scalar_t prev_x)
{
    const auto res = x - prev_x;
#ifndef NDEBUG
    if (std::isnan(res)) throw std::invalid_argument(
                "Diff return is NaN of " + std::to_string(x) + " and " + std::to_string(prev_x));
#endif
    return res;
}

template<typename scalar_t> scalar_t
inv_log_return(
        const scalar_t log_ret_x,
        const scalar_t x_1
        )
{
    const auto res = x_1 * std::exp(log_ret_x);
#ifndef NDEBUG
    if (std::isnan(res)) throw std::invalid_argument(
                "Inverse log return is NaN of " + std::to_string(x_1) + " and " + std::to_string(log_ret_x));
#endif
    return res;
}

template<typename scalar_t> scalar_t
inv_diff_return(
        const scalar_t diff_ret_x,
        const scalar_t prev_actual_x)
{
    const auto res = prev_actual_x + diff_ret_x;
#ifndef NDEBUG
    if (std::isnan(res)) throw std::invalid_argument(
                "Inverse diff return is NaN of " + std::to_string(prev_actual_x) + " and " + std::to_string(diff_ret_x));
#endif
    return res;
}


template<class T>
T ABS(T X) {
    if (X >= 0)
        return X;
    else
        return -X;
}

template<class T>
T SIGN(T X) {
    if (X >= 0)
        return (T) 1;
    else
        return (T) -1;
}

double get_uniform_random_value();

std::vector<double> get_uniform_random_vector(const size_t size);

// TODO: rewrite with iterators
std::vector<double> get_uniform_random_vector(const std::pair<std::vector<double>, std::vector<double>>& boundaries);


// TODO: rewrite with template
template<typename T>
std::vector<size_t> argsort(const std::vector<T>& v)
{
    std::vector<size_t> result (v.size());
    std::iota(result.begin(), result.end(), 0);

    std::sort(result.begin(), result.end(), [&](size_t i, size_t j) {return v[i] < v[j];});

    return result;
}

std::vector<double> operator* (const std::vector<double>& v1, const double& m);

std::vector<double> operator* (const double& m, const std::vector<double>& v1);

std::vector<double> operator* (const std::vector<double>& v1, const std::vector<double>& v2);

std::vector<double> operator+ (const std::vector<double>& v1, const std::vector<double>& v2);

std::vector<double> operator- (const std::vector<double>& v1, const std::vector<double>& v2);

std::vector<double> operator-(const std::vector<double>& v);

std::vector<double> operator^(const std::vector<double>& v, const double a);

std::vector<double> operator^(const double a, const std::vector<double>& v);

double dot_product(double const * a, double const * b, const size_t size);

#define DEFAULT_T_EPSILON (std::numeric_limits<float>::epsilon())

inline bool is_equal(double const &t1, double const &t2)
{
    static double const epsilon = std::numeric_limits<float>::epsilon();
    if(t1 == t2)
        return true;

    if (fabs(t1) < epsilon && fabs(t2) < epsilon)
        return true;

    if(t2 == 0)
        return false;

    return fabs(1. - t1/t2) < epsilon;
}

inline bool equals(const double t1, const double t2, const double epsilon)
{
    return fabs(t1 - t2) < epsilon;
}

// Add to vector
#define AVEC_PUSH(v, val) {\
const auto _n_elem  = v.size(); \
v.resize(_n_elem + 1); \
v(_n_elem) = (val); }

// Add to vector thread safe
#define AVEC_PUSH_TS(v, val, m) { \
    std::scoped_lock l(m); \
    const auto _n_elem  = v.size(); \
    v.resize(_n_elem + 1); \
    v(_n_elem) = (val); }

namespace armd {
    void check_mat(const arma::mat &input);

    void print_mat(const arma::mat & input, const std::string mat_name = "");

    void serialize_mat(const std::string &filename, const arma::mat &input, const std::string mat_name = "");

    void serialize_vec(const std::string &filename, const arma::uvec &input, const std::string vec_name = "");

    arma::uvec set_to_arma_uvec(const std::set<size_t> input);

    arma::uvec complement_vectors(std::set<size_t> svi, arma::uvec new_ixs);

    void shuffle_matrix(const arma::mat & x, const arma::mat & y, arma::uvec & shuffled_rows, arma::uvec & shuffled_cols);

    arma::mat pdist(const arma::mat & input);

    double mean_all(const arma::mat & input);

    arma::mat rows(arma::mat & input, arma::uvec & ixs);

    arma::mat rows(arma::mat & input, std::set<size_t> & ixs);

    void print_arma_sizes(const arma::mat &input, const std::string input_name);

    arma::uvec subview_indexes(arma::uvec batch_ixs, arma::uvec ix_tracker);

    arma::mat shuffle_admat(const arma::mat & to_shuffle, const size_t level);

    arma::mat fixed_shuffle(const arma::mat & to_shuffle);
}

double calc_quant_offset_mul(const double main_to_aux_period_ratio, const double level, const double levels_count);


}
}
