#pragma once
#ifndef ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_GCC
#endif

#include <set>
#include <armadillo>
#include <vector>
#include <any>
#include <cstdarg>
#include <numeric>
#ifdef ENABLE_OPENCL
#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#endif
#include <boost/math/ccmath/ccmath.hpp>
#include <mkl_cblas.h>
#include "common.hpp"
#include "common/compatibility.hpp"
#include "util/math_utils.hpp"
#include "common/gpu_handler.hpp"
#include "common/parallelism.hpp"

namespace svr {

template <typename A, typename B> using common_signed_t = std::conditional_t<std::is_unsigned_v<A> && std::is_unsigned_v<B>,
        std::common_type_t<A, B>,
        std::common_type_t<std::make_signed_t<A>, std::make_signed_t<B>>>;

#define _ABS(X) ((X) >= 0 ? (X) : (-X))
#define _ABSDIF(T1, T2) (T1 > T2 ? T1 - T2 : T2 - T1)
#define _SIGN(X) ((X) > 0. ? 1.: (X) < 0 ? -1. : 0.)
#define _MIN(X, Y) (((X) > (Y)) ? (Y) : (X)) // X returned if smaller or equal
#define _MAX(X, Y) (((X) < (Y)) ? (Y) : (X)) // X returned if larger or equal
#define MAXAS(X, Y) if ((X) < (Y)) (X) = (Y) // (((X) < (Y)) && ((X) = (Y))) // Y assigned to X only if Y larger than X
#define MINAS(X, Y) if ((X) > (Y)) (X) = (Y) // (((X) > (Y)) && ((X) = (Y))) // Y assigned to X only if Y smaller than X
#define OMPMINAS(X, Y) X = _MIN(X, Y)
#define OMPMAXAS(X, Y) X = _MAX(X, Y)

#define CDIV(X, Y) boost::math::ccmath::ceil(double(X) / double(Y))
#define CDIVI(X, Y) ((X) / (Y) + ((X) % (Y) != 0))
#define LDi(i, m, ld) (((i) % (m)) + ((i) / (m)) * (ld))

constexpr double C_pi_2 = 2 * M_PI;

template <typename Dividend, typename Divisor> inline constexpr common_signed_t<Dividend, Divisor> cdivi(Dividend x, Divisor y)
{
    if constexpr (std::is_unsigned_v<Dividend> && std::is_unsigned_v<Divisor>) {
        // quotient is always positive
        return x / y + (x % y != 0);  // uint / uint
    }
    else if constexpr (std::is_signed_v<Dividend> && std::is_unsigned_v<Divisor>) {
        auto sy = static_cast<std::make_signed_t<Divisor>>(y);
        bool quotientPositive = x >= 0;
        return x / sy + (x % sy != 0 && quotientPositive);  // int / uint
    }
    else if constexpr (std::is_unsigned_v<Dividend> && std::is_signed_v<Divisor>) {
        auto sx = static_cast<std::make_signed_t<Dividend>>(x);
        bool quotientPositive = y >= 0;
        return sx / y + (sx % y != 0 && quotientPositive);  // uint / int
    }
    else {
        bool quotientPositive = (y >= 0) == (x >= 0);
        return x / y + (x % y != 0 && quotientPositive);  // int / int
    }
}

template<typename T1, typename T2> inline constexpr T1 cdiv(const T1 l, const T2 r) { return CDIV((l), (r)); }

std::vector<double> operator*(const std::vector<double> &v1, const double &m);

std::vector<double> operator*(const double &m, const std::vector<double> &v1);

std::vector<double> operator*(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator+(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator-(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator-(const std::vector<double> &v);

std::vector<double> operator^(const std::vector<double> &v, const double a);

std::vector<double> operator^(const double a, const std::vector<double> &v);

template<typename T>
std::vector<T> &operator/=(std::vector<T> &lsh, const T rhs)
{
    for (T &v: lsh) v /= rhs;
    return lsh;
}


template<typename T>
std::atomic<T> &operator+=(std::atomic<T> &lhs, const T &rhs)
{
    lhs.store(lhs.load() + rhs);
    return lhs;
}


template<typename T>
std::atomic<T> &operator&=(std::atomic<T> &lhs, const T &rhs)
{
    lhs.store(lhs.load() & rhs);
    return lhs;
}


namespace common {


template<typename T> T absdiffratio(const T a, const T b)
{
    if (a == 0 && b == 0) return 0;
    return std::abs(a - b) / (std::abs(a) + std::abs(b));
}

template<typename T> T absdiffinvratio(const T a, const T b)
{
    if (a == 0 && b == 0) return 0;
    return T(1) - std::abs(a - b) / (std::abs(a) + std::abs(b));
}

struct pseudo_random_dev {
    static std::atomic<double> state;

    double operator()()
    {
        state = std::fmod(state + .123456789, 1.);
        return state;
    }

    static double max(const double max)
    {
        state = std::fmod(state + .123456789, 1.);
        return max * state;
    }
};

/// @brief Returns a reproducibly seeded 64-bit RNG
template<typename RNG> RNG reproducibly_seeded_64()
{
    constexpr auto n_words = RNG::word_count();

    // A seed array we will fill for this RNG.
    std::array<typename RNG::word_type, n_words> seed_array;
    pseudo_random_dev dev;
    std::generate(seed_array.begin(), seed_array.end(), [&dev] { return dev(); });

    // Construct the RNG
    RNG retval(seed_array.cbegin(), seed_array.cend());
    return retval;
}


size_t fft_len(const size_t input_len);

arma::vec add_to_arma(const std::vector<double> &v1, const std::vector<double> &v2);

template<typename T> std::enable_if_t<std::is_integral_v<T>, T> bounce(const T v, const T lim)
{
    return v > lim ? lim - v % lim : v;
}

template<typename T> std::enable_if_t<not std::is_integral_v<T>, T> bounce(const T v, const T lim)
{
    return v > lim ? lim - std::fmod(v, lim) : v;
}

const size_t C_int_nan = 0xDeadBeef;
const size_t C_ip_max = 0x100000000;

/**
 * Extend division reminder to vectors
 *
 * @param   a       Dividend
 * @param   n       Divisor
 */

template<typename T> arma::Mat<T>
mod(const arma::Mat<T> &a, const T n)
{
    return a - arma::floor(a / n) * n;
}

#ifdef ENABLE_OPENCL
void cholesky_check(viennacl::matrix<double> &A);
#endif

// Sign-safe natural logarithm of scalar
template<typename T> inline std::enable_if_t<std::is_scalar_v<T>, T> sln(const T v)
{
    return v < 0 ? -1. * std::log(std::abs(v)) : std::log(v);
}

// Sign-safe natural logarithm of arma matrix
template<typename T> inline arma::Mat<T> slog(const arma::Mat<T> &m)
{
    arma::Mat<T> r(arma::size(m));
    const arma::uvec nsd_ixs = arma::find(r < 0);
    r.elem(nsd_ixs) = -1. * arma::log(arma::abs(m.elem(nsd_ixs)));
    const arma::uvec pd_ixs = arma::find(r >= 0);
    r.elem(pd_ixs) = arma::log(m.elem(pd_ixs));
    return r;
}

// Sign-safe natural logarithm of arma matrix in-place
template<typename T> inline T slog_I(T &&m)
{
    const arma::uvec nsd_ixs = arma::find(m < 0);
    m.elem(nsd_ixs) = -1. * arma::log(arma::abs(m.elem(nsd_ixs)));
    const arma::uvec pd_ixs = arma::find(m >= 0);
    m.elem(pd_ixs) = arma::log(m.elem(pd_ixs));
    return m;
}

template<typename T> inline std::enable_if_t<std::is_scalar_v<T>, T> sexp(const T v)
{
    return v < 0 ? -1. * std::exp(std::abs(v)) : std::exp(v);
}

// Sign-safe natural exponent of arma matrix
template<typename T> inline arma::Mat<T> sexp(const arma::Mat<T> &m)
{
    arma::Mat<T> r(arma::size(m));
    const arma::uvec nsd_ixs = arma::find(r < 0);
    r.elem(nsd_ixs) = -1. * arma::exp(arma::abs(m.elem(nsd_ixs)));
    const arma::uvec pd_ixs = arma::find(r >= 0);
    r.elem(pd_ixs) = arma::exp(m.elem(pd_ixs));
    return r;
}

// Sign-safe natural exponent of arma matrix in-place
template<typename T> inline T &sexp_I(T &&m)
{
    const arma::uvec nsd_ixs = arma::find(m < 0);
    m.elem(nsd_ixs) = -1. * arma::exp(arma::abs(m.elem(nsd_ixs)));
    const arma::uvec pd_ixs = arma::find(m >= 0);
    m.elem(pd_ixs) = arma::exp(m.elem(pd_ixs));
    return m;
}

template<typename T> bool isnormalz(const T v)
{ return v == 0 || std::isnormal(v); }

arma::vec levy(const size_t d);

double randouble();

arma::cx_mat matlab_fft(const arma::mat &input);

arma::cx_mat matlab_fft(const arma::cx_mat &input);

arma::cx_mat fftshift(const arma::cx_mat &input);

arma::cx_mat matlab_ifft(const arma::cx_mat &input);

arma::cx_mat ifftshift(const arma::cx_mat &input);

void negate(double *const v, const size_t len);

void equispaced(arma::mat &x0, const arma::mat &bounds, const arma::vec &pows, uint64_t sobol_ctr = 0);

double alpha(const double reference_error, const double prediction_error);

double mape(const double absolute_error, const double absolute_label);

template<typename T, typename U> double imprv(const T newv, const U oldv)
{
    return 100. * (1. - double(newv) / double(oldv));
}

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

double fdsum(CPTR(float) v, const unsigned len);

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

template<typename T> T
sumabs(const arma::Mat<T> &m)
{
    return arma::accu(arma::abs(m));
}

template<> double sumabs(const std::vector<double> &v);

template<> float sumabs(const std::vector<float> &v);

template<> double sumabs(const arma::Mat<double> &m);

template<> float sumabs(const arma::Mat<float> &m);

template<typename T> arma::uvec find(const arma::Mat <T> &m1, const arma::Mat <T> &m2)
{
    arma::uvec r;
OMP_FOR_(m2.n_elem, ordered)
    for (const auto e2: m2) {
        const arma::uvec r_e = arma::find(m1 == e2);
        if (r_e.empty()) continue;
#pragma omp ordered
        r.insert_rows(r.n_rows, r_e);
    }
    return r;
}

template<typename T> arma::uvec find_ge(const arma::Mat <T> &m1, const arma::Mat <T> &m2)
{
    arma::uvec r;
OMP_FOR_(m2.n_elem, ordered)
    for (const auto e2: m2) {
        const arma::uvec r_e = arma::find(m1 >= e2);
        if (r_e.empty()) continue;
#pragma omp ordered
        r.insert_rows(r.n_rows, r_e);
    }
    return r;
}

template<typename T> arma::Mat<T> &normalize_cols_I(arma::Mat<T> &m)
{
    const arma::Row<T> means = arma::mean(m);
    OMP_FOR(m.n_cols)
    for (unsigned c = 0; c < m.n_cols; ++c) {
        m.col(c) -= means[c];
        m.col(c) /= arma::median(m.col(c));
    }
    return m;
}

template<typename T> arma::Mat<T> normalize_cols(const arma::Mat<T> &m)
{
    const arma::Row<T> means = arma::mean(m);
    arma::Mat<T> r(arma::size(m), ARMA_DEFAULT_FILL);
    OMP_FOR(m.n_cols)
    for (size_t c = 0; c < m.n_cols; ++c) {
        r.col(c) = m.col(c) - means[c];
        r.col(c) /= m.n_rows > 10 ? arma::median(r.col(c)) : arma::mean(r.col(c));
    }
    return r;
}

template<typename T> arma::Col<T> &normalize_I(arma::Col<T> &v)
{
    const auto mean = arma::mean(v);
    v -= mean;
    v /= arma::median(v);
    return v;
}

template<typename T> arma::Mat<T> &normalize_I(arma::Mat<T> &m)
{
    const auto mean = arma::accu(m) / double(m.n_elem);
    m -= mean;
    m /= arma::median(arma::vectorise(m));
    return m;
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
        result[t] *= mult * std::pow(1. - double(t) / double(values.size()), exp);
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

template<typename T> inline auto &shift_I(arma::Mat<T> &m, const arma::Col<unsigned> &shiftings)
{
    if (shiftings.n_elem != m.n_cols) LOG4_THROW("Shiftings " << arma::size(shiftings) << " not compatible with matrix " << arma::size(m));
    OMP_FOR(m.n_cols)
    for (size_t i = 0; i < m.n_cols; ++i)
        if (shiftings[i])
            m.col(i) = arma::shift(m.col(i), shiftings[i]);
    return m;
}

template<typename T> inline arma::Mat<T> shift(const arma::Mat<T> &m, const arma::Col<unsigned> &shiftings)
{
    arma::Mat<T> v(arma::size(m), ARMA_DEFAULT_FILL);
    if (shiftings.n_elem != m.n_cols) LOG4_THROW("Shiftings " << arma::size(shiftings) << " not compatible with matrix " << arma::size(m));
    OMP_FOR(m.n_cols)
    for (size_t i = 0; i < m.n_cols; ++i) v.col(i) = shiftings[i] ? arma::shift(m.col(i), shiftings[i]).eval() : m.col(i);
    return v;
}

template<typename T> arma::Col<T> inline stretch_crop(const arma::subview<T> &v, const double f, size_t n = 0)
{
    if (f == 1) return v;
    if (!n) n = v.n_elem;
    arma::Col<T> r(n);
    arma::uvec div(n);
UNROLL()
    for (size_t out_i = 0; out_i < r.n_elem; ++out_i) {
        r[out_i] += v[size_t(out_i / f) % v.n_elem];
        ++div[out_i];
    }
    return r / div;
}

template<typename T> arma::Col<T> stretch(const arma::Col<T> &v, const double f)
{
    if (f == 1) return v;

    arma::Col<T> r(v.n_elem * f);
    arma::uvec div(r.n_elem);
UNROLL()
    for (unsigned out_i = 0; out_i < r.n_elem; ++out_i) {
        r[out_i] += v[double(out_i) / f];
        ++div[out_i];
    }
    return r / div;
}

template<typename T> arma::Col<T> stretch(const arma::Col<T> &v, const size_t n)
{
    return stretch(v, double(n) / double(v.n_elem));
}

template<typename T> std::vector<T>
operator+(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
    const size_t output_len = std::min<size_t>(lhs.size(), rhs.size());
    std::vector<T> res(output_len);
#pragma omp parallel for simd num_threads(adj_threads(output_len))
    for (size_t i = 0; i < output_len; ++i)
        res[i] = lhs[i] + rhs[i];
    return res;
}

template<typename T> T
snap_copy(const T v, const std::vector<T> &sorted_possible_values)
{
    if (v <= (sorted_possible_values.front() + *std::next(sorted_possible_values.begin())) / 2.) return sorted_possible_values.front();
    if (v > (sorted_possible_values[sorted_possible_values.size() - 2] + sorted_possible_values.back()) / 2.) return sorted_possible_values.back();
    for (size_t i = 1; i < sorted_possible_values.size() - 1; ++i) {
        if (v > (sorted_possible_values[i - 1] + sorted_possible_values[i]) / 2. && v < (sorted_possible_values[i] + sorted_possible_values[i + 1]) / 2.)
            return sorted_possible_values[i];
    }
    LOG4_THROW("Cannot snap " << v);
    return std::numeric_limits<T>::quiet_NaN();
}

template<typename T, template<typename, typename> typename Container> T
snap_inplace(T &v, const Container<T, std::allocator<T>> &sorted_possible_values)
{
    if (v < (sorted_possible_values.front() + *std::next(sorted_possible_values.begin())) / 2.) {
        v = sorted_possible_values.front();
        return v;
    }
    if (v > (sorted_possible_values[sorted_possible_values.size() - 2] + sorted_possible_values.back()) / 2.) {
        v = sorted_possible_values.back();
        return v;
    }
    for (size_t i = 1; i < sorted_possible_values.size() - 1; ++i) {
        if (v > (sorted_possible_values[i - 1] + sorted_possible_values[i]) / 2. && v < (sorted_possible_values[i] + sorted_possible_values[i + 1]) / 2.) {
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

bool is_power_of_two(size_t value);

size_t next_power_of_two(size_t value);

template<typename C, typename T = typename C::value_type> T
max(const C &container)
{
    T max = 0;
    for (const auto &elem: container)
        if (elem > max)
            max = elem;
    return max;
}

double stdscore(const double *const v, const size_t len);

double meanabs_hiquant(const double *const v, const size_t len, const double q);

double meanabs_loquant(const double *const v, const size_t len, const double q);

double meanabs_quant(const double *const v, const size_t len, const double q);

double mean_hiquant(const double *const v, const size_t len, const double q);

double mean_loquant(const double *const v, const size_t len, const double q);

double medianabs(double *const v, const size_t len);

// Use for deque and vector containers
template<typename T, typename U, typename E = typename T::value_type, typename F = typename U::value_type>
void keep_indices(T &d, const U &indices)
{
    if (d.size() <= indices.size() || d.size() <= common::max(indices) || d.empty()) return;
    using size_T = typename T::size_type;
    size_T last = 0;
UNROLL()
    for (size_T i = 0; i < d.size(); ++i, ++last) {
        while (std::find(indices.cbegin(), indices.cend(), i) == indices.cend() && i < d.size()) ++i;
        if (i >= d.size()) break;
        d[last] = d[i];
    }
    d.resize(last);
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

__device__ __host__ __forceinline__ double scale(const double v, const double sf, const double dc)
{
    return (v - dc) / sf;
}

template<typename T> inline T scale(const T &v, const double sf, const double dc)
{
    return (v - dc) / sf;
}

template<> arma::mat scale(const arma::mat &m, const double sf, const double dc);


template<typename T> __host__ __device__ inline T &scale_I(T &v, const double sf, const double dc)
{
    return v = (v - dc) / sf;
}

template<> arma::mat &scale_I(arma::mat &m, const double sf, const double dc);


template<typename T> inline T unscale(const T &v, const double sf, const double dc)
{
    return v * sf + dc;
}

template<typename T> inline T &unscale_I(T &v, const double sf, const double dc)
{
    return v = v * sf + dc;
}

template<> arma::mat unscale(const arma::mat &m, const double sf, const double dc);
template<> arma::mat &unscale_I(arma::mat &m, const double sf, const double dc);

template<typename T> inline T constrain(T v, const T min, const T max)
{
    if (v < min) v = min;
    else if (v > max) v = max;
    return v;
}

template<typename T> arma::Mat<T>
norm(const arma::subview<std::complex<T>> &cxm)
{
    arma::Mat<T> norm_cxm(arma::size(cxm));
#pragma omp parallel for num_threads(adj_threads(cxm.n_elem)) schedule(static, 1 + cxm.n_elem / C_n_cpu)
    for (size_t i = 0; i < cxm.n_elem; ++i) norm_cxm[i] = std::norm(cxm[i]);
    return norm_cxm;
}


template<typename T> arma::Mat<T>
norm(const arma::Mat<std::complex<T>> &cxm)
{
    arma::Mat<T> norm_cxm(arma::size(cxm));
#pragma omp parallel for num_threads(adj_threads(cxm.n_elem)) schedule(static, 1 + cxm.n_elem / C_n_cpu)
    for (size_t i = 0; i < cxm.n_elem; ++i) norm_cxm[i] = std::norm(cxm[i]);
    return norm_cxm;
}

template<typename T> inline T
mean_asymm(const arma::Mat<T> &m, const size_t last)
{
    return .5 * (arma::mean(arma::vectorise(m.submat(m.n_rows - last, m.n_cols - 1, m.n_rows - 2, m.n_cols - 1))) +
                 arma::mean(arma::vectorise(m.submat(m.n_rows - 1, m.n_cols - last, m.n_rows - 1, m.n_cols - 2))));
}

template<typename T> T
meanabs(const arma::Mat<T> &m)
{
    return arma::mean(arma::abs(arma::vectorise(m)));
}

template<> double meanabs<double>(const arma::Mat<double> &m);

template<typename T> inline T medianabs(const arma::Mat<T> &m)
{
    return arma::median(arma::abs(arma::vectorise(m)));
}

template<typename T> inline T
meanabs(const std::span<T> &v)
{
    return std::reduce(C_default_exec_policy, v.begin(), v.end(), 0., [](const double acc, const double val) { return acc + std::abs(val); }) / double(v.size());
}

template<> double meanabs(const std::span<double> &v);

template<> float meanabs(const std::span<float> &v);

template<typename T> inline T
meanabs(const std::vector<T> &v)
{
    return meanabs(std::span(v));
}

template<typename T> T
meanabs(const std::deque<T> &v)
{
    return std::reduce(C_default_exec_policy, v.cbegin(), v.cend(), 0., [](const double acc, const double val) { return acc + std::abs(val); }) / double(v.size());
}


template<typename scalar_t> scalar_t
meanabs(const typename std::vector<scalar_t>::const_iterator &begin, const typename std::vector<scalar_t>::const_iterator &end)
{
    return std::reduce(C_default_exec_policy, begin, end, 0., [](const double acc, const double v) { return acc + std::abs(v); }) / double(end - begin);
}

template<> double
meanabs(const typename std::vector<double>::const_iterator &begin, const typename std::vector<double>::const_iterator &end);

template<typename T> inline arma::Mat<T> extrude_rows(const arma::Mat<T> &m, const uint32_t ct)
{
    arma::Mat<T> r = m;
    while (r.n_cols < ct) r = arma::join_horiz(r, m);
    if (r.n_cols > ct) r.shed_cols(ct, r.n_cols - 1);
    return r;
}

template<typename T> inline arma::Mat<T> extrude_cols(const arma::Mat<T> &m, const uint32_t ct)
{
    assert(m.n_rows > 0);
    arma::Mat<T> r(ct, m.n_cols);
    OMP_FOR(ct / m.n_rows)
    for (uint32_t start_i = 0; start_i < ct; start_i += m.n_rows) {
        const auto end_i = std::min<uint32_t>(ct, start_i + m.n_rows) - 1;
        r.rows(start_i, end_i) = m.rows(0, m.n_rows - 1);
    }
    return r;
}

template<typename T> std::stringstream present_s(const arma::Mat<T> &m)
{
    std::stringstream res;
    res << std::setprecision(std::numeric_limits<double>::max_digits10) << "elements " << m.n_elem << ", size " << arma::size(m);
    if (m.empty()) {
        res << ", empty";
        return res;
    }
    arma::Col<T> vm((T *)m.memptr(), m.n_elem, true, true);
    bool m_has_nan = false, m_has_inf = false;
    vm.for_each([&m_has_nan, &m_has_inf](T &val) {
        if (std::isnan(val)) {
            val = 0;
            m_has_nan = true;
        } else if (std::isinf(val)) {
            val = 0;
            m_has_inf = true;
        }
    });
    if (m_has_inf) res << ", has infinite";
    if (m_has_nan) res << ", has nan";
    res << ", mean " << arma::mean(vm) << ", max " << arma::max(vm) << ", index max " << vm.index_max() << ", min " << arma::min(vm) << ", index min " << vm.index_min() <<
            ", stddev " << arma::stddev(vm) << ", var " << arma::var(vm) << ", median " << arma::median(vm) << ", medianabs " << arma::median(arma::abs(vm)) << ", range " <<
            arma::range(vm) << ", meanabs " << meanabs(vm) << ", values " << common::to_string(m.mem,  std::min<size_t>(m.n_elem, 5));
    return res;
}

template<typename T> std::string present(const arma::Mat<T> &m)
{
    if constexpr (std::is_integral_v<T>) return present_s(arma::conv_to<arma::mat>::from(m)).str();
    else return present_s(m).str();
}

template<typename T> std::string present(const std::span<T> &v)
{
    return present(arma::vec(v.data(), v.size(), false, true));
}

template<typename T> std::string present(const std::vector<T> &v)
{
    return present(arma::vec(v.data(), v.size(), false, true));
}

template<typename T> std::string
present(const arma::subview<T> &m)
{
    std::stringstream res;
    const arma::vec vm = arma::conv_to<arma::vec>::from(m);
    res << std::setprecision(std::numeric_limits<double>::max_digits10) <<
        "elements " << m.n_elem << ", size " << arma::size(m) << ", mean " << arma::mean(vm) << ", max " << arma::max(vm) << ", min " << arma::min(vm) << ", stddev "
        << arma::stddev(vm) <<
        ", var " << arma::var(vm) << ", median " << arma::median(vm) << ", medianabs " << arma::median(arma::abs(vm)) << ", range " << arma::range(vm) << ", meanabs "
        << arma::mean(arma::abs(vm)) << ", values ";
    if (m.n_elem) {
        for (uint16_t i = 0; i < std::min<size_t>(m.n_elem, 5) - 1; ++i) res << m[i] << ", ";
        res << m.back();
    }
    return res.str();
}

std::string present_chunk(const arma::uvec &u, const double head_factor);

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

double get_uniform_random_value();

std::vector<double> get_uniform_random_vector(const size_t size);

// TODO: rewrite with iterators
std::vector<double> get_uniform_random_vector(const std::pair<std::vector<double>, std::vector<double>> &boundaries);


// TODO: rewrite with template
template<typename T>
std::vector<size_t> argsort(const std::vector<T> &v)
{
    std::vector<size_t> result(v.size());
    std::iota(result.begin(), result.end(), 0);

    std::sort(result.begin(), result.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });

    return result;
}


double dot_product(double const *a, double const *b, const size_t size);

template<typename T> inline bool equals(const T t1, const T t2, const T epsilon = std::numeric_limits<T>::epsilon())
{
    return fabs(t1 - t2) < epsilon;
}

void shuffle_matrix(const arma::mat &x, const arma::mat &y, arma::uvec &shuffled_rows, arma::uvec &shuffled_cols);

arma::mat pdist(const arma::mat &input);

double mean(const arma::mat &input);

double mean(const double *const input, const size_t len);

double max(const arma::mat &input);

double min(const arma::mat &input);

arma::mat shuffle_admat(const arma::mat &to_shuffle, const size_t level);

arma::uvec fixed_shuffle(const arma::uvec &to_shuffle);

double calc_quant_offset_mul(const double main_to_aux_period_ratio, const double level, const double levels_count);

double get_quantization_max(const double main_to_aux_period_ratio);

arma::mat shuffle_admat(const arma::mat &to_shuffle, const size_t level);

arma::uvec fixed_shuffle(const arma::uvec &to_shuffle);

arma::uvec complement_vectors(std::set<size_t> svi, arma::uvec new_ixs);

arma::uvec subview_indexes(arma::uvec batch_ixs, arma::uvec ix_tracker);

struct safe_double_less {
    bool operator()(const double left, const double right) const;
};

}
}
