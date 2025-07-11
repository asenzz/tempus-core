//
// Created by zarko on 10/07/2025.
//

#ifndef MATH_UTILS_TPP
#define MATH_UTILS_TPP

#include "math_utils.hpp"

namespace svr {

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

template<typename T> std::vector<T> &operator/=(std::vector<T> &lsh, const T rhs)
{
    for (T &v: lsh) v /= rhs;
    return lsh;
}

template<typename T> std::atomic<T> &operator+=(std::atomic<T> &lhs, const T &rhs)
{
    lhs.store(lhs.load() + rhs);
    return lhs;
}

template<typename T> std::atomic<T> &operator&=(std::atomic<T> &lhs, const T &rhs)
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

template<typename T> T mean(const arma::Mat<T> &input)
{
    return arma::mean(arma::vectorise(input));
}

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

template<typename T> std::enable_if_t<std::is_integral_v<T>, T> bounce(const T v, const T lim)
{
    return v > lim ? lim - v % lim : v;
}

template<typename T> std::enable_if_t<not std::is_integral_v<T>, T> bounce(const T v, const T lim)
{
    return v > lim ? lim - std::fmod(v, lim) : v;
}

template<typename T> arma::Mat<T> mean_mask(const arma::Mat<T> &x, const int32_t radius)
{
    arma::Mat<T> r(arma::size(x), ARMA_DEFAULT_FILL);
    const auto n_sum = 2 * radius + 1;
    OMP_FOR_(x.n_elem, collapse(2))
    for (uint32_t k = 0; k < x.n_cols; ++k) {
        for (int32_t i = 0; i < int32_t(x.n_rows); ++i) {
            double sum = 0;
            for (auto j = -int32_t(radius); j <= radius; ++j) {
                auto ij = i + j;
                while (ij < 0) ij += x.n_rows;
                while (ij >= int32_t(x.n_rows)) ij -= x.n_rows;
                sum += x(ij, k);
            }
            r(i, k) = sum / n_sum;
        }
    }
    return r;
}

template<typename T> arma::Mat<T> mod(const arma::Mat<T> &a, const T n)
{
    return a - arma::floor(a / n) * n;
}

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

template<typename T, typename U> double imprv(const T newv, const U oldv)
{
    return 100. * (1. - double(newv) / double(oldv));
}

template<typename T> bool equal_to(const T v1, const T v2, const T eps)
{
    T diff = v1 - v2;
    diff = diff > T(0) ? diff : -diff;
    return diff <= eps;
}

template<typename T> bool less(const T v1, const T v2, const T eps)
{
    return v1 < v2 - eps;
}

template<typename T> bool greater(const T v1, const T v2, const T eps)
{
    return v1 > v2 + eps;
}

template<typename T> bool is_contained(const T v, const T from, const T to, const T eps)
{
    return from - eps <= v && v <= to + eps;
}

template<typename T> bool is_zero(const T v, const T eps)
{
    return std::abs(v) < eps;
}

template<typename scalar_t> scalar_t sum(const std::vector<scalar_t> &v)
{
    scalar_t sum = 0;
    for (size_t t = 0; t < v.size(); ++t) sum += v[t];
    return sum;
}

template<typename scalar_t> scalar_t sumabs(const std::vector<scalar_t> &v)
{
    scalar_t sum = 0;
    for (size_t t = 0; t < v.size(); ++t) sum += std::abs(v[t]);
    return sum;
}

template<typename T> T sumabs(const arma::Mat<T> &m)
{
    return arma::accu(arma::abs(m));
}

template<typename T> arma::uvec find(const arma::Mat<T> &m1, const arma::Mat<T> &m2)
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

template<typename T> arma::uvec find_ge(const arma::Mat<T> &m1, const arma::Mat<T> &m2)
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
apply_norma(const std::vector<scalar_t> &values, size_t num_bins)
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
    if (shiftings.n_elem != m.n_cols)
        LOG4_THROW("Shiftings " << arma::size(shiftings) << " not compatible with matrix " << arma::size(m));
    OMP_FOR(m.n_cols)
    for (size_t i = 0; i < m.n_cols; ++i)
        if (shiftings[i])
            m.col(i) = arma::shift(m.col(i), shiftings[i]);
    return m;
}

template<typename T> inline arma::Mat<T> shift(const arma::Mat<T> &m, const arma::Col<unsigned> &shiftings)
{
    arma::Mat<T> v(arma::size(m), ARMA_DEFAULT_FILL);
    if (shiftings.n_elem != m.n_cols)
        LOG4_THROW("Shiftings " << arma::size(shiftings) << " not compatible with matrix " << arma::size(m));
    OMP_FOR(m.n_cols)
    for (size_t i = 0; i < m.n_cols; ++i) v.col(i) = shiftings[i] ? arma::shift(m.col(i), shiftings[i]).eval() : m.col(i);
    return v;
}

template<typename T> arma::Col<T> stretch_crop(const arma::subview<T> &v, const double f, size_t n)
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

template<typename T> arma::Col<T> stretch(const arma::Col<T> &v, size_t n)
{
    return stretch(v, double(n) / double(v.n_elem));
}

template<typename T> std::vector<T> operator+(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
    const size_t output_len = std::min<size_t>(lhs.size(), rhs.size());
    std::vector<T> res(output_len);
#pragma omp parallel for simd num_threads(adj_threads(output_len))
    for (size_t i = 0; i < output_len; ++i)
        res[i] = lhs[i] + rhs[i];
    return res;
}

template<typename T> T snap_copy(const T v, const std::vector<T> &sorted_possible_values)
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

template<typename T, template<typename, typename> typename Container> T snap_inplace(T &v, const Container<T, std::allocator<T> > &sorted_possible_values)
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


template<typename T> T min(const T &arg1, const T &arg2, const T &arg3)
{
    T m = arg1;
    if (arg2 < m) m = arg2;
    if (arg3 < m) m = arg3;
    return m;
}

template<typename C, typename T> T max(const C &container)
{
    T max = 0;
    for (const auto &elem: container)
        if (elem > max)
            max = elem;
    return max;
}

template<typename T, typename U, typename E, typename F> void keep_indices(T &d, const U &indices)
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

template<typename T> arma::Mat<T> join_rows(const size_t arg_ct...)
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

template<typename T> __host__ __device__ inline T &scale_I(T &v, const double sf, const double dc)
{
    return v = (v - dc) / sf;
}

template<typename T> inline T unscale(const T &v, const double sf, const double dc)
{
    return v * sf + dc;
}

template<typename T> inline T &unscale_I(T &v, const double sf, const double dc)
{
    return v = v * sf + dc;
}

template<typename T> inline T constrain(T v, const T min, const T max)
{
    if (v < min) v = min;
    else if (v > max) v = max;
    return v;
}

template<typename T> arma::Mat<T> norm(const arma::subview<std::complex<T> > &cxm)
{
    arma::Mat<T> norm_cxm(arma::size(cxm));
#pragma omp parallel for num_threads(adj_threads(cxm.n_elem)) schedule(static, 1 + cxm.n_elem / C_n_cpu)
    for (size_t i = 0; i < cxm.n_elem; ++i) norm_cxm[i] = std::norm(cxm[i]);
    return norm_cxm;
}


template<typename T> arma::Mat<T> norm(const arma::Mat<std::complex<T> > &cxm)
{
    arma::Mat<T> norm_cxm(arma::size(cxm));
#pragma omp parallel for num_threads(adj_threads(cxm.n_elem)) schedule(static, 1 + cxm.n_elem / C_n_cpu)
    for (size_t i = 0; i < cxm.n_elem; ++i) norm_cxm[i] = std::norm(cxm[i]);
    return norm_cxm;
}

template<typename T> inline T mean_asymm(const arma::Mat<T> &m, const size_t last)
{
    return .5 * (arma::mean(arma::vectorise(m.submat(m.n_rows - last, m.n_cols - 1, m.n_rows - 2, m.n_cols - 1))) +
                 arma::mean(arma::vectorise(m.submat(m.n_rows - 1, m.n_cols - last, m.n_rows - 1, m.n_cols - 2))));
}

template<typename T> T meanabs(const arma::Mat<T> &m)
{
    return arma::mean(arma::abs(arma::vectorise(m)));
}

template<typename T> inline T medianabs(const arma::Mat<T> &m)
{
    return arma::median(arma::abs(arma::vectorise(m)));
}

template<typename T> inline T meanabs(const std::span<T> &v)
{
    return std::reduce(C_default_exec_policy, v.begin(), v.end(), 0., [](const double acc, const double val) { return acc + std::abs(val); }) / double(v.size());
}


template<typename T> inline T meanabs(const std::vector<T> &v)
{
    return meanabs(std::span(v));
}

template<typename T> T meanabs(const std::deque<T> &v)
{
    return std::reduce(C_default_exec_policy, v.cbegin(), v.cend(), 0., [](const double acc, const double val) { return acc + std::abs(val); }) / double(v.size());
}

template<typename scalar_t> scalar_t meanabs(const typename std::vector<scalar_t>::const_iterator &begin, const typename std::vector<scalar_t>::const_iterator &end)
{
    return std::reduce(C_default_exec_policy, begin, end, 0., [](const double acc, const double v) { return acc + std::abs(v); }) / double(end - begin);
}

template<template<typename> class T, typename S> inline arma::Mat<S> extrude_cols(const T<S> &m, const uint32_t ct)
{
    arma::Mat<S> r = m;
    while (r.n_cols < ct) r = arma::join_horiz(r, m);
    if (r.n_cols > ct) r.shed_cols(ct, r.n_cols - 1);
    return r;
}

template<template<typename> class T, typename S> inline arma::Mat<S> extrude_rows(const T<S> &m, const uint32_t ct)
{
    assert(m.n_rows > 0);
    arma::Mat<S> r(ct, m.n_cols);
    OMP_FOR(ct / m.n_rows)
    for (DTYPE(ct) start_i = 0; start_i < ct; start_i += m.n_rows) {
        const auto end_i = std::min<DTYPE(ct) >(ct, start_i + m.n_rows) - 1;
        r.rows(start_i, end_i) = m.rows(0, m.n_rows - 1);
    }
    return r;
}

template<template<typename> class T, typename S> inline arma::Mat<S> stretch_cols(const T<S> &m, const uint32_t ct)
{
    if (m.n_cols == ct) return m;
    assert(m.n_cols > 0 && ct > 0);
    arma::Mat<S> r(m.n_rows, ct, arma::fill::zeros);
    const auto coef = ct / float(m.n_cols);
    if (coef < 1) {
        arma::u32_vec div(ct);
        for (DTYPE(ct) i = 0; i < ct; ++i) {
            r.col(i) = m.col(i / coef);
            div[i] += 1;
        }
        for (DTYPE(ct) i = 0; i < ct; ++i)
            r.col(i) /= div[i];
    } else
        for (DTYPE(ct) i = 0; i < ct; ++i)
            r.col(i) = m.col(i / coef);

    return r;
}

template<template<typename> class T, typename S> inline arma::Mat<S> stretch_rows(const T<S> &m, const uint32_t ct)
{
    if (m.n_rows == ct) return m;
    assert(m.n_rows > 0 && ct > 0);
    arma::Mat<S> r(ct, m.n_cols, arma::fill::zeros);
    const auto coef = ct / float(m.n_rows);
    if (coef < 1) {
        arma::u32_vec div(ct);
        for (DTYPE(ct) i = 0; i < ct; ++i) {
            r.row(i) = m.row(i / coef);
            div[i] += 1;
        }
        for (DTYPE(ct) i = 0; i < ct; ++i)
            r.row(i) /= div[i];
    } else
        for (DTYPE(ct) i = 0; i < ct; ++i)
            r.row(i) = m.row(i / coef);

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
    arma::Col<T> vm((T *) m.memptr(), m.n_elem, true, true);
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
            arma::range(vm) << ", meanabs " << meanabs(vm) << ", values " << common::to_string(m.mem, std::min<size_t>(m.n_elem, 5));
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

template<typename T> std::string present(const arma::subview<T> &m)
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

template<typename scalar_t> scalar_t log_return(const scalar_t x, const scalar_t x_1)
{
    const auto res = std::log(x) - std::log(x_1);
#ifndef NDEBUG
    if (std::isnan(res))
        throw std::invalid_argument(
            "Log return is NaN of " + std::to_string(x) + " and " + std::to_string(x_1));
#endif
    return res;
}

template<typename scalar_t> scalar_t diff_return(const scalar_t x, const scalar_t prev_x)
{
    const auto res = x - prev_x;
#ifndef NDEBUG
    if (std::isnan(res))
        throw std::invalid_argument(
            "Diff return is NaN of " + std::to_string(x) + " and " + std::to_string(prev_x));
#endif
    return res;
}

template<typename scalar_t> scalar_t inv_log_return(const scalar_t log_ret_x, const scalar_t x_1
)
{
    const auto res = x_1 * std::exp(log_ret_x);
#ifndef NDEBUG
    if (std::isnan(res))
        throw std::invalid_argument(
            "Inverse log return is NaN of " + std::to_string(x_1) + " and " + std::to_string(log_ret_x));
#endif
    return res;
}

template<typename scalar_t> scalar_t inv_diff_return(const scalar_t diff_ret_x, const scalar_t prev_actual_x)
{
    const auto res = prev_actual_x + diff_ret_x;
#ifndef NDEBUG
    if (std::isnan(res))
        throw std::invalid_argument(
            "Inverse diff return is NaN of " + std::to_string(prev_actual_x) + " and " + std::to_string(diff_ret_x));
#endif
    return res;
}

template<typename T> std::vector<size_t> argsort(const std::vector<T> &v)
{
    std::vector<size_t> result(v.size());
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(), [&](size_t i, size_t j) { return v[i] < v[j]; });
    return result;
}

template<typename T> inline bool equals(const T t1, const T t2, const T epsilon)
{
    return fabs(t1 - t2) < epsilon;
}

template<typename T> inline void mirror_copy(T *out, const T *const in, const uint32_t n, const uint32_t half_n)
{
    assert(n % 2 == 0);
    for (uint32_t i = 0; i < n; ++i) out[i] = in[i < half_n ? i : n - i - 1];
}

template<typename T> inline bool above_eps(const T x)
{
    return std::abs(x) > std::numeric_limits<T>::epsilon();
}

}
}

#endif //MATH_UTILS_TPP
