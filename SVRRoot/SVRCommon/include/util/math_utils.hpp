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
template<typename A, typename B> using common_signed_t = std::conditional_t<std::is_unsigned_v<A> && std::is_unsigned_v<B>,
    std::common_type_t<A, B>,
    std::common_type_t<std::make_signed_t<A>, std::make_signed_t<B> > >;

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

template<typename Dividend, typename Divisor> inline constexpr common_signed_t<Dividend, Divisor> cdivi(Dividend x, Divisor y);

template<typename T1, typename T2> inline constexpr T1 cdiv(const T1 l, const T2 r) { return CDIV((l), (r)); }

std::vector<double> operator*(const std::vector<double> &v1, double &m);

std::vector<double> operator*(double &m, const std::vector<double> &v1);

std::vector<double> operator*(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator+(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator-(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> operator-(const std::vector<double> &v);

std::vector<double> operator^(const std::vector<double> &v, double a);

std::vector<double> operator^(double a, const std::vector<double> &v);

template<typename T> std::vector<T> &operator/=(std::vector<T> &lsh, const T rhs);

template<typename T> std::atomic<T> &operator+=(std::atomic<T> &lhs, const T &rhs);

template<typename T> std::atomic<T> &operator&=(std::atomic<T> &lhs, const T &rhs);

namespace common {
template<typename T> T absdiffratio(const T a, const T b);

template<typename T> T absdiffinvratio(const T a, const T b);

struct pseudo_random_dev
{
    static std::atomic<double> state;

    double operator()();

    static double max(double max);
};

/// @brief Returns a reproducibly seeded 64-bit RNG
template<typename RNG> RNG reproducibly_seeded_64();

size_t fft_len(size_t input_len);

arma::vec add_to_arma(const std::vector<double> &v1, const std::vector<double> &v2);

template<typename T> std::enable_if_t<std::is_integral_v<T>, T> bounce(const T v, const T lim);

template<typename T> std::enable_if_t<not std::is_integral_v<T>, T> bounce(const T v, const T lim);

constexpr size_t C_int_nan = 0xDeadBeef;
constexpr size_t C_ip_max = 0x100000000;

/**
 * Extend division reminder to vectors
 *
 * @param   a       Dividend
 * @param   n       Divisor
 */

template<typename T> arma::Mat<T> mod(const arma::Mat<T> &a, const T n);

#ifdef ENABLE_OPENCL
void cholesky_check(viennacl::matrix<double> &A);
#endif

// Sign-safe natural logarithm of scalar
template<typename T> inline std::enable_if_t<std::is_scalar_v<T>, T> sln(const T v);

// Sign-safe natural logarithm of arma matrix
template<typename T> inline arma::Mat<T> slog(const arma::Mat<T> &m);

// Sign-safe natural logarithm of arma matrix in-place
template<typename T> inline T slog_I(T &&m);

template<typename T> inline std::enable_if_t<std::is_scalar_v<T>, T> sexp(const T v);

// Sign-safe natural exponent of arma matrix
template<typename T> inline arma::Mat<T> sexp(const arma::Mat<T> &m);

// Sign-safe natural exponent of arma matrix in-place
template<typename T> inline T &sexp_I(T &&m);

template<typename T> bool isnormalz(const T v);

arma::vec levy(size_t d);

double randouble();

arma::cx_mat matlab_fft(const arma::mat &input);

arma::cx_mat matlab_fft(const arma::cx_mat &input);

arma::cx_mat fftshift(const arma::cx_mat &input);

arma::cx_mat matlab_ifft(const arma::cx_mat &input);

arma::cx_mat ifftshift(const arma::cx_mat &input);

void negate(double *const v, size_t len);

void equispaced(arma::mat &x0, const arma::mat &bounds, const arma::vec &pows, uint64_t sobol_ctr = 0);

double alpha(double reference_error, double prediction_error);

double mape(double absolute_error, double absolute_label);

template<typename T, typename U> double imprv(const T newv, const U oldv);

template<typename T> bool equal_to(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon());

template<typename T> bool less(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon());

template<typename T> bool greater(const T v1, const T v2, const T eps = std::numeric_limits<T>::epsilon());

template<typename T> bool is_contained(const T v, const T from, const T to, const T eps = std::numeric_limits<T>::epsilon());

template<typename T> bool is_zero(const T v, const T eps = std::numeric_limits<T>::epsilon());

double fdsum(CPTR(float) v, const unsigned len);

template<typename scalar_t> scalar_t sum(const std::vector<scalar_t> &v);

template<typename scalar_t> scalar_t sumabs(const std::vector<scalar_t> &v);

template<typename T> T sumabs(const arma::Mat<T> &m);

template<> double sumabs(const std::vector<double> &v);

template<> float sumabs(const std::vector<float> &v);

template<> double sumabs(const arma::Mat<double> &m);

template<> float sumabs(const arma::Mat<float> &m);

template<typename T> arma::uvec find(const arma::Mat<T> &m1, const arma::Mat<T> &m2);

template<typename T> arma::uvec find_ge(const arma::Mat<T> &m1, const arma::Mat<T> &m2);

template<typename T> arma::Mat<T> &normalize_cols_I(arma::Mat<T> &m);

template<typename T> arma::Mat<T> normalize_cols(const arma::Mat<T> &m);

template<typename T> arma::Col<T> &normalize_I(arma::Col<T> &v);

template<typename T> arma::Mat<T> &normalize_I(arma::Mat<T> &m);

template<typename scalar_t> void normalize(std::vector<scalar_t> &v, double min, double max);

template<typename scalar_t> std::vector<scalar_t> apply_norma(const std::vector<scalar_t> &values, size_t num_bins);

template<typename scalar_t> std::vector<scalar_t> skew_mask(const std::vector<scalar_t> &values, double exp, double mult);

template<typename scalar_t> std::vector<scalar_t> stretch_cut_mask(const std::vector<scalar_t> &values, double factor, double size_multiplier);

template<typename T> inline auto &shift_I(arma::Mat<T> &m, const arma::Col<unsigned> &shiftings);

template<typename T> inline arma::Mat<T> shift(const arma::Mat<T> &m, const arma::Col<unsigned> &shiftings);

template<typename T> arma::Col<T> inline stretch_crop(const arma::subview<T> &v, double f, size_t n = 0);

template<typename T> arma::Col<T> stretch(const arma::Col<T> &v, double f);

template<typename T> arma::Col<T> stretch(const arma::Col<T> &v, size_t n);

template<typename T> std::vector<T> operator+(const std::vector<T> &lhs, const std::vector<T> &rhs);

template<typename T> T snap_copy(const T v, const std::vector<T> &sorted_possible_values);

template<typename T, template<typename, typename> typename Container> T
snap_inplace(T &v, const Container<T, std::allocator<T> > &sorted_possible_values);

template<typename T> T min(const T &arg1, const T &arg2, const T &arg3);

bool is_power_of_two(size_t value);

size_t next_power_of_two(size_t value);

template<typename C, typename T = typename C::value_type> T max(const C &container);

double stdscore(double *const v, size_t len);

double meanabs_hiquant(double *const v, size_t len, double q);

double meanabs_loquant(double *const v, size_t len, double q);

double meanabs_quant(double *const v, size_t len, double q);

double mean_hiquant(double *const v, size_t len, double q);

double mean_loquant(double *const v, size_t len, double q);

double medianabs(double *const v, size_t len);

// Use for deque and vector containers
template<typename T, typename U, typename E = typename T::value_type, typename F = typename U::value_type> void keep_indices(T &d, const U &indices);

template<typename T> arma::Mat<T> join_rows(size_t arg_ct...);

__device__ __host__ __forceinline__ double scale(double v, double sf, double dc);

template<typename T> inline T scale(const T &v, double sf, double dc);

template<> arma::mat scale(const arma::mat &m, double sf, double dc);

template<typename T> __host__ __device__ inline T &scale_I(T &v, double sf, double dc);

template<> arma::mat &scale_I(arma::mat &m, double sf, double dc);

template<typename T> inline T unscale(const T &v, double sf, double dc);

template<typename T> inline T &unscale_I(T &v, double sf, double dc);

template<> arma::mat unscale(const arma::mat &m, double sf, double dc);

template<> arma::mat &unscale_I(arma::mat &m, double sf, double dc);

template<typename T> inline T constrain(T v, const T min, const T max);

template<typename T> arma::Mat<T> norm(const arma::subview<std::complex<T> > &cxm);

template<typename T> arma::Mat<T> norm(const arma::Mat<std::complex<T> > &cxm);

template<typename T> inline T mean_asymm(const arma::Mat<T> &m, size_t last);

template<typename T> T meanabs(const arma::Mat<T> &m);

template<> double meanabs<double>(const arma::Mat<double> &m);

template<> float meanabs<float>(const arma::Mat<float> &m);

template<typename T> inline T medianabs(const arma::Mat<T> &m);

template<typename T> inline T meanabs(const std::span<T> &v);

template<> double meanabs(const std::span<double> &v);

template<> float meanabs(const std::span<float> &v);

template<typename T> inline T meanabs(const std::vector<T> &v);

template<typename T> T meanabs(const std::deque<T> &v);

template<typename scalar_t> scalar_t meanabs(const typename std::vector<scalar_t>::const_iterator &begin, const typename std::vector<scalar_t>::const_iterator &end);

template<> double meanabs(const typename std::vector<double>::const_iterator &begin, const typename std::vector<double>::const_iterator &end);

template<template<typename> class T, typename S> inline arma::Mat<S> extrude_cols(const T<S> &m, uint32_t ct);

template<template<typename> class T, typename S> inline arma::Mat<S> extrude_rows(const T<S> &m, uint32_t ct);

template<template<typename> class T, typename S> inline arma::Mat<S> stretch_cols(const T<S> &m, uint32_t ct);

template<template<typename> class T, typename S> inline arma::Mat<S> stretch_rows(const T<S> &m, uint32_t ct);

template<typename T> std::stringstream present_s(const arma::Mat<T> &m);

template<typename T> std::string present(const arma::Mat<T> &m);

template<typename T> std::string present(const std::span<T> &v);

template<typename T> std::string present(const std::vector<T> &v);

template<typename T> std::string present(const arma::subview<T> &m);

std::string present_chunk(const arma::uvec &u, double head_factor);

#define INF (9.9e99)

std::vector<std::vector<double> > transpose_matrix(const std::vector<std::vector<double> > &vvmat);

template<typename scalar_t> scalar_t log_return(const scalar_t x, const scalar_t x_1);

template<typename scalar_t> scalar_t diff_return(const scalar_t x, const scalar_t prev_x);

template<typename scalar_t> scalar_t inv_log_return(const scalar_t log_ret_x, const scalar_t x_1);

template<typename scalar_t> scalar_t inv_diff_return(const scalar_t diff_ret_x, const scalar_t prev_actual_x);

double get_uniform_random_value();

std::vector<double> get_uniform_random_vector(size_t size);

std::vector<double> get_uniform_random_vector(const std::pair<std::vector<double>, std::vector<double> > &boundaries);

template<typename T> std::vector<size_t> argsort(const std::vector<T> &v);

double dot_product(double const *a, double const *b, size_t size);

template<typename T> inline bool equals(const T t1, const T t2, const T epsilon = std::numeric_limits<T>::epsilon());

void shuffle_matrix(const arma::mat &x, const arma::mat &y, arma::uvec &shuffled_rows, arma::uvec &shuffled_cols);

arma::mat pdist(const arma::mat &input);

template<typename T> T mean(const arma::Mat<T> &input);

#ifdef USE_IPP // IPP freezes when initialized in multiple shared objects

template<> double mean(const arma::Mat<double> &input);

template<> float mean(const arma::Mat<float> &input);

#endif // USE_IPP

double mean(double *const input, size_t len);

template<typename T> arma::Mat<T> mean_mask(const arma::Mat<T> &x, int32_t radius);

double max(const arma::mat &input);

double min(const arma::mat &input);

arma::mat shuffle_admat(const arma::mat &to_shuffle, size_t level);

arma::uvec fixed_shuffle(const arma::uvec &to_shuffle);

double calc_quant_offset_mul(double main_to_aux_period_ratio, double level, double levels_count);

double get_quantization_max(double main_to_aux_period_ratio);

arma::mat shuffle_admat(const arma::mat &to_shuffle, size_t level);

arma::uvec fixed_shuffle(const arma::uvec &to_shuffle);

arma::uvec complement_vectors(std::set<size_t> svi, arma::uvec new_ixs);

arma::uvec subview_indexes(arma::uvec batch_ixs, arma::uvec ix_tracker);

struct safe_double_less
{
    bool operator()(double left, double right) const;
};

template<typename T> inline void mirror_copy(T *out, const T *const in, uint32_t n, uint32_t half_n);

template<typename T> inline bool above_eps(const T x);
}
}

#include "math_utils.tpp"
