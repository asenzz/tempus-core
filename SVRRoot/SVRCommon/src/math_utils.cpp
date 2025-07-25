#ifdef USE_IPP

#include <ipp.h>

#endif
#ifdef ENABLE_OPENCL
#include <viennacl/vector_proxy.hpp>
#endif

#include <common.hpp>
#include <map>
#include <mkl_vml.h>
#include "sobol.hpp"
#include <xoshiro.h>

namespace svr {
std::vector<double> operator*(const std::vector<double> &v1, const double &m)
{
    std::vector<double> ret(v1.size());
    std::transform(C_default_exec_policy, v1.begin(), v1.end(), ret.begin(), [m](const auto val) -> double { return val * m; });
    return ret;
}

std::vector<double> operator*(const double &m, const std::vector<double> &v1)
{
    return v1 * m;
}

std::vector<double> operator*(const std::vector<double> &v1, const std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    std::vector<double> ret(v1.size());
    std::transform(C_default_exec_policy, v1.begin(), v1.end(), v2.begin(), ret.begin(),
                   [](const auto val1, const auto val2) -> double { return val1 * val2; });
    return ret;
}

std::vector<double> operator+(const std::vector<double> &v1, const std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    std::vector<double> ret(v1.size());
    std::transform(C_default_exec_policy, v1.begin(), v1.end(), v2.begin(), ret.begin(),
                   [](const auto val1, const auto val2) -> double { return val1 + val2; });
    return ret;
}

std::vector<double> operator-(const std::vector<double> &v1, const std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    std::vector<double> ret(v1.size());
    std::transform(C_default_exec_policy, v1.begin(), v1.end(), v2.begin(), ret.begin(),
                   [](const auto val1, const auto val2) -> double { return val1 - val2; });
    return ret;
}

std::vector<double> operator-(const std::vector<double> &v)
{
    std::vector<double> result{v};
    std::for_each(result.begin(), result.end(), [](double &x) { x *= -1.0; });

    return result;
}

std::vector<double> operator^(const std::vector<double> &v, const double a)
{
    assert(a >= 0);
    std::vector<double> result{v};
    std::for_each(result.begin(), result.end(), [=](double &x) { x = std::pow(x, a); });

    return result;
}

std::vector<double> operator^(const double a, const std::vector<double> &v)
{
    assert(a >= 0);
    std::vector<double> result{v};
    std::for_each(result.begin(), result.end(), [=](double &x) { x = std::pow(a, x); });

    return result;
}

namespace common {
double pseudo_random_dev::operator()()
{
    state = std::fmod(state + .123456789, 1.);
    return state;
}

double pseudo_random_dev::max(const double max)
{
    state = std::fmod(state + .123456789, 1.);
    return max * state;
}

std::string present_chunk(const arma::uvec &u, const double head_factor)
{
    const size_t head_n = u.n_rows * head_factor;
    std::stringstream s;
    s << "size " << arma::size(u);
    if (u.n_elem < 20)
        s << ", elements " << to_string(u, u.n_elem);
    else
        s << ", head len " << head_n << ", tail len " << u.n_rows - head_n <<
                ", head " << to_string(u, 0, 5) <<
                ", cross-over " << to_string(u, head_n - 2, 5) <<
                ", tail " << to_string(u, u.n_elem - 5, 5);
    return s.str();
}

arma::vec levy(const size_t d) // Return colvec of columns d
{
    constexpr double beta = 3. / 2.;
    constexpr double beta_1 = 1. / beta;
    static const double sigma = std::pow(tgamma(1. + beta) * sin(M_PI * beta / 2.) / (tgamma((1. + beta) / 2.) * beta * std::pow(2, ((beta - 1.) / 2.))), 1. / beta);

    arma::vec u(d, arma::fill::randn), v(d, arma::fill::randn);
    return sigma * u / arma::pow(arma::abs(v), beta_1);
}

double randouble()
{
    static std::once_flag flag1;
    std::call_once(flag1, []() { srand48(9879798797); });
    return drand48();
}


arma::cx_mat matlab_fft(const arma::mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    //use this for real matrix
    return arma::conj(arma::fft(input));
}


arma::cx_mat
matlab_fft(const arma::cx_mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    //use this for complex matrix
    return arma::conj(arma::fft(input));
}


arma::cx_mat
fftshift(const arma::cx_mat &input)
{
    arma::cx_mat output(arma::size(input));
    const int N = input.n_elem;
    omp_pfor_i__(0, N, output(i) = input((i + N / 2) % N));
    return output;
}


arma::cx_mat
matlab_ifft(const arma::cx_mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    return arma::ifft(arma::conj(input));
}


arma::cx_mat
ifftshift(const arma::cx_mat &input)
{
    arma::cx_mat output(arma::size(input));
    const auto N = input.n_elem;
#pragma omp parallel for schedule(static, 1 + N / C_n_cpu) num_threads(adj_threads(N))
    for (size_t i = 0; i < N; ++i)
        output(i) = input((i + N - N / 2) % N);
    return output;
}


std::vector<std::vector<double> >
transpose_matrix(const std::vector<std::vector<double> > &vvmat)
{
    if (vvmat.empty()) return {};

    std::vector<std::vector<double> > result(vvmat[0].size(), std::vector<double>(vvmat.size()));
    omp_pfor__(j, 0, vvmat[0].size(),
               for (size_t i = 0; i < vvmat.size(); ++i)
               result[j][i] = vvmat[i][j];
    )

    return result;
}

#ifdef ENABLE_OPENCL

void cholesky_check(viennacl::matrix<double> &A)
{
    const int n = A.size1() / 2;
    double *L = (double *) calloc(n * n, sizeof(double));
    if (L == nullptr) LOG4_THROW("Can not allocate memory for Cholesky matrix!");

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];

            double v = A(i, j) - s;

            if (i == j) {
                if (v <= 0) {
                    free(L);
                    LOG4_THROW("Kernel Matrix is not semi-positive definite.");
                }
                L[i * n + j] = sqrt(v);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j]) * v;
            }
        }
    free(L);
}

#endif

bool is_power_of_two(size_t value)
{
    return value != 0 && !(value & (value - 1));
}

size_t next_power_of_two(size_t value)
{
    size_t next_power_of_two = 1;
    while (next_power_of_two < value) next_power_of_two <<= 1;
    return next_power_of_two;
}


double get_uniform_random_value()
{
    static std::random_device rd; // Will be used to obtain a seed for the random number engine
    static std::mt19937_64 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    static std::mutex uniform_random_value_mutex;
    std::scoped_lock ml(uniform_random_value_mutex);
    std::uniform_real_distribution<double> dis(0., 1.);
    const auto rand_num = dis(gen);
    return rand_num;
}


std::vector<double> get_uniform_random_vector(const size_t size)
{
    std::vector<double> result(size);
    for_each(result.begin(), result.end(), [](double &val) { val = get_uniform_random_value(); });
    return result;
}

std::vector<double>
get_uniform_random_vector(const std::pair<std::vector<double>, std::vector<double> > &boundaries)
{
    std::vector<double> random_vector{get_uniform_random_vector(boundaries.first.size())};
    const auto l = std::min<size_t>(boundaries.first.size(), boundaries.second.size());
#pragma omp parallel for num_threads(adj_threads(l))
    for (size_t i = 0; i < l; ++i)
        random_vector[i] = random_vector[i] * (boundaries.second[i] - boundaries.first[i]) + boundaries.first[i];

    return random_vector;
}

arma::vec add_to_arma(const std::vector<double> &v1, const std::vector<double> &v2)
{
    arma::vec ret(v1.size());
    std::transform(C_default_exec_policy, v1.begin(), v1.end(), v2.begin(), ret.begin(), [](const auto val1, const auto val2) { return val1 + val2; });
    return ret;
}

#ifdef ENABLE_OPENCL

double dot_product(double const *a, double const *b, const size_t length)
{
    double result = 0;

    static gpu_kernel kernels("blas_kernels");
    viennacl::ocl::context &ctx = kernels.ctx();

    viennacl::ocl::kernel &kernel_1 = ctx.get_kernel("blas_kernels", "dot_product");

    svr::cl12::ocl_context context(ctx.handle());

    cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, length * sizeof(double), (void *) a);
    cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, length * sizeof(double), (void *) b);
    cl::Buffer C(context, CL_MEM_READ_WRITE, length * sizeof(double));

    svr::cl12::ocl_kernel dot_product(kernel_1.handle());

    dot_product.setArg(0, A);
    dot_product.setArg(1, B);
    dot_product.setArg(2, C);

    svr::cl12::command_queue queue(ctx.get_queue().handle());

    cl::NDRange glob_range(length / 16);

    queue.enqueueNDRangeKernel(dot_product, cl::NullRange, glob_range, cl::NullRange);

    std::vector<double> c(length, 0);

    queue.enqueueReadBuffer(C, CL_TRUE, 0, length * sizeof(double), c.data());

    for (auto const &vc: c) result += vc;

    return result;
}

#endif

arma::uvec complement_vectors(const std::set<size_t> &svi, const arma::uvec &new_ixs)
{
    //arma::uvec complement_ixs;
    std::vector<arma::uword> complement_ixs;
    for (size_t i = 0; i < new_ixs.n_elem; ++i)
        if (svi.find(new_ixs(i)) == svi.end())
            complement_ixs.emplace_back(new_ixs(i));
    LOG4_DEBUG("complement_ixs size svi size  " << complement_ixs.size() << " " << svi.size());
    arma::uvec ret_values(complement_ixs);
    return ret_values;
}

arma::uvec subview_indexes(const arma::uvec &batch_ixs, const arma::uvec &ix_tracker)
{
    arma::uvec real_ixs(batch_ixs.n_elem);
    size_t current_ix = 0;
    for (size_t i = 0; i < batch_ixs.n_elem; ++i) {
        for (size_t j = 0; j < ix_tracker.n_elem; ++j) {
            if (batch_ixs(i) == ix_tracker(j))
                real_ixs(current_ix++) = j;
        }
    }
    return real_ixs;
}


void shuffle_matrix(
    const arma::mat &x,
    const arma::mat &y,
    arma::uvec &shuffled_rows,
    arma::uvec &shuffled_cols)
{
    static std::mutex shuffle_mutex;
    arma::uvec cols = arma::linspace<arma::uvec>(0, x.n_cols - 1, x.n_rows);
    const std::scoped_lock shuffle_guard(shuffle_mutex);
    shuffled_cols = arma::shuffle(cols);
    shuffled_rows = arma::shuffle(arma::linspace<arma::uvec>(0, x.n_rows - 1, x.n_rows));
}

arma::mat pdist(const arma::mat &input)
{
    arma::mat foo(1, (input.n_rows * (input.n_rows - 1)) / 2);
    size_t k = 0;
    for (size_t i = 0; i < input.n_rows - 1; ++i) {
        for (size_t j = i + 1; j < input.n_rows; ++j) {
            foo(k) = norm((input.row(i) - input.row(j)), 2);
            ++k;
        }
    }
    return foo;
}

template<>
double sumabs(const std::vector<double> &v) // TODO Unify *ASUM usage and MAT and Vector iterator usage
{
    return cblas_dasum(v.size(), v.data(), 1);
}

template<>
float sumabs(const std::vector<float> &v)
{
    return cblas_sasum(v.size(), v.data(), 1);
}

template<>
double sumabs(const arma::Mat<double> &m)
{
    return cblas_dasum(m.n_elem, m.mem, 1);
}

template<>
float sumabs(const arma::Mat<float> &m)
{
    return cblas_sasum(m.n_elem, m.mem, 1);
}

template<>
double meanabs(const arma::Mat<double> &m) // TODO Use iterator based functions
{
    return cblas_dasum(m.n_elem, m.mem, 1) / m.n_elem;
}

template<>
float meanabs(const arma::Mat<float> &m) // TODO Use iterator based functions
{
    return cblas_sasum(m.n_elem, m.mem, 1) / m.n_elem;
}

template<>
double meanabs(const typename std::vector<double>::const_iterator &begin, const typename std::vector<double>::const_iterator &end)
{
    const auto n = end - begin;
    return cblas_dasum(n, &*begin, 1) / n;
}

template<>
float meanabs(const typename std::vector<float>::const_iterator &begin, const typename std::vector<float>::const_iterator &end)
{
    const auto n = end - begin;
    return cblas_sasum(n, &*begin, 1) / n;
}

template<>
double meanabs<double>(const std::span<double> &v)
{
    return cblas_dasum(v.size(), v.data(), 1) / v.size();
}

template<>
float meanabs(const std::span<float> &v)
{
    return cblas_sasum(v.size(), v.data(), 1) / v.size();
}

double fdsum(CPTR(float) v, const unsigned len)
{
    double r = 0;
    OMP_FOR_(len, simd reduction(+:r) firstprivate(len) shared(v))
    for (unsigned i = 0; i < len; ++i) r += v[i];
    return r;
}

double max(const arma::mat &input)
{
#ifdef USE_IPP // IPP freezes on init
    double r;
    ip_errchk(ippsMax_64f(input.mem, input.n_elem, &r));
    return r;
#else
    return input.max();
#endif
}

double min(const arma::mat &input)
{
#ifdef USE_IPP
    double r;
    ip_errchk(ippsMin_64f(input.mem, input.n_elem, &r));
    return r;
#else
    return input.min();
#endif
}

#ifdef USE_IPP // IPP freezes when initialized in multiple shared objects

#define TSCALAR double

template<> double mean(const arma::Mat<TSCALAR> &input)
{
    TSCALAR r;
    ip_errchk(ippsMean_64f(input.mem, input.n_elem, &r));
    return r;
}

#undef TSCALAR
#define TSCALAR float

template<> TSCALAR mean(const arma::Mat<TSCALAR> &input)
{
    TSCALAR r;
    ip_errchk(ippsMean_32f(input.mem, input.n_elem, &r, IppHintAlgorithm::ippAlgHintAccurate));
    return r;
}

#undef TSCALAR

#endif

double mean(const double *const input, const size_t len)
{
#ifdef USE_IPP // IPP freezes when initialized in multiple shared objects
    double r;
    ip_errchk(ippsMean_64f(input, len, &r));
    LOG4_TRACE("Returning mean " << r << " for " << common::to_string(input, std::min<size_t>(len, 5)));
    return r;
#else
    std::accumulate(C_default_exec_policy, input, input + len, 0.0) / len;
#endif
}

double stdscore(const double *const v, const size_t len)
{
    const auto meanabs = cblas_dasum(len, v, 1) / len;
    double stddev;
    ip_errchk(ippsStdDev_64f(v, len, &stddev));
    return meanabs * std::pow(stddev, .1);
}

double medianabs(double *const v, const size_t len)
{
    ip_errchk(ippsAbs_64f_I(v, len));
    ip_errchk(ippsSortAscend_64f_I(v, len));
    const auto div_r = std::ldiv(len, 2);
    if (div_r.rem) return (v[div_r.quot] + v[div_r.quot + 1]) / 2.;
    else return v[div_r.quot];
}


inline double *const abssort(const double *const v, const size_t len)
{
    assert(v);
    assert(len);
    auto tmp = (double *const) ALIGNED_ALLOC_(MEM_ALIGN, len * sizeof(double));
    memcpy(tmp, v, len * sizeof(double));
    ip_errchk(ippsAbs_64f_I(tmp, len));
    ip_errchk(ippsSortAscend_64f_I(tmp, len));
    return tmp;
}

double meanabs_hiquant(const double *const v, const size_t len, const double q)
{
    const auto tmp = common::abssort(v, len);
    double r;
    ip_errchk(ippsMean_64f(tmp + ptrdiff_t(len * (1 - q)), len * q, &r));
    ALIGNED_FREE_(tmp);
    return r;
}

double meanabs_loquant(const double *const v, const size_t len, const double q)
{
    const auto tmp = common::abssort(v, len);
    double r;
    ip_errchk(ippsMean_64f(tmp, len * q, &r));
    ALIGNED_FREE_(tmp);
    return r;
}

double meanabs_quant(const double *const v, const size_t len, const double q)
{
    const auto tmp = common::abssort(v, len);
    double r;
    ip_errchk(ippsMean_64f(tmp + ptrdiff_t((1. - q) * len / 2.), len * q, &r));
    ALIGNED_FREE_(tmp);
    return r;
}


inline double *const sort(const double *const v, const size_t len)
{
    assert(v);
    assert(len);
    auto tmp = (double *const) ALIGNED_ALLOC_(MEM_ALIGN, len * sizeof(double));
    memcpy(tmp, v, len * sizeof(double));
    ip_errchk(ippsAbs_64f_I(tmp, len));
    ip_errchk(ippsSortAscend_64f_I(tmp, len));
    return tmp;
}

double mean_hiquant(const double *const v, const size_t len, const double q)
{
    const auto tmp = common::sort(v, len);
    double r;
    ip_errchk(ippsMean_64f(tmp + ptrdiff_t(len * (1 - q)), len * q, &r));
    ALIGNED_FREE_(tmp);
    return r;
}

double mean_loquant(const double *const v, const size_t len, const double q)
{
    const auto tmp = common::sort(v, len);
    double r;
    ip_errchk(ippsMean_64f(tmp, len * q, &r));
    ALIGNED_FREE_(tmp);
    return r;
}

arma::mat shuffle_admat(const arma::mat &to_shuffle, const size_t level)
{
    constexpr int value = 9879871;
    arma::arma_rng::set_seed(value + level);
    static std::mutex mx;
    const std::scoped_lock l(mx);
    return arma::shuffle(to_shuffle);
}

arma::uvec fixed_shuffle(const arma::uvec &to_shuffle)
{
    constexpr size_t prime_multiplier = 6700417;
    constexpr size_t remainder = 42;
    auto result = to_shuffle;
    OMP_FOR_i(to_shuffle.size()) result((i * prime_multiplier + remainder) % to_shuffle.n_elem) = to_shuffle(i);
    return result;
}

template<>
arma::mat scale(const arma::mat &m, const double sf, const double dc)
{
    arma::mat r(arma::size(m), ARMA_DEFAULT_FILL);
#ifdef USE_IPP
    ip_errchk(ippsNormalize_64f(m.mem, r.memptr(), m.n_elem, dc, sf));
#else
    vdLinearFrac(m.n_elem, m.mem, m.mem, 1., -dc, 0, sf, r.memptr());
#endif
    return r;
}

template<>
arma::mat &scale_I(arma::mat &m, const double sf, const double dc)
{
#ifdef USE_IPP
    ip_errchk(ippsNormalize_64f_I(m.memptr(), m.n_elem, dc, sf));
#else
    vdLinearFrac(m.n_elem, m.mem, m.mem, 1., -dc, 0, sf, m.memptr());
#endif
    return m;
}

template<>
arma::mat unscale(const arma::mat &m, const double sf, const double dc)
{
    arma::mat r(arma::size(m), ARMA_DEFAULT_FILL);
    vdLinearFrac(m.n_elem, m.mem, m.mem, sf, dc, 0, 1, r.memptr());
    return r;
}

template<>
arma::mat &unscale_I(arma::mat &m, const double sf, const double dc)
{
    vdLinearFrac(m.n_elem, m.mem, m.mem, sf, dc, 0, 1, m.memptr());
    return m;
}

void negate(double *const v, const size_t len)
{
    DTYPE(len) i = 0;
#ifdef NDEBUG
    __m256d neg_factor = _mm256_set1_pd(-1.);

    // Process 4 doubles at a time
    for (; i + 3 < len; i += 4) {
        __m256d data = _mm256_loadu_pd(&v[i]);
        __m256d result = _mm256_mul_pd(data, neg_factor);
        _mm256_storeu_pd(&v[i], result);
    }
#endif
    // Handle remaining elements
    for (; i < len; ++i) v[i] = -v[i];
}

std::atomic<double> pseudo_random_dev::state = .54321;

constexpr uint32_t max_D = 64;

// #define INIT_SOBOL
#define PERFECT_REPRODUCIBILITY

void equispaced(arma::mat &x0, const arma::mat &bounds, const arma::vec &pows, uint64_t sobol_ctr)
{
    LOG4_BEGIN();

#ifdef INIT_SOBOL
    if (!sobol_ctr) sobol_ctr = init_sobol_ctr();
#endif
    const uint32_t n = x0.n_cols;
    const uint32_t D = x0.n_rows;
    if (bounds.n_cols != 2 || bounds.n_rows != D)
        THROW_EX_FS(std::invalid_argument, "n " << n << ", D " << D << ", bounds " << arma::size(bounds));
    const arma::vec range = bounds.col(1) - bounds.col(0);
#ifndef PERFECT_REPRODUCIBILITY
    const auto init_random = n < 2;
#endif
    const auto n_1 = std::max<long double>(1, n - 1);
    const auto seed_pseud_rand = (uint_fast64_t) pseudo_random_dev::max(1.);
    // auto gen = reproducibly_seeded_64<xso::rng64>();
    auto gen = std::mt19937_64(seed_pseud_rand);
    std::uniform_real_distribution<double> dis(0., 1.);
#ifdef PERFECT_REPRODUCIBILITY
    OMP_FOR_(n, ordered firstprivate(n, D, sobol_ctr, max_D, n_1))
#else
    OMP_FOR_(n, ordered firstprivate(n, D, init_random, sobol_ctr, max_D, n_1))
#endif
    for (uint32_t i = 0; i < n; ++i) {
        const auto i_n_1 = (i % max_D) / n_1;
        for (uint32_t j = 0; j < D; ++j) {
            const auto pow_j = pows.empty() ? 1. : pows[j];
#ifdef PERFECT_REPRODUCIBILITY
            if constexpr (false)
#else
            if (init_random || j >= max_D) // Use a pseudo-random number
#endif
#ifdef INIT_SOBOL
                x0(j, i) = sobolnum(j, sobol_ctr + i);
#else
#pragma omp ordered
                x0(j, i) = dis(gen);
#endif
            else // Produce equidistant grid of parameter combinations
                x0(j, i) = std::fmod<long double>(std::pow<long double>(2, j % max_D) * i_n_1, 1);
#ifndef NDEBUG
            LOG4_TRACE("Particle " << i << ", parameter " << j << ", value " << x0(j, i) << ", ub " << bounds(j, 1) << ", lb " << bounds(j, 0) << ", pow " << pow_j);
#endif
            x0(j, i) = std::pow(range[j], pow_j) * x0(j, i) + bounds(j, 0);
        }
    }
    if (x0.has_nonfinite())
        LOG4_THROW("Illegal init values in " << present(x0));
    LOG4_TRACE("Scaled particles, parameters matrix " << to_string(x0, 6) << ", bounds " << to_string(bounds, 6));
}

double alpha(const double reference_error, const double prediction_error)
{
    return 100 * (reference_error / prediction_error - 1);
    // return 100 * (1 - prediction_error / reference_error); // Alternative way to calculate alpha
}

double mape(const double absolute_error, const double absolute_label)
{
    return 100. * absolute_error / absolute_label;
}

size_t fft_len(const size_t input_len)
{
    return input_len / 2 + 1;
}

bool safe_double_less::operator()(const double left, const double right) const
{
    const auto leftNaN = std::isnan(left);
    const auto rightNaN = std::isnan(right);
    return leftNaN != rightNaN ? leftNaN < rightNaN : left < right;
}
} //namespace common
} //namespace svr
