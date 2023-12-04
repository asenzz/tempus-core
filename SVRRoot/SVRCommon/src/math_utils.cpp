#include "util/math_utils.hpp"
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"

#include <viennacl/vector_proxy.hpp>
#include <common.hpp>
#include <map>
#include <cstdarg>

namespace svr {
namespace common {

std::vector<double> levy(const size_t d) // Return colvec of columns d
{
    constexpr double beta = 3. / 2.;
    static const double sigma = std::pow(tgamma(1. + beta) * sin(M_PI * beta / 2.) / (tgamma((1. + beta) / 2.) * beta * std::pow(2, ((beta - 1.) / 2.))), 1. / beta);

    arma::mat u(1, d, arma::fill::randn);
    u = u * sigma;
    arma::mat v(1, d, arma::fill::randn);
    arma::mat step = u / arma::pow(arma::abs(v), 1. / beta);
    std::vector<double> res;
    common::arma_to_vector(step, res);
    return res;
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
    __omp_pfor_i(0, N, output(i) = input((i + N / 2) % N));
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
    const int N = input.n_elem;
    __omp_pfor_i(0, N, output(i) = input((i + N - N / 2) % N));
    return output;
}

void mirror_tail(std::vector<double> &input, const size_t last_data_ct)
{
    LOG4_WARN("Adding mirrored tail " << input.size() - last_data_ct << " to data input size " << input.size() << ", last data count " << last_data_ct);
    if (input.size() < last_data_ct) LOG4_THROW("input.size() " << input.size() << " < last_data_ct " << last_data_ct);
    auto input_data = input.data();
    const auto input_size = input.size();
// #pragma omp target data map(tofrom: input_data[0:input_size]) map(to: last_data_ct, input_size)
// #pragma omp parallel for default(none) shared(input_data, input_size, last_data_ct)
    for (size_t i = 0; i < input_size - last_data_ct; ++i)
        input_data[input_size - last_data_ct - 1 - i] = input_data[input_size + ( (i / last_data_ct) % 2 ? -1 - i % last_data_ct : -last_data_ct + i % last_data_ct)];
}


arma::mat fmod(const arma::mat &a, const double n)
{
    return a - arma::floor(a / n) * n;
}

std::vector<std::vector<double>>
transpose_matrix(const std::vector<std::vector<double>> &vvmat)
{
    if (vvmat.empty()) return {};

    std::vector<std::vector<double>> result(vvmat[0].size(), std::vector<double>(vvmat.size()));
    __omp_pfor(j, 0, vvmat[0].size(),
        for (size_t i = 0; i < vvmat.size(); ++i)
            result[j][i] = vvmat[i][j];
    )

    return result;
}

std::set<size_t> get_adjacent_indexes(const size_t level, const double ratio, const size_t level_count)
{
    std::set<size_t> level_indexes;
    if (level_count == 0 or ratio == 0) return {level};
    //const size_t full_count = level_count * ratio - 1;
    const size_t half_count = (double(level_count) * ratio - 1.) / 2.;
    ssize_t min_index;
    ssize_t max_index;
    if (ratio == 1) { // TODO Hack, fix!
        min_index = 0;
        max_index = level_count - 1;
    } else {
        min_index = level - half_count;
        max_index = level + half_count;
    }
    if (min_index < 0) {
        max_index -= min_index;
        min_index -= min_index;
    }
    if (max_index >= ssize_t(level_count)) {
        min_index -= max_index - level_count + 1;
        max_index -= max_index - level_count + 1;
    }
    for (ssize_t level_index = min_index; level_index <= max_index; ++level_index) {
        if ((level_index >= 0 and level_index < ssize_t(level_count))
            and (level_index > ssize_t(level_count) / 2 or (level_index < ssize_t(level_count) / 2 and level_index % 2 == 0)))
            level_indexes.insert(level_index);
        else
            LOG4_TRACE("Skipping level " << std::to_string(level_index) << " adjacent ratio " << ratio << " for level " << level);
    }
    LOG4_TRACE("Adjacent ratio " << ratio << " for level " << level << " includes levels " << deep_to_string(level_indexes));
    return level_indexes;
}


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
    static std::random_device rd;  //Will be used to obtain a seed for the random number engine
    static std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
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
#pragma omp parallel for
    for (size_t i = 0; i < std::min<size_t>(boundaries.first.size(), boundaries.second.size()); ++i)
        random_vector[i] = random_vector[i] * (boundaries.second[i] - boundaries.first[i]) + boundaries.first[i];

    return random_vector;
}

std::vector<double> operator *(const std::vector<double> &v1, const double &m)
{
    std::vector<double> ret(v1.size());
    transform(v1.begin(), v1.end(), ret.begin(), [&m](double val) -> double { return val * m; });
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
    transform(v1.begin(), v1.end(), v2.begin(), ret.begin(),
              [](double val1, double val2) -> double { return val1 * val2; });
    return ret;
}

std::vector<double> operator+(const std::vector<double> &v1, const std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    std::vector<double> ret(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), ret.begin(),
              [](double val1, double val2) -> double { return val1 + val2; });
    return ret;
}

std::vector<double> operator-(const std::vector<double> &v1, const std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    std::vector<double> ret(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), ret.begin(),
              [](double val1, double val2) -> double { return val1 - val2; });
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

#ifdef VIENNACL_WITH_OPENCL

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

    for (auto const &vc : c) result += vc;

    return result;
}

#endif

namespace armd {

void check_mat(const arma::mat &input)
{
    for (size_t i = 0; i < input.n_rows; ++i) {
        for (size_t j = 0; j < input.n_cols; ++j) {
            if (fabs(input(i, j)) > 1000. || isinf(input(i, j))) {
                LOG4_ERROR("bad input at " << i << ", " << j << " = " << input(i, j));
                throw std::runtime_error("Error in check_mat");
            }
        }
    }
}

std::stringstream arma_to_sstream(const arma::mat &input, const std::string mat_name = "")
{
    std::stringstream ss;
    if (mat_name != "") ss << mat_name << std::endl;
    ss << "Size: " << input.n_rows << " " << input.n_cols << std::endl;
    for (size_t i = 0; i < input.n_rows; ++i) {
        for (size_t j = 0; j < input.n_cols; ++j)
            ss << input(i, j) << " ";
        ss << "\n";
    }
    return ss;
}

void print_mat(const arma::mat &input, const std::string mat_name)
{
    LOG4_INFO(arma_to_sstream(input, mat_name).str());
}


void serialize_mat(const std::string &filename, const arma::mat &input, const std::string mat_name)
{
    const auto &ss = arma_to_sstream(input, mat_name);
    LOG4_FILE(filename, ss.str());
}

void serialize_vec(const std::string &filename, const arma::uvec &input, const std::string vec_name)
{
    std::stringstream ss;
    if (vec_name != "")
        ss << vec_name << std::endl;
    ss << "Size: " << input.n_elem << std::endl;
    for (size_t i = 0; i < input.n_elem; ++i)
        ss << input(i) << " ";
    ss << "\n";
    LOG4_FILE(filename, ss.str());
}


//should replace for with /* cilk_ */ for for the following two functions
void copy_from_arma_to_vcl(const arma::mat &input, viennacl::matrix<double> &output)
{
    output.resize(input.n_rows, input.n_cols, false);
    __omp_pfor_i(0, input.n_rows, for (size_t j = 0; j < input.n_cols; ++j) output(i, j) = input(i, j) )
}

void copy_from_vcl_to_arma(const viennacl::matrix<double> &input, arma::mat &output)
{
    output.set_size(input.size1(), input.size2());
    __omp_tpfor_i(size_t, 0, input.size1(), for (size_t j = 0; j < input.size2(); ++j) output(i, j) = input(i, j) )
}

arma::uvec set_to_arma_uvec(const std::set<size_t> &input)
{
    std::vector<size_t> support_vector_indexes(input.begin(), input.end());
    return arma::conv_to<arma::uvec>::from(support_vector_indexes);
}

arma::uvec complement_vectors(const std::set<size_t> &svi, const arma::uvec &new_ixs)
{
    //arma::uvec complement_ixs;
    std::vector<long long unsigned int> complement_ixs;

    for (size_t i = 0; i < new_ixs.n_elem; ++i) {
        if (svi.find(new_ixs(i)) == svi.end()) {
            complement_ixs.push_back(new_ixs(i));
        }
    }
    LOG4_DEBUG("complement_ixs size svi size  " << complement_ixs.size() << " " << svi.size());
    arma::uvec ret_values(complement_ixs);
    return ret_values;
}

void shuffle_matrix(
        const arma::mat &x,
        const arma::mat &y,
        arma::uvec &shuffled_rows,
        arma::uvec &shuffled_cols)
{
    size_t x_train_rows = x.n_rows;
    size_t x_train_cols = x.n_cols;
    arma::uvec cols = arma::linspace<arma::uvec>(0, x_train_cols - 1, x_train_cols);
    {
        static std::mutex shuffle_mutex;
        std::scoped_lock shuffle_guard(shuffle_mutex);
        shuffled_cols = arma::shuffle(cols);
        shuffled_rows = arma::shuffle(arma::linspace<arma::uvec>(0, x_train_rows - 1, x_train_rows));
    }
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

double mean_all(const arma::mat &input)
{
    return accu(input) / input.n_rows / input.n_cols;
}

arma::mat rows(arma::mat &input, arma::uvec &ixs)
{
    arma::mat selected_rows(ixs.n_elem, input.n_cols);
    for (size_t i = 0; i < ixs.n_elem; ++i)
        selected_rows.row(i) = input.row(ixs[i]);
    return selected_rows;
}

arma::mat rows(const arma::mat &input, const std::set<size_t> &ixs)
{
    arma::mat selected_rows;
    size_t i = 0;
    for (auto ix : ixs) {
        selected_rows.insert_rows(i++, input.row(ix));
        //selected_rows.row(i++)  = input.row(ix);
    }
    return selected_rows;
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

void print_arma_sizes(const arma::mat &input, const std::string input_name)
{
    LOG4_DEBUG("Rows and columns for " << input_name << " are " << input.n_rows << " " << input.n_cols);
}

arma::mat shuffle_admat(const arma::mat &to_shuffle, const size_t level)
{
    constexpr int value = 9879871;
    arma::arma_rng::set_seed(value + level);
    static std::mutex mx;
    std::scoped_lock l(mx);
    return arma::shuffle(to_shuffle);
}

arma::mat fixed_shuffle(const arma::mat &to_shuffle)
{
    constexpr size_t prime_multiplier = 6700417;
    constexpr size_t remainer = 42;
    arma::mat result = to_shuffle;
    __omp_pfor_i (0, to_shuffle.n_elem, result((i * prime_multiplier + remainer) % to_shuffle.n_elem) = to_shuffle(i) )
    return result;
}


}//namespace armd

double calc_quant_offset_mul(const double main_to_aux_period_ratio, const double level, const double levels_count)
{
#ifdef QUANTIZE_FIXED
    return QUANTIZE_FIXED;
    if (level < levels_count / 2 - 4) return QUANTIZE_FIXED;
    else if (level == levels_count / 2 - 4) return 3;
    else return 1;
#elif defined(QUANTIZE_FIXED_MAX) && defined(QUANTIZE_FIXED_MIN)
    return (QUANTIZE_FIXED_MAX - QUANTIZE_FIXED_MIN) * std::pow(1. - level / levels_count, QUANTIZE_FEAT_EXP) + QUANTIZE_FIXED_MIN;
#else
    return (1. + (QUANTIZE_FEAT_MUL * main_to_aux_period_ratio - 1.) * (std::pow(1. - (level / levels_count), QUANTIZE_FEAT_EXP)));
#endif
}


} //namespace common
} //namespace svr
