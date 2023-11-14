#include "deprecated/r_matrix.h"
#include "deprecated/r_matrix_impl.hpp"
#include "util/math_utils.hpp"
#include <armadillo>

namespace svr::batch {

using namespace datamodel;

// TODO Unify all Q and R matrix related methods with ones in OnlineSVR class!

r_matrix_cpu::r_matrix_cpu(const svr::datamodel::vmatrix<double> *p_kernel_matrix,
                           const std::vector<int64_t> &sv_indices,
                           bool update_r_matrix_enabled)
        : R(new svr::datamodel::vmatrix<double>), Q(p_kernel_matrix), last_qxs_row(0), last_qxs_col(0),
          update_r_matrix_enabled_(update_r_matrix_enabled), logging_interval(0)
{

    viennacl::matrix<double> vcl_R(sv_indices.size() + 1, sv_indices.size() + 1,
                                   viennacl::context(viennacl::MAIN_MEMORY));
    //It is important to create vcl_R on the CPU, because of the copy_from
    r_matrix::compute_r_matrix(p_kernel_matrix, sv_indices, vcl_R);
    R->copy_from(vcl_R);

    //my_qxs.reserve(1600);
}

#ifdef R_MATRIX_DEBUG_OUTPUT
namespace {
template<typename T>
void print_buffer(const std::string &buf_name, cl12::CommandQueue & queue, cl::Buffer & buffer, size_t sz)
{
    std::ostringstream ostr;

    std::vector<T> buf(sz);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sz * sizeof(T), buf.data());

    ostr << buf_name << ": size: " << sz << " , values: ";

    for(size_t i = 0; i < std::min(1UL, buf.size()); ++i)
        ostr << buf[i];

    for(size_t i = 1; i < buf.size(); ++i)
        ostr << ", " << buf[i];

    ostr << std::endl;

    std::cout << ostr.str();
}

template<typename T>
void print_buffer(const std::string &buf_name, cl12::CommandQueue & queue, cl::Buffer & buffer, size_t cols, size_t rows)
{
    std::ostringstream ostr;

    std::vector<T> buf(cols * rows);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, cols * rows * sizeof(T), buf.data());

    ostr << buf_name << ": cols: " << cols << ": rows: " << rows << " , values: \n";

    for(size_t row = 0; row < rows; ++row)
    {
        for(size_t i = 0; i < std::min(1UL, cols); ++i)
            ostr << "\t" << buf[cols * row + i];
        for(size_t i = 1; i < cols; ++i)
            ostr << ", \t" << buf[cols * row + i];
        ostr << "\n";
    }

    ostr << std::endl;

    std::cout << ostr.str();
}


template<typename T>
void print_buffer(const std::string &buf_name, vektor<T> const & vec)
{
    std::ostringstream ostr;

    ostr << buf_name << ": size: " << vec.get_length() << " , values: ";

    for(int i = 0; i < std::min(1L, vec.get_length()); ++i)
        ostr << vec.get_value(i);

    for(int i = 1; i < vec.get_length(); ++i)
        ostr << ", " << vec.get_value(i);

    ostr << std::endl;

    std::cout << ostr.str();
}

template<typename T>
void print_buffer(const std::string &buf_name, vmatrix<T> const & matrix)
{
    std::ostringstream ostr;

    ostr << buf_name << ": cols: " << matrix.get_length_cols() << ", rows: " << matrix.get_length_rows() << ", values: \n";

    for(int row = 0; row < matrix.get_length_rows(); ++row)
    {
        for(int i = 0; i < std::min(1L, matrix[row].get_length()); ++i)
            ostr << "\t" << matrix[row][i];
        for(int i = 1; i < matrix[row].get_length(); ++i)
            ostr << ", \t" << matrix[row][i];
        ostr << "\n";
    }

    std::cout << ostr.str();
}
}
#endif

vektor<double> *r_matrix_cpu::get_qsi(std::vector<int64_t> const &sv_indexes, int64_t sample_index)
{
    auto V2 = new vektor<double>(sv_indexes.size());
    V2->resize(sv_indexes.size());
    auto const sz = sv_indexes.size() - 1;
    V2->set_at(0, 1);
    __omp_pfor_i(0, sz, V2->operator[](i + 1) = Q->get_value(sample_index, sv_indexes[i]));
    return V2;
}

vektor<double> *r_matrix_cpu::get_qxi(int sample_index)
{
    auto const sz = sample_index + 1;
    auto V = new vektor<double>(sz);
    V->resize(sz);
    __omp_pfor_i(0, sz, V->operator[](i) = Q->get_value(sample_index, i));
    return V;
}

/* vmatrix<double> const &r_matrix_cpu::get_qxs(std::vector<int64_t> const &sv_indexes, int64_t samples_trained_number)
{
// *** partial update
    update_qxs(sv_indexes, samples_trained_number);
    return my_qxs;
}
*/
/* 
namespace {
void grow_existing_rows(std::vector<int64_t> const &sv_indexes, const size_t last_qxs_row,
                        const size_t last_qxs_col, svr::datamodel::vmatrix<double> &my_qxs,
                        const svr::datamodel::vmatrix<double> *Q)
{
    size_t const cur_col = sv_indexes.size() - 1;
    for (size_t irow = 0; irow < last_qxs_row; ++irow) {
        my_qxs[irow].memresize(cur_col + 1);
        my_qxs[irow].resize(cur_col + 1);
        * cilk_ * for (auto icol = last_qxs_col; icol < cur_col; ++icol)my_qxs.set_value(irow, icol + 1, Q->get_value(irow, sv_indexes[icol]));
    }
}

void fill_new_rows(std::vector<int64_t> const &sv_indexes, const int64_t samples_trained_number,
                   const size_t last_qxs_row, svr::datamodel::vmatrix<double> &my_qxs,
                   const svr::datamodel::vmatrix<double> *Q)
{
    size_t const cur_col = sv_indexes.size() - 1;
    for (size_t irow = last_qxs_row; irow < size_t(samples_trained_number); ++irow)
        * cilk_ * for (size_t icol = 0; icol < cur_col; ++icol)
            my_qxs.set_value(irow, icol + 1, Q->get_value(irow, sv_indexes[icol]));
}
}
void r_matrix_cpu::update_qxs(std::vector<int64_t> const &sv_indexes, int64_t samples_trained_number)
{
    size_t const sz = sv_indexes.size() - 1;
    my_qxs.resize(samples_trained_number, 1.0, sz + 1);

    grow_existing_rows(sv_indexes, last_qxs_row, last_qxs_col, my_qxs, Q);
    fill_new_rows(sv_indexes, samples_trained_number, last_qxs_row, my_qxs, Q);

    last_qxs_row = size_t(samples_trained_number);
    last_qxs_col = sz;
}
*/

vektor<double> *r_matrix_cpu::get_beta(std::vector<int64_t> const &sv_indexes, int64_t samples_trained_number)
{
    const auto p_qsi = get_qsi(sv_indexes, samples_trained_number);
    auto p_beta = R->product_vector(p_qsi);
    delete p_qsi;
    p_beta->product_scalar(-1);
    return p_beta;
}

vektor<double> *r_matrix_cpu::get_gamma(
        std::vector<int64_t> const &sv_indexes,
        const ssize_t sample_index,
        vektor<double> *p_beta)
{
    const auto p_qsi = get_qsi(sv_indexes, sample_index);
    auto support_size = sv_indexes.size();
    vektor<double> *p_gamma = new vektor<double>(0., support_size);
    __omp_pfor_i (0, support_size, p_gamma->set_at(i, Q->get_value(sv_indexes[i], sv_indexes[i]) + p_beta->product_vector_scalar(*p_qsi)));
    delete p_qsi;
    return p_gamma;
}


void copy(viennacl::matrix<double> &out, const arma::Mat<double> &in)
{
    if (out.size1() < in.n_rows || out.size2() < in.n_cols)
        out.resize(in.n_rows, in.n_cols);
#pragma omp parallel for collapse(2) default(shared)
    for (size_t i = 0; i < in.n_rows; ++i)
        for (size_t j = 0; j < in.n_cols; ++j)
            out(i, j) = in(i, j);
}


void r_matrix::compute_r_matrix(
        const svr::datamodel::vmatrix<double> *p_kernel_matrix,
        const std::vector<int64_t> &sv_indices,
        viennacl::matrix<double> &result_matrix)
{
    arma::mat arma_input(sv_indices.size() + 1, sv_indices.size() + 1);
    arma_input(0, 0) = 0;
    __omp_pfor_i (0, sv_indices.size(),
        arma_input(0, i + 1) = 1;
        arma_input(i + 1, 0) = 1;
    )

#pragma omp parallel for collapse(2) default(shared) schedule(dynamic)
    for (size_t i = 0; i < sv_indices.size(); ++i)
        for (size_t j = 0; j < sv_indices.size(); ++j)
            arma_input(i + 1, j + 1) = p_kernel_matrix->get_value(sv_indices[i], sv_indices[j]);

    arma::mat arma_output = arma::inv(arma_input);
    svr::batch::copy(result_matrix, arma_output);
}


void r_matrix_cpu::update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t sample_index)
{
    LOG4_BEGIN();

    if (!update_r_matrix_enabled_) {
        LOG4_TRACE("Update R matrix disabled!");
        return;
    }

    if (!(++logging_interval % 100))
        LOG4_DEBUG(
                "Updating R matrix on: " << sv_indexes.size() << " instances, sample index: " << sample_index
                                         << " Q matrix size: " << Q->get_length_rows() << "x" << Q->get_length_cols()
                                         << " (99 messages skipped)");

    vektor<double> *p_beta(nullptr), *p_gamma(nullptr);
    //calc gamma
    if (sv_indexes.size() - 1 == 0) {
        p_gamma = new vektor<double>(1., sample_index + 1);
    } else {
        p_beta = get_beta(sv_indexes, sample_index);
        p_gamma = get_gamma(sv_indexes, sample_index, p_beta);
    }

    //add sample to R
    if (R->get_length_rows() == 0) {
        vektor<double> *V1 = new vektor<double>(2);
        V1->add_fast(-(*Q)[sample_index][sample_index]);
        V1->add_fast(1);
        R->add_row_ref(V1);
        vektor<double> *V2 = new vektor<double>(2);
        V2->add_fast(1);
        V2->add_fast(0);
        R->add_row_ref(V2);
    } else {
        vektor<double> *p_zeros = vektor<double>::zero_vector(R->get_length_cols());
        R->add_col_copy(p_zeros);
        p_zeros->add(0);
        R->add_row_ref(p_zeros);
        if (p_gamma->get_value(sample_index) != 0) {
            p_beta->add(1);
            double gamma_value = p_gamma->get_value(sample_index);
            size_t size_r_matrix = R->get_length_rows();
            __omp_pfor_i (0, size_r_matrix,
                          for (size_t j = 0; j < size_r_matrix; j++) {
                    R->set_value(i, j, R->get_value(i, j) + p_beta->get_value(i) * p_beta->get_value(j) / gamma_value);
                }
            )
        }
    }
    delete p_gamma;
    delete p_beta;

    LOG4_END();
}

size_t r_matrix_cpu::size_of() const
{
    size_t result = sizeof(*this);

    if (R) result += R->size_of();

    //result += Q->size_of() + my_qxs.size_of();
    //result += Q->size_of();

    return result;
}


/****************************************/
/* OpenCL implementation follows below. */
/****************************************/

#ifdef VIENNACL_WITH_OPENCL

r_matrix_gpu::r_matrix_gpu(const svr::datamodel::vmatrix<double> *p_kernel_matrix,
                           svr::datamodel::vmatrix<double> const &r_matrix, bool update_r_matrix_enabled)
        : update_r_matrix_enabled_(update_r_matrix_enabled), logging_interval(0),
          qm_cols(p_kernel_matrix->get_length_cols()), qm_rows(p_kernel_matrix->get_length_rows()),
          qm_buf_size(qm_cols * qm_rows * sizeof(double)), gpu_kernel("r_matrix_kernels"), ctx(gpu_kernel.ctx()),
          context(ctx.handle()), queue(ctx.get_queue().handle()),
          qm_pinned_buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, qm_buf_size, nullptr),
          qm_buf_start(
                reinterpret_cast<double *>(queue.enqueueMapBuffer(qm_pinned_buffer, CL_TRUE, CL_MAP_WRITE, 0,
                                                                  qm_buf_size))), r_size(r_matrix.get_length_cols()),
          r_pinned_buffer(nullptr)
{
    if (!qm_buf_start) throw std::runtime_error("Cannot allocate device buffer");

    /*cilk_*/for (size_t irow = 0; irow < qm_rows; ++irow)
        /*cilk_*/for (size_t icol = 0; icol < qm_cols; ++icol)
            qm_buf_start[irow * qm_cols + icol] = p_kernel_matrix->get_value(irow, icol);
            // *qm_buf_iter++ = p_kernel_matrix->get_value(irow, icol);

    int err = queue.enqueueWriteBuffer(qm_pinned_buffer, CL_FALSE, 0, qm_buf_size, qm_buf_start);
    CL_CHECK(err);

    const size_t r_buf_size = r_size * r_size * sizeof(double);

    r_pinned_buffer = new cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, r_buf_size, nullptr);

    double *const r_buf_start = reinterpret_cast<double *>(queue.enqueueMapBuffer(*r_pinned_buffer, CL_TRUE,
                                                                                  CL_MAP_WRITE, 0, r_buf_size));
    if (!r_buf_start) throw std::runtime_error("Cannot allocate device buffer");

    //double *r_buf_iter = r_buf_start;
    /*cilk_*/for (int i = 0; i < r_matrix.get_length_rows(); ++i)
        /*cilk_*/for (int j = 0; j < r_matrix.get_length_cols(); ++j)
            r_buf_start[i * r_matrix.get_length_cols() + j] = r_matrix[i][j];

    err = queue.enqueueWriteBuffer(*r_pinned_buffer, CL_FALSE, 0, r_buf_size, r_buf_start);
    CL_CHECK(err);

    err = queue.finish();
    CL_CHECK(err);
}


r_matrix_gpu::~r_matrix_gpu()
{
    queue.enqueueUnmapMemObject(qm_pinned_buffer, qm_buf_start);
    delete r_pinned_buffer;
}


void r_matrix_gpu::update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t sample_index)
{
    if (!update_r_matrix_enabled_)
        return;

    if (!(++logging_interval % 100))
        LOG4_DEBUG("Computing R matrix on: " << sv_indexes.size()
                                             << " instances, sample index: " << sample_index
                                             << " Q matrix size: " << qm_rows << "x" << qm_cols
                                             << " (99 messages skipped)");

    beta_gamma bg = build_beta_gamma(sv_indexes, sample_index);

    cl::Buffer *new_r_pinned_buffer = add_zeros_to_r(*r_pinned_buffer, int64_t(r_size), int64_t(r_size));
    ++r_size;

    delete r_pinned_buffer;
    r_pinned_buffer = new_r_pinned_buffer;

    if (bg.gamma_last != 0)
        rebuild_r(bg.beta, *r_pinned_buffer, int64_t(r_size), bg.gamma_last);
}

svr::datamodel::vmatrix<double> *r_matrix_gpu::get_r()
{
    svr::datamodel::vmatrix<double> *result = new svr::datamodel::vmatrix<double>(r_size);
    int err = 0;
    for (size_t row = 0; row < r_size; ++row) {
        vektor<double> *p_vec = new vektor<double>(r_size);
        p_vec->resize(r_size);
        result->set_row_ref_fast(p_vec, row);
        err = queue.enqueueReadBuffer(
                *r_pinned_buffer, CL_FALSE, row * r_size * sizeof(double),
                r_size * sizeof(double), &p_vec->operator[](0));
    }

    err = queue.finish();
    CL_CHECK(err);
    return result;
}

r_matrix_gpu::beta_gamma r_matrix_gpu::build_beta_gamma(std::vector<int64_t> const &sv_indexes, int64_t sample_index)
{
    beta_gamma result;

    queue.finish();

    size_t const beta_sz = sv_indexes.size();

    cl::Buffer sv_indexes_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, beta_sz * sizeof(sv_indexes[0]),
                                 (void *) sv_indexes.data());

    cl::Buffer qsi = build_qsi(sv_indexes_buffer, beta_sz, sample_index);

    result.beta = mul_mat_vec_invert(*r_pinned_buffer, r_size, r_size, qsi, 1);
    double one = 1.0;
    int err = queue.enqueueWriteBuffer(result.beta, CL_FALSE, r_size * sizeof(double), sizeof(double), &one);
    CL_CHECK(err);

    cl::Buffer qxi = build_qxi(sample_index + 1);

    cl::Buffer qxs = build_qxs(sv_indexes_buffer, beta_sz, sample_index + 1);

    result.gamma = mul_mat_vec(qxs, sample_index + 1, beta_sz, result.beta);

    add_vec_vec(result.gamma, sample_index + 1, qxi);

    err = queue.enqueueReadBuffer(result.gamma, CL_TRUE, sizeof(double) * sample_index, sizeof(double), &result.gamma_last);
    CL_CHECK(err);

    return result;
}


size_t r_matrix_gpu::size_of() const
{
    size_t result = sizeof(*this);

    result += r_size * r_size * sizeof(double);

    result += qm_buf_size;

    return result;
}

cl::Buffer r_matrix_gpu::build_qsi(cl::Buffer &sv_indexes, size_t sv_indexes_size, int64_t SampleIndex)
{
    cl::Buffer bqsi(context, CL_MEM_READ_WRITE, sv_indexes_size * sizeof(double));

    cl12::ocl_kernel build_qsi(ctx.get_kernel("r_matrix_kernels", "build_qsi").handle());

    build_qsi.set_args(qm_pinned_buffer, qm_cols, sv_indexes, sv_indexes_size, SampleIndex, bqsi)
            .enqueue(queue, cl::NullRange, sv_indexes_size, cl::NullRange);

    return bqsi;
}

cl::Buffer r_matrix_gpu::build_qxi(const int64_t sample_index)
{
    cl::Buffer bqxi(context, CL_MEM_READ_WRITE, sample_index * sizeof(double));

    cl12::ocl_kernel build_qxi(ctx.get_kernel("r_matrix_kernels", "build_qxi").handle());

    build_qxi.set_args(qm_pinned_buffer, qm_cols, sample_index - 1, bqxi)
            .enqueue(queue, cl::NullRange, sample_index, cl::NullRange);

    return bqxi;
}

cl::Buffer r_matrix_gpu::build_qxs(cl::Buffer &sv_indexes, size_t sv_indexes_size, int64_t SampleIndex)
{
    cl::Buffer bqxs(context, CL_MEM_READ_WRITE, sv_indexes_size * SampleIndex * sizeof(double));

    cl12::ocl_kernel build_qxs(ctx.get_kernel("r_matrix_kernels", "build_qxs").handle());

    build_qxs.set_args(qm_pinned_buffer, qm_cols, sv_indexes, sv_indexes_size, bqxs);

    build_qxs.enqueue(queue, cl::NullRange, SampleIndex, cl::NullRange);

    return bqxs;
}

cl::Buffer
r_matrix_gpu::mul_mat_vec_invert(cl::Buffer &matrix, size_t rows, size_t cols, cl::Buffer &vector, size_t size_delta)
{
    cl::Buffer result(context, CL_MEM_READ_WRITE,
                      (rows + size_delta) * sizeof(double)); // Later a value will be added to beta

    cl12::ocl_kernel mul_mat_vec_invert(ctx.get_kernel("r_matrix_kernels", "mul_mat_vec_invert").handle());

    mul_mat_vec_invert.set_args(matrix, rows, cols, vector, result)
            .enqueue(queue, cl::NullRange, rows, cl::NullRange);

    return result;
}

cl::Buffer r_matrix_gpu::mul_mat_vec(cl::Buffer &matrix, size_t rows, size_t cols, cl::Buffer &vector)
{
    cl::Buffer result(context, CL_MEM_READ_WRITE, rows * sizeof(double));

    cl12::ocl_kernel mul_mat_vec(ctx.get_kernel("r_matrix_kernels", "mul_mat_vec").handle());

    mul_mat_vec.set_args(matrix, rows, cols, vector, result)
            .enqueue(queue, cl::NullRange, rows, cl::NullRange);

    return result;
}

void r_matrix_gpu::add_vec_vec(cl::Buffer &in_out, size_t size, cl::Buffer &in)
{
    cl12::ocl_kernel add_vec_vec(ctx.get_kernel("r_matrix_kernels", "add_vec_vec").handle());

    add_vec_vec.set_args(in_out, in)
            .enqueue(queue, cl::NullRange, size, cl::NullRange);
}

cl::Buffer *r_matrix_gpu::add_zeros_to_r(cl::Buffer &old_r, int64_t r_old_rows, int64_t r_old_cols)
{
    cl::Buffer *result = new cl::Buffer(context, CL_MEM_READ_WRITE,
                                        (r_old_rows + 1) * (r_old_cols + 1) * sizeof(double));

    cl12::ocl_kernel add_zeros_to_r(ctx.get_kernel("r_matrix_kernels", "add_zeros_to_r").handle());

    add_zeros_to_r.set_args(old_r, *result)
            .enqueue(queue, cl::NullRange, cl::NDRange(r_old_rows, r_old_cols), cl::NullRange);

    return result;
}


void r_matrix_gpu::rebuild_r(cl::Buffer &beta, cl::Buffer &r, std::size_t rr_size, double gamma_sample_index)
{
    cl12::ocl_kernel rebuild_r(ctx.get_kernel("r_matrix_kernels", "rebuild_r").handle());

    rebuild_r.set_args(beta, r, gamma_sample_index)
            .enqueue(queue, cl::NullRange, cl::NDRange(rr_size, rr_size), cl::NullRange);
}

/******************************************************************************/
/* R Matrix CPU implementation. */
/******************************************************************************/

#endif

r_matrix::r_matrix(const svr::datamodel::vmatrix<double> *p_kernel_matrix, const std::vector<int64_t> &sv_indexes,
                   bool update_r_matrix_enabled)
        : cpu(*new r_matrix_cpu(p_kernel_matrix, sv_indexes, update_r_matrix_enabled)), gpu(nullptr)
{
}

r_matrix::~r_matrix()
{
    delete &cpu;
    delete gpu;
}


void r_matrix::update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t sample_index)
{
//    if(gpu)
//    {
//        gpu->update_r_matrix(sv_indexes, SampleIndex);
//        return;
//    }
//
//    if(SampleIndex > 1)
//    {
//        gpu = new r_matrix_gpu(cpu.Q, *cpu.R, cpu.update_r_matrix_enabled_);
//        gpu->update_r_matrix(sv_indexes, SampleIndex);
//        return;
//    }

    cpu.update_r_matrix(sv_indexes, sample_index);
}


svr::datamodel::vmatrix<double> *r_matrix::get_r()
{
#ifdef VIENNACL_WITH_OPENCL
    if (gpu) return gpu->get_r();
#endif
    return cpu.get_r();
}


} //svr
