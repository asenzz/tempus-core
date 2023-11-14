#ifndef R_MATRIX_IMPL_HPP
#define R_MATRIX_IMPL_HPP

#include "r_matrix.h"
#include "common/gpu_handler.hpp"

class RMatrixTests_RMatrixContainer_Test;
class RMatrixTests_RMatrixMethods_Test;
class RMatrixGpuTests_RMatrixMethods_Test;
class RMatrixGpuTests_RMatrixMethods1_Test;

namespace svr {
namespace batch {

class r_matrix;

class r_matrix_cpu
{
    friend class ::RMatrixTests_RMatrixContainer_Test;
    friend class ::RMatrixTests_RMatrixMethods_Test;
    friend class ::RMatrixGpuTests_RMatrixMethods_Test;
    friend class ::RMatrixGpuTests_RMatrixMethods1_Test;
    friend class r_matrix;
public:
    explicit r_matrix_cpu(const svr::datamodel::vmatrix<double> * p_kernel_matrix, const std::vector<int64_t> &sv_indexes, bool update_r_matrix_enabled = true);
    void update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t SampleIndex);
    svr::datamodel::vmatrix<double> *get_r() {return R;}

    size_t size_of() const;
private:
    svr::datamodel::vektor<double>* get_qsi(std::vector<int64_t> const &sv_indexes, int64_t SampleIndex);
    svr::datamodel::vektor<double>* get_qxi(int SampleIndex);
   // svr::datamodel::vmatrix<double> const & get_qxs(std::vector<int64_t> const &sv_indexes,
                                                    //int64_t SamplesTrainedNumber);
    //void update_qxs(std::vector<int64_t> const &sv_indexes, int64_t SamplesTrainedNumber);
    svr::datamodel::vektor<double>* get_beta(std::vector<int64_t> const &sv_indexes, int64_t SamplesTrainedNumber);
    svr::datamodel::vektor<double>* get_gamma(std::vector<int64_t> const &sv_indexes, const ssize_t sample_index, svr::datamodel::vektor<double> *beta);

    svr::datamodel::vmatrix<double> *R;
    const svr::datamodel::vmatrix<double> *Q;
    //svr::datamodel::vmatrix<double> my_qxs;
    size_t last_qxs_row, last_qxs_col;

    bool update_r_matrix_enabled_;
    size_t logging_interval;
};


class r_matrix_gpu
{
#ifdef VIENNACL_WITH_OPENCL
    friend class ::RMatrixGpuTests_RMatrixMethods_Test;
    friend class r_matrix;
public:
    explicit r_matrix_gpu(const svr::datamodel::vmatrix<double> *p_kernel_matrix, svr::datamodel::vmatrix<double> const & r_matrix, bool update_r_matrix_enabled = true);
    ~r_matrix_gpu();

    void update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t sample_index);
    svr::datamodel::vmatrix<double> *get_r();

    size_t size_of() const;
private:
    bool update_r_matrix_enabled_;
    size_t logging_interval;
    size_t const qm_cols, qm_rows, qm_buf_size;
    svr::common::gpu_kernel gpu_kernel;
    viennacl::ocl::context ctx;
    svr::cl12::ocl_context context;
    svr::cl12::command_queue queue;
    cl::Buffer qm_pinned_buffer;
    double * const qm_buf_start;
    size_t r_size;
    cl::Buffer * r_pinned_buffer;

    struct beta_gamma
    {
        cl::Buffer beta, gamma;
        double gamma_last;
    };
    beta_gamma build_beta_gamma(std::vector<int64_t> const &sv_indexes, int64_t SampleIndex);
    cl::Buffer build_qsi(cl::Buffer &bsv_indexes, size_t sv_indexes_size, int64_t SampleIndex);
    cl::Buffer build_qxi(const int64_t sample_index);
    cl::Buffer build_qxs(cl::Buffer &sv_indexes, size_t sv_indexes_size, int64_t SampleIndex);
    cl::Buffer* add_zeros_to_r(cl::Buffer &old_r, int64_t r_old_rows, int64_t r_old_cols);
    void rebuild_r(cl::Buffer & beta, cl::Buffer & r, size_t r_size, double GammaSampleIndex);

    cl::Buffer mul_mat_vec(cl::Buffer & matrix, size_t rows, size_t cols, cl::Buffer & vector);
    cl::Buffer mul_mat_vec_invert(cl::Buffer & matrix, size_t rows, size_t cols, cl::Buffer & vector, size_t size_delta = 0);
    void add_vec_vec(cl::Buffer & in_out, size_t size, cl::Buffer & in);
#endif
};


}
}


#endif /* R_MATRIX_IMPL_HPP */

