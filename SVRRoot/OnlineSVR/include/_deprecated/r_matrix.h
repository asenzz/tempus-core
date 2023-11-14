#pragma once

#include "model/_deprecated/vmatrix.tcc"
#include "model/_deprecated/vektor.tcc"
#include <vector>
#include <fstream>
#include "common/gpu_handler.hpp"

namespace svr {
namespace batch {

class r_matrix_cpu;
class r_matrix_gpu;

class r_matrix
{
public:
    explicit r_matrix(const svr::datamodel::vmatrix<double> * p_kernel_matrix, const std::vector<int64_t> &sv_indices, bool update_r_matrix_enabled = true);
    ~r_matrix();

    void update_r_matrix(const std::vector<int64_t> &sv_indexes, const int64_t sample_index);
static void compute_r_matrix(const svr::datamodel::vmatrix<double> * p_kernel_matrix , const std::vector<int64_t> &sv_indices , viennacl::matrix<double> & result_matrix);
    svr::datamodel::vmatrix<double> *get_r();
private:
    r_matrix_cpu & cpu;
    r_matrix_gpu * gpu;
};


} //batch
} //svr

