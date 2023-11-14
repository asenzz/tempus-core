#ifndef _SMO_H
#define _SMO_H

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/prod.hpp"

#include "model/_deprecated/vmatrix.tcc"
#include "model/_deprecated/vektor.tcc"

#include "util/MemoryManager.hpp"

#include "model/SVRParameters.hpp"
#include "common/gpu_handler.hpp"
/* must be after gpu handler because of opencl functions used in some kernels */
#include "onlinesvr.hpp"
#include "_deprecated/syncarray.hpp"

namespace svr {
namespace smo {

#define TAU 1e-12

class smo_solver
{
    const size_t max_iter_ = common::__max_iter;
    const double smo_epsilon_divisor_ = common::__smo_epsilon_divisor;
public:
    smo_solver(const size_t max_iter = common::__max_iter, const double smo_epsilon_divisor = common::__smo_epsilon_divisor) :
        max_iter_(max_iter), smo_epsilon_divisor_(smo_epsilon_divisor) {}
    virtual ~smo_solver() {}

    size_t
    solve(const viennacl::matrix<double> &kernel_matrix,
          const viennacl::vector<double> &linear_term,
          const std::vector<int> &y_,
          const datamodel::SVRParameters &param,
          std::vector<double> &alpha2,
          double &out_rho);

    size_t
    solve(
            const viennacl::matrix<double> &quadratic_matrix,
            std::vector<double> q_diagonal,
            const viennacl::vector<double> &linear_term,
            const std::vector<int> &y_,
            const datamodel::SVRParameters &param,
            std::vector<double> &alpha2,
            double &out_rho);

private:
    size_t active_size = 0;
    std::vector<int> y;
    enum class alpha_val{ LOWER_BOUND, UPPER_BOUND, FREE };
    std::vector<alpha_val> alpha_status;
    std::vector<double> alpha;
    double C = 0.;
    size_t l = 0;

    inline void update_alpha_status(const int ix);
    int select_working_set(int &i, int &j, std::vector<double> & q_diagonal, viennacl::matrix<double> const & Q, viennacl::matrix<double> const & gradient_m, const double epsilon);
    double calculate_rho(viennacl::matrix<double> const & gradient_m);

    bool is_free(int i) { return alpha_status[i] == alpha_val::FREE; }
};

class kernel_matrix : public viennacl::matrix<double> {
    SyncArray<double> _diag;
    void init_diag();

public:
    size_t size() const { return size1() * size2(); }
    const SyncArray<double> &diag() const
    {
        if (_diag.size() < 1) LOG4_THROW("Diagonal not initialized!");
        return _diag;
    }
    void get_rows(const SyncArray<int> &idx, SyncArray<double> &kernel_rows) const;

    kernel_matrix(const arma::mat &learning_data, const datamodel::SVRParameters &param);
    kernel_matrix(const datamodel::vmatrix<double> &learning_data, const datamodel::SVRParameters &param);

    void ocl_prepare(const arma::mat &learning_data, const datamodel::SVRParameters & param);
    void ocl_prepare(const datamodel::vmatrix<double> &learning_data, const datamodel::SVRParameters &param);
};

class smo_quadratic_matrix : public viennacl::matrix<double> {
    static void form_quadratic_matrix(const viennacl::matrix<double> &kernel_matrix, viennacl::matrix<double> &quadratic_matrix);
public:
    smo_quadratic_matrix(const datamodel::vmatrix<double> &learning_data, const datamodel::SVRParameters &param);
    explicit smo_quadratic_matrix(const viennacl::matrix<double> &kernel_matrix);
    virtual ~smo_quadratic_matrix() {}
};


}
}


#endif /* _SMO_H */
