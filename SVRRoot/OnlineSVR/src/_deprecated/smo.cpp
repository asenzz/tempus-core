#include "deprecated/smo.h"
#include "deprecated/smo_exception.h"
#include "util/PerformanceUtils.hpp"
#include "kernel_factory.hpp"

#include <atomic>

namespace svr::smo {

void cholesky_check(viennacl::matrix<double> &A)
{

    int n = A.size1();

    std::unique_ptr<double> L(new double[n * n]);
//    double *L = (double*)calloc(n * n, sizeof(double));
    if (L.get() == nullptr) {
        throw std::runtime_error("Can not allocate memory for Cholesky matrix!");
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L.get()[i * n + k] * L.get()[j * n + k];

            double v = A(i, j) - s;

            if (i == j) {
                if (v <= 0) { /*free(L);*/ throw std::runtime_error("Kernel Matrix is not positive semi-definite."); }
                L.get()[i * n + j] = sqrt(v);
            } else {
                L.get()[i * n + j] = (1.0 / L.get()[j * n + j]) * v;
            }
        }
//    free(L);
}

void cholesky_check1(viennacl::matrix<double> &A)
{

    int n = A.size1();

    std::vector<double> L(n * n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];

            double v = A(i, j) - s;

            if (i == j) {
                if (v <= 0) { throw std::runtime_error("Kernel Matrix is not positive semi-definite."); }
                L[i * n + j] = sqrt(v);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j]) * v;
            }
        }
}

void cholesky_check_vcl(viennacl::matrix<double> &A)
{

    int n = A.size1();

    viennacl::matrix<double> L(n * n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i + 1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L(i, k) * L(j, k);

            double v = A(i, j) - s;

            if (i == j) {
                if (v <= 0) { throw std::runtime_error("Kernel Matrix is not positive semi-definite."); }
                L(i, j) = sqrt(v);
            } else {
                L(i, j) = (1.0 / L(j, j)) * v;
            }
        }
}

size_t
smo_solver::solve(
        const viennacl::matrix<double> &kernel_matrix,
        const viennacl::vector<double> &linear_term,
        const std::vector<int> &y_,
        const SVRParameters &param,
        std::vector<double> &alpha2,
        double &out_rho)
{
    smo_quadratic_matrix quadratic_matrix(kernel_matrix);
    return solve(quadratic_matrix, svr::datamodel::diag(quadratic_matrix), linear_term, y_, param, alpha2, out_rho);
}

size_t
smo_solver::solve(
        const viennacl::matrix<double> &quadratic_matrix,
        std::vector<double> q_diagonal,
        const viennacl::vector<double> &linear_term,
        const std::vector<int> &y_,
        const SVRParameters &param,
        std::vector<double> &alpha2,
        double &out_rho)
{
    l = quadratic_matrix.size1();

    alpha = alpha2;
    y = y_;
    C = param.get_svr_C();

    // initialize alpha_status
    alpha_status = std::vector<alpha_val>(l, alpha_val::FREE);
    /*cilk_*/for (decltype(alpha_status.size()) i = 0; i < alpha_status.size(); ++i) update_alpha_status(i);

    // initialize active set (for shrinking)
    std::vector<int> active_set(size_t(l), 0);
    /*cilk_*/for (size_t i = 0; i < active_set.size(); ++i) active_set[i] = i;
    active_size = l;

    // initialize gradient
    viennacl::matrix<double> gradient_m = viennacl::matrix<double>(1, l);
    const auto p_linear_term = VCL_DOUBLE_PTR(linear_term);
    /*cilk_*/for (size_t k = 0; k < l; ++k) gradient_m(0, k) = p_linear_term[k];

    //    gradient = linear_term;
    //    gradient_bar = viennacl::scalar_vector<double>(l, 0);

    std::atomic_flag alock = ATOMIC_FLAG_INIT;
    const viennacl::range qmatrix_row_range(0, l);
    /*cilk_*/for (size_t i = 0; i < l; ++i) {
        if (alpha_status[i] != alpha_val::LOWER_BOUND) {
            //get i-th row from KM
            const viennacl::matrix_range<viennacl::matrix<double>> q_row(
                    quadratic_matrix, viennacl::range(i, i + 1), qmatrix_row_range);
            const auto q_row_alpha = alpha[i] * q_row;
            ATOMIC_FLAG_LOCK(alock);
            gradient_m += q_row_alpha;
            ATOMIC_FLAG_UNLOCK(alock);
        }
    }

    // optimization step

    //Copied from svm.cpp:567
    // const auto max_iter = std::max(max_iter_, l > INT_MAX / 100 ? INT_MAX : 100 * l);
    size_t iter = 0;
    while (iter < max_iter_) {
        int i = 0;
        int j = 0;
        if (select_working_set(i, j, q_diagonal, quadratic_matrix, gradient_m, param.get_svr_epsilon()) != 0) {
            // reset active set size and check
            active_size = l;
            //if (select_working_set(i, j, q_diagonal, quadratic_matrix, gradient_m, param.get_svr_epsilon()) != 0)
            break;
            //            else
            //                counter = 1;	// do shrinking next iteration
        }
        if (-1 == i || -1 == j)
            throw svr::smo::smo_max_iter_error(
                    "SMO select_working_set returned a -1 as an index, stopping SMO.");

        ++iter;
        if (iter % 15000 == 0)
            LOG4_INFO("SMO working long ... ¯\\_(ツ)_/¯ iterations : " << iter);

        // update alpha[i] and alpha[j], handle bounds carefully
        viennacl::matrix_range<viennacl::matrix<double>> q_i_row(
                quadratic_matrix, viennacl::range(i, i + 1), qmatrix_row_range);
        viennacl::matrix_range<viennacl::matrix<double>> q_j_row(
                quadratic_matrix, viennacl::range(j, j + 1), qmatrix_row_range);
        //        viennacl::vector<double> q_i_row = viennacl::row<double>(Q, i);
        //        viennacl::vector<double> q_j_row = viennacl::row<double>(Q, j);

        double C_i = C, C_j = C;
        const auto old_alpha_i = alpha[i];
        const auto old_alpha_j = alpha[j];

        if (y[i] != y[j]) {
            auto quad_coef = q_diagonal[i] + q_diagonal[j] + 2. * quadratic_matrix(i, j);
            // Changed from if(quad_coef <= 0)
            if (quad_coef <= TAU) quad_coef = TAU;

            const auto delta = (-gradient_m(0, i) - gradient_m(0, j)) / quad_coef;
            const auto diff = alpha[i] - alpha[j];

            alpha[i] += delta;
            alpha[j] += delta;
            if (diff > 0) {
                if (alpha[j] < 0) {
                    alpha[j] = 0;
                    alpha[i] = diff;
                }
            } else {
                if (alpha[i] < 0) {
                    alpha[i] = 0;
                    alpha[j] = -diff;
                }
            }

            if (diff > C_i - C_j) //diff > 0
            {
                if (alpha[i] > C_i) {
                    alpha[i] = C_i;
                    alpha[j] = C_i - diff;
                }
            } else {
                if (alpha[j] > C_j) {
                    alpha[j] = C_j;
                    alpha[i] = C_j + diff;
                }
            }

        } else {
            auto quad_coef = q_diagonal[i] + q_diagonal[j] - 2. * quadratic_matrix(i, j);
            if (quad_coef <= TAU) quad_coef = TAU;
            const auto delta = (gradient_m(0, i) - gradient_m(0, j)) / quad_coef;
            const auto sum = alpha[i] + alpha[j];

            alpha[i] -= delta;
            alpha[j] += delta;

            if (sum > C_i) {
                if (alpha[i] > C_i) {
                    alpha[i] = C_i;
                    alpha[j] = sum - C_i;
                }
            } else {
                if (alpha[j] < 0) {
                    alpha[j] = 0.;
                    alpha[i] = sum;
                }
            }
            if (sum > C_j) {
                if (alpha[j] > C_j) {
                    alpha[j] = C_j;
                    alpha[i] = sum - C_j;
                }
            } else {
                if (alpha[i] < 0) {
                    alpha[i] = 0.;
                    alpha[j] = sum;
                }
            }
        }

        // update gradient
        const auto delta_alpha_i = alpha[i] - old_alpha_i;
        const auto delta_alpha_j = alpha[j] - old_alpha_j;
        gradient_m += q_i_row * delta_alpha_i + q_j_row * delta_alpha_j;

        // update alpha_status and gradient_bar
        update_alpha_status(i);
        update_alpha_status(j);
    }

    if (l < 1) THROW_EX_FS(smo_zero_active_error,"Active set is zero!");

    if (iter >= max_iter_) {
        if (active_size < l) active_size = l;
        THROW_EX_FS(smo_max_iter_error, "Reached max number of iterations.");
    }
    // calculate rho
    out_rho = calculate_rho(gradient_m);

    // put back the solution
    /*cilk_*/for (decltype(l) i = 0; i < l; i++) alpha2[active_set[i]] = alpha[i];

    return iter;
}

inline void smo_solver::update_alpha_status(const int ix)
{
    if (alpha[ix] >= C) alpha_status[ix] = alpha_val::UPPER_BOUND;
    else if (alpha[ix] <= 0) alpha_status[ix] = alpha_val::LOWER_BOUND;
    else alpha_status[ix] = alpha_val::FREE;
}

/* TODO Move this to a custom ViennaCL kernel */
// return 1 if already optimal, return 0 otherwise

int smo_solver::select_working_set(
        int &out_i,
        int &out_j,
        std::vector<double> &q_diagonal,
        viennacl::matrix<double> const &Q,
        viennacl::matrix<double> const &gradient_m,
        const double epsilon)
{
    // return i,j such that
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    double const *const iter_gradient_begin = reinterpret_cast<double *> (gradient_m.handle().ram_handle().get());
    double const *const iter_gradient_end = iter_gradient_begin + gradient_m.size2();

    auto iter_y = y.begin();
    auto iter_alpha_status = alpha_status.begin();
    size_t t = 0;

    for (auto iter_gradient = iter_gradient_begin; iter_gradient != iter_gradient_end; ++iter_gradient) {
        double grad_cur_value = *iter_gradient;
        if (*iter_y == 1) {
            if (*iter_alpha_status != alpha_val::UPPER_BOUND)
                if (-grad_cur_value >= Gmax) {
                    Gmax = -grad_cur_value;
                    Gmax_idx = t;
                }
        } else {
            if (*iter_alpha_status != alpha_val::LOWER_BOUND)
                if (grad_cur_value >= Gmax) {
                    Gmax = grad_cur_value;
                    Gmax_idx = t;
                }
        }
        t++;
        ++iter_y;
        ++iter_alpha_status;
    }
    int i = Gmax_idx;
    int yi = y[0];
    double QDi = q_diagonal[0];
    viennacl::vector<double> Qi_row;
    if (i != -1) {
        Qi_row = viennacl::row(Q, i);
        QDi = q_diagonal[i];
        yi = y[i];
    }
    t = 0;
    auto iter_y2 = y.begin();
    auto iter_alpha_status2 = alpha_status.begin();
    auto iter_QD = q_diagonal.begin();
    auto iter_Qi_row = Qi_row.begin();
    for (auto iter_gradient = iter_gradient_begin; iter_gradient != iter_gradient_end; ++iter_gradient) {
        __builtin_prefetch(&*iter_y2);
        const auto grad_cur_value = *iter_gradient;
        if (*iter_y2 == +1) {
            if (*iter_alpha_status2 != alpha_val::LOWER_BOUND) {
                const auto grad_diff = Gmax + grad_cur_value;
                if (grad_cur_value >= Gmax2)
                    Gmax2 = grad_cur_value;

                if (grad_diff > 0.0) {
                    double obj_diff;
                    const auto quad_coef = QDi + *iter_QD - 2.0 * yi * (*iter_Qi_row);

                    if (quad_coef > 0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU;

                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = t;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        } else {
            if (*iter_alpha_status2 != alpha_val::UPPER_BOUND) {
                const auto grad_diff = Gmax - grad_cur_value;
                if (-grad_cur_value >= Gmax2)
                    Gmax2 = -grad_cur_value;

                if (grad_diff > 0.0) {
                    double obj_diff;
                    double quad_coef = QDi + *iter_QD + 2.0 * yi * (*iter_Qi_row);

                    if (quad_coef > 0.0)
                        obj_diff = -(grad_diff * grad_diff) / quad_coef;
                    else
                        obj_diff = -(grad_diff * grad_diff) / TAU;

                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = t;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }

        ++t;
        ++iter_y2;
        ++iter_alpha_status2;
        ++iter_QD;
        ++iter_Qi_row;
    }

    if (Gmax + Gmax2 < epsilon / 100.) return 1;
    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return 0;
}

double smo_solver::calculate_rho(viennacl::matrix<double> const &gradient_m)
{
    double r;
    /*cilk::reducer<cilk::op_add<int>>*/ int red_nr_free = 0;
    /*cilk::reducer<cilk::op_add<double>>*/ double red_sum_free = 0;
    /*cilk::reducer<cilk::op_min<double>>*/ double red_min_ub = 0;
    /*cilk::reducer<cilk::op_max<double>>*/ double red_max_lb = 0;
    viennacl::vector<double> gradient(gradient_m.size2());
    auto p_gradient = VCL_DOUBLE_PTR(gradient);
    /*cilk_*/for (decltype(gradient_m.size2()) k = 0; k < gradient_m.size2(); ++k) p_gradient[k] = gradient_m(0, k);

    viennacl::vector<double> y_(y.size());
    auto p_y_ = VCL_DOUBLE_PTR(y_);
    /*cilk_*/for (decltype(y.size()) i = 0; i < y.size(); ++i) p_y_[i] = y[i];

    viennacl::vector<double> y_gradient = viennacl::linalg::element_prod(y_, gradient);
    const auto p_y_gradient = VCL_DOUBLE_PTR(y_gradient);
    /*cilk_*/for (size_t i = 0; i < active_size; ++i) {  // TODO Parallelize
        if (alpha_status[i] == alpha_val::UPPER_BOUND) {
            if (y[i] == -1) red_min_ub = 0;//cilk::min_of(*red_min_ub, p_y_gradient[i]);
            else red_max_lb = 0;//cilk::max_of(red_max_lb, p_y_gradient[i]);
        } else if (alpha_status[i] == alpha_val::LOWER_BOUND) {
            if (y[i] == +1) red_min_ub = 0;// cilk::min_of(*red_min_ub,p_y_gradient[i]);
            else red_max_lb = 0; // cilk::max_of(red_max_lb, p_y_gradient[i]);
        } else {
            red_nr_free += 1;
            red_sum_free += p_y_gradient[i];
        }
    }
    const auto nr_free = red_nr_free;
    const auto sum_free = red_sum_free;
    const auto min_ub = red_min_ub;
    const auto max_lb = red_max_lb;

    if (nr_free > 0) r = sum_free / double(nr_free);
    else r = (min_ub + max_lb) / 2.;
    return r;
}

void kernel_matrix::init_diag()
{
    _diag.resize(size1());
    for (size_t i = 0; i < _diag.size(); ++i) _diag.host_data()[i] = 1.;
}

void
kernel_matrix::get_rows(
        const SyncArray<int> &idx,
        SyncArray<double> &kernel_rows) const
{//compute multiple rows of kernel matrix according to idx
    if (kernel_rows.size() < idx.size() * size2()) LOG4_THROW("Kernel rows memory is too small.");

    const auto idx_data = idx.host_data();
    /*cilk_*/for (size_t i = 0; i < size2(); ++i) {
        /*cilk_*/for (size_t j = 0; j < idx.size(); ++j) {
            kernel_rows.host_data()[j * size2() + i] = this->operator()(idx_data[j], i);
        }
    }
}


smo_quadratic_matrix::smo_quadratic_matrix(const viennacl::matrix<double> &kernel_matrix)
{
    //svr::common::memory_manager::instance().wait();

    viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);
    resize(kernel_matrix.size1() * 2, kernel_matrix.size2() * 2, true);
    form_quadratic_matrix(kernel_matrix, *this);
}

smo_quadratic_matrix::smo_quadratic_matrix(
        const svr::datamodel::vmatrix<double> &learning_data,
        const SVRParameters &param) : smo_quadratic_matrix(kernel_matrix(learning_data, param))
{}

kernel_matrix::kernel_matrix(
        const arma::mat &learning_data,
        const SVRParameters &param)
{
    viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);

    if (learning_data.empty()) LOG4_THROW("Learning data empty.");

#if defined(VIENNACL_WITH_OPENCL) && !defined(AVITOHOL)
    //const auto limit_num_rows = (size_t) std::sqrt( // TODO Fix calculation, gives too little GPU memory!
    //       svr::common::gpu_handler::get().get_max_gpu_data_chunk_size() / sizeof(double)) * 0.8;
    if (true) { //(size_t) learning_data.get_length_rows() < limit_num_rows) {
        ocl_prepare(learning_data, param);
    } else {
        LOG4_THROW("Learning data doesn't fit in GPU memory!");
    }
#else //VIENNACL_WITH_OPENCL
    cpu_prepare(learning_data, param);
#endif //VIENNACL_WITH_OPENCL
    init_diag();
}

kernel_matrix::kernel_matrix(
        const svr::datamodel::vmatrix<double> &learning_data,
        const SVRParameters &param)
{
    viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);

    if (learning_data.empty()) LOG4_THROW("Learning data empty.");

#if defined(VIENNACL_WITH_OPENCL) && !defined(AVITOHOL)
    //const auto limit_num_rows = (size_t) std::sqrt( // TODO Fix calculation, gives too little GPU memory!
    //       svr::common::gpu_handler::get().get_max_gpu_data_chunk_size() / sizeof(double)) * 0.8;
    if (true) { //(size_t) learning_data.get_length_rows() < limit_num_rows) {
        ocl_prepare(learning_data, param);
    } else {
        LOG4_THROW("Learning data doesn't fit in GPU memory!");
    }
#else //VIENNACL_WITH_OPENCL
    cpu_prepare(learning_data, param);
#endif //VIENNACL_WITH_OPENCL

    init_diag();
}

void
smo_quadratic_matrix::form_quadratic_matrix(
        const viennacl::matrix<double> &kernel_matrix,
        viennacl::matrix<double> &quadratic_matrix)
{
    //form quadratic matrix
    viennacl::range q_m_row_top_range(0, quadratic_matrix.size1() / 2);
    viennacl::range q_m_row_bottom_range(quadratic_matrix.size1() / 2, quadratic_matrix.size1());
    viennacl::range q_m_col_left_range(0, quadratic_matrix.size2() / 2);
    viennacl::range q_m_cols_right_range(quadratic_matrix.size2() / 2, quadratic_matrix.size2());

    viennacl::matrix_range<viennacl::matrix<double>> quadratic_matrix_top_left(
            quadratic_matrix, q_m_row_top_range, q_m_col_left_range);
    viennacl::matrix_range<viennacl::matrix<double>> quadratic_matrix_top_right(
            quadratic_matrix, q_m_row_top_range, q_m_cols_right_range);

    viennacl::matrix_range<viennacl::matrix<double>> quadratic_matrix_bottom_left(
            quadratic_matrix, q_m_row_bottom_range, q_m_col_left_range);
    viennacl::matrix_range<viennacl::matrix<double>> quadratic_matrix_bottom_right(
            quadratic_matrix, q_m_row_bottom_range, q_m_cols_right_range);
    quadratic_matrix_top_left = kernel_matrix;
    quadratic_matrix_top_right = kernel_matrix * (-1.0);
    quadratic_matrix_bottom_left = kernel_matrix * (-1.0);
    quadratic_matrix_bottom_right = kernel_matrix;
}

#ifdef VIENNACL_WITH_OPENCL


void kernel_matrix::ocl_prepare(const arma::mat &learning_data, const SVRParameters &param)
{
    auto context = svr::common::gpu_context();
    auto & ctx =  context.ctx();
    viennacl::matrix<double> vcl_x(learning_data.n_rows, learning_data.n_cols, ctx);
    viennacl::matrix<double> kernel_matrix(learning_data.n_rows, learning_data.n_rows, ctx);
    svr::common::arma_to_vcl(learning_data, vcl_x);
    PROFILE_EXEC_TIME((*IKernel<double>::get_kernel(param.get_kernel_type(),param))(ctx, vcl_x, kernel_matrix), "Kernel matrix computation");
    resize(kernel_matrix.size1(), kernel_matrix.size2(), true);
    viennacl::fast_copy(kernel_matrix,(double *) this->handle().ram_handle().get()); // from GPU to CPU

    //viennacl::matrix<double> kernel_matrix_cpu(learning_data.n_rows, learning_data.n_rows,  viennacl::context(viennacl::MAIN_MEMORY));
    //viennacl::fast_copy(p_kernel_matrices, (double *) kernel_matrix_cpu.handle().ram_handle().get()); // from GPU to CPU
    //resize(p_kernel_matrices.size1() * 2, p_kernel_matrices.size2() * 2, true);
    LOG4_DEBUG("Quadratic matrix GPU mem usage " << this->internal_size() * sizeof(double) / 1048576 << " MB");
    //PROFILE_EXEC_TIME(form_quadratic_matrix(kernel_matrix_cpu, *this), "form_quadratic_matrix");
}


void kernel_matrix::ocl_prepare(const datamodel::vmatrix<double> &learning_data, const SVRParameters &param)
{
    auto context = svr::common::gpu_context();
    auto & ctx =  context.ctx();
    viennacl::matrix<double> vcl_x(learning_data.get_length_rows(), learning_data.get_length_cols(), ctx);
    viennacl::matrix<double> kernel_matrix(learning_data.get_length_rows(), learning_data.get_length_rows(), ctx);
    learning_data.copy_to(vcl_x);
    PROFILE_EXEC_TIME((*IKernel<double>::get_kernel(param.get_kernel_type(),param))(ctx, vcl_x, kernel_matrix), "Kernel matrix computation");
    resize(kernel_matrix.size1(), kernel_matrix.size2(), true);
    viennacl::fast_copy(kernel_matrix,(double *) this->handle().ram_handle().get()); // from GPU to CPU
    //viennacl::matrix<double> kernel_matrix_cpu(learning_data.get_length_rows(), learning_data.get_length_rows(),  viennacl::context(viennacl::MAIN_MEMORY));
    //viennacl::fast_copy(p_kernel_matrices,(double *) kernel_matrix_cpu.handle().ram_handle().get()); // from GPU to CPU
    //resize(p_kernel_matrices.size1() * 2, p_kernel_matrices.size2() * 2, true);
    LOG4_DEBUG("Quadratic matrix GPU mem usage " << this->internal_size() * sizeof(double) / 1048576 << " MB");
    //PROFILE_EXEC_TIME(form_quadratic_matrix(kernel_matrix_cpu, *this), "form_quadratic_matrix");
}


/*
void smo_quadratic_matrix::init_kernel_matrix(const SVRParameters &svr_parameters, const arma::mat &x_train, arma::mat &p_kernel_matrices)
{
    auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
    if (svr_parameters.get_kernel_type() == kernel_type_e::RBF_EXPONENTIAL) {
        (*kernel)(x_train, p_kernel_matrices);
    } else {
        svr::common::gpu_context context;
        auto &ctx = context.ctx();
        viennacl::matrix<double> dense_kernel_matrix_gpu((int) x_train.n_rows, (int) x_train.n_rows, ctx);
        viennacl::matrix<double> support_vectors_data_vcl_gpu(x_train.n_rows, x_train.n_cols, ctx);
        common::arma_to_vcl(x_train, support_vectors_data_vcl_gpu);
        PROFILE_EXEC_TIME((*kernel)(ctx, support_vectors_data_vcl_gpu, dense_kernel_matrix_gpu),
                          "Kernel matrix computation " << int(context.id()));
        common::vcl_to_arma(dense_kernel_matrix_gpu, p_kernel_matrices);
    }
}
*/

#endif //VIENNACL_WITH_OPENCL


} //svr
