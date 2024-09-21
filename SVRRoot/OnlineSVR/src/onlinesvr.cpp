#include <algorithm>
#include <armadillo>
#include </opt/intel/oneapi/mkl/latest/include/mkl_lapacke.h>
#include <cmath>
#include <deque>
#include <execution>
#include <magma_v2.h>
#include <memory>
#include <sys/mman.h>
#include <ipp.h>
#include <osqp/osqp.h>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "appcontext.hpp"
#include "SVRParametersService.hpp"
#include "onlinesvr.hpp"
#include "mat_solve_gpu.hpp"
#include "cuqrsolve.cuh"
#include "model/Entity.hpp"
#include "model/SVRParameters.hpp"
#include "util/math_utils.hpp"
#include "cuda_path.hpp"
#include "calc_cache.hpp"
#include "kernel_factory.hpp"
#include "pprune.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif

class onlinesvr_lib_init
{
public:
    onlinesvr_lib_init()
    {
        mlockall(MCL_CURRENT | MCL_FUTURE);
        // ip_errchk(ippInit());
        ma_errchk(magma_init());
    }

    ~onlinesvr_lib_init()
    {
        munlockall();
        ma_errchk(magma_finalize());
    }
};

const auto l = []() {
    return onlinesvr_lib_init();
}();

namespace svr {
namespace datamodel {

OnlineMIMOSVR::OnlineMIMOSVR() : Entity(0), multiout(PROPS.get_multiout())
{
    LOG4_WARN("Created OnlineMIMOSVR object with default constructor and default multistep_len " << multiout);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}


void OnlineMIMOSVR::parse_params()
{
    if (param_set.empty()) LOG4_THROW("At least one parameter set should be supplied to construct an SVR model.");

    const auto p = *param_set.cbegin();
    level = p->get_decon_level();
    step = p->get_step();
    gradient = p->get_grad_level();
    if (p_dataset) max_chunk_size = p_dataset->get_max_chunk_size();
    column_name = p->get_input_queue_column_name();

    LOG4_END();
}

OnlineMIMOSVR::OnlineMIMOSVR(
        const bigint id,
        const bigint model_id,
        const t_param_set &param_set,
        const Dataset_ptr &p_dataset) :
        Entity(id),
        model_id(model_id),
        p_dataset(p_dataset),
        param_set(param_set)
{
    if (model_id) scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model_id);
    multiout = PROPS.get_multiout();
    parse_params();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

OnlineMIMOSVR::OnlineMIMOSVR(
        const bigint id,
        const bigint model_id,
        const t_param_set &param_set,
        const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const bpt::ptime &last_value_time,
        const matrices_ptr &kernel_matrices,
        const Dataset_ptr &p_dataset) :
        Entity(id), model_id(model_id), p_dataset(p_dataset), param_set(param_set)
{
    if (model_id) scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model_id);
    multiout = PROPS.get_multiout();
    parse_params();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    PROFILE_EXEC_TIME(
            batch_train(p_xtrain, p_ytrain, p_ylastknown, last_value_time, kernel_matrices),
            "Batch SVM train on " << arma::size(*p_ytrain) << " labels and " << arma::size(*p_xtrain) << " features, parameters " << *front(param_set));
}

void OnlineMIMOSVR::set_dataset(const Dataset_ptr &p_dataset_)
{
    p_dataset = p_dataset_;
}

dtype(OnlineMIMOSVR::p_dataset) OnlineMIMOSVR::get_dataset() const
{
    return p_dataset;
}

dtype(OnlineMIMOSVR::p_dataset) &OnlineMIMOSVR::get_dataset()
{
    return p_dataset;
}

dtype(OnlineMIMOSVR::samples_trained) OnlineMIMOSVR::get_samples_trained_number() const noexcept
{
    return samples_trained;
}

void OnlineMIMOSVR::clear_kernel_matrix()
{
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_kernel_matrices->size()))
    for (auto &k: *p_kernel_matrices) k.clear();
}

dtype(OnlineMIMOSVR::gradient) OnlineMIMOSVR::get_gradient_level() const noexcept
{
    return gradient;
}

dtype(OnlineMIMOSVR::level) OnlineMIMOSVR::get_decon_level() const noexcept
{
    return level;
}

dtype(OnlineMIMOSVR::step) OnlineMIMOSVR::get_step() const noexcept
{
    return step;
}

dtype(OnlineMIMOSVR::multiout) OnlineMIMOSVR::get_multiout() const noexcept
{
    return multiout;
}

arma::uvec OnlineMIMOSVR::get_active_ixs() const
{
    arma::uvec active_ixs;
    for (const auto &ix: ixs) active_ixs.insert_rows(active_ixs.n_rows, ix);
    return arma::unique(active_ixs);
}

dtype(OnlineMIMOSVR::param_set) OnlineMIMOSVR::get_param_set() const noexcept
{
    return param_set;
}

dtype(OnlineMIMOSVR::param_set) &OnlineMIMOSVR::get_param_set() noexcept
{
    return param_set;
}

void OnlineMIMOSVR::set_param_set(const dtype(OnlineMIMOSVR::param_set) &param_set_)
{
    for (const auto &new_p: param_set_)
        set_params(new_p, new_p->get_chunk_index());
}

void OnlineMIMOSVR::set_params(const SVRParameters_ptr &p_svr_parameters_, const unsigned chunk_ix)
{
    auto it_target_params = business::SVRParametersService::find(param_set, chunk_ix, gradient);
    if (it_target_params == param_set.cend())
        param_set.emplace(p_svr_parameters_);
    else
        **it_target_params = *p_svr_parameters_;
}

void OnlineMIMOSVR::set_params(const SVRParameters &param, const unsigned chunk_ix)
{
    auto p_target_params = get_params_ptr(chunk_ix);
    if (p_target_params)
        *p_target_params = param;
    else
        param_set.emplace(ptr<SVRParameters>(param));
}

SVRParameters &OnlineMIMOSVR::get_params(const unsigned chunk_ix) const
{
    return **business::SVRParametersService::find(param_set, chunk_ix, gradient);
}

SVRParameters_ptr OnlineMIMOSVR::get_params_ptr(const unsigned chunk_ix) const
{
    return business::SVRParametersService::find_ptr(param_set, chunk_ix, gradient);
}

business::calc_cache &OnlineMIMOSVR::ccache()
{
    return p_dataset->get_calc_cache();
}

arma::mat &OnlineMIMOSVR::get_features()
{
    return *p_features;
}

arma::mat &OnlineMIMOSVR::get_labels()
{
    return *p_labels;
}


bool OnlineMIMOSVR::needs_tuning(const t_param_set &param_set)
{
    if (PROPS.get_tune_parameters()) return true;

    bool res = false;
    for (const auto &p: param_set) res |= p->get_svr_kernel_param() == 0;
    return res;
}


const dq_scaling_factor_container_t &OnlineMIMOSVR::get_scaling_factors() const
{
    return scaling_factors;
}

void OnlineMIMOSVR::set_scaling_factor(const DQScalingFactor_ptr &p_sf)
{
    business::DQScalingFactorService::add(scaling_factors, p_sf);
}

void OnlineMIMOSVR::set_scaling_factors(const dq_scaling_factor_container_t &new_scaling_factors)
{
    business::DQScalingFactorService::add(scaling_factors, new_scaling_factors);
}

bool OnlineMIMOSVR::needs_tuning() const
{
    return needs_tuning(param_set) || ixs.empty();
}

void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

// Unstable avoid! OpenCL
arma::mat OnlineMIMOSVR::do_ocl_solve(CPTR(double) host_a, double *host_b, const int m, const unsigned nrhs)
{
    LOG4_THROW("Deprecated!");
    return {};
#if 0
    if (m * (m + nrhs) * sizeof(double) > common::gpu_handler_hid::get().get_max_gpu_data_chunk_size())
        THROW_EX_FS(std::runtime_error,
                "Not enough memory on GPU, required " << m * nrhs * sizeof(double) << " bytes, available " <<
                     common::gpu_handler_hid::get().get_max_gpu_data_chunk_size() << " bytes.");

    auto context = common::gpu_context();
    viennacl::ocl::context &ctx = context.ctx();
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_command_queue queue = cq_command_queue();

    cl_int err;
    cl_mem device_a = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, m * m * sizeof(double), nullptr, &err);
    CL_CHECK(err);
    cl_mem device_b = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, m * nrhs * sizeof(double), nullptr, &err);
    CL_CHECK(err);

    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m * m * sizeof(double), host_a, 0, nullptr, nullptr);
    CL_CHECK(err);
    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, m * nrhs * sizeof(double), host_b, 0, nullptr, nullptr);
    CL_CHECK(err);

    clFinish(queue);
    int info;
    dposv_gpu(MagmaUpper, m, nrhs, device_a, m, device_b, m, queue, &info);
    clFinish(queue);
    if (info != 0) {
        LOG4_ERROR("Error info in dposv, lets try dgesv " << int(info));
        std::vector<int> ipiv(m);
        //copy once again, otherwise device_a may have been changed by dposv
        CL_CHECK(err);
        clFinish(queue);
        CL_CHECK(err);
        dgesv_gpu(m, nrhs, device_a, m, ipiv.data(), device_b, m, ctx, &info);
        if (info != 0) {
            LOG4_ERROR("Error in dgesv, bailing out " << int(info));
            throw std::runtime_error("Error in dgesv.");
        }
    }
    arma::mat result(m, nrhs);
    clEnqueueReadBuffer(queue, device_b, CL_TRUE, 0, m * nrhs * sizeof(double), result.memptr(), 0, nullptr, nullptr);
    clFinish(queue);
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    return result;
#endif
}

#if 0 // Bandicoot
coot::mat OnlineMIMOSVR::solve_irwls(const coot::mat &K_epsco, const coot::mat &K, const coot::mat &rhs, const size_t iters, const size_t gpu_id)
{
    magma_queue_t magma_queue = nullptr;
    magma_init(); // initialize Magma
    magma_queue_create(gpu_id, &magma_queue);
    if (!magma_queue) LOG4_THROW("Failed creating MAGMA queue.");
    const auto m = K.n_rows;
    const auto b_n = rhs.n_cols;
    coot::mat solved = rhs;
    coot::get_rt().synchronise();
    magma_int_t err, info;
    if ((err = magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, K.get_dev_mem(false).cuda_mem_ptr, m, solved.get_dev_mem(false).cuda_mem_ptr, m, &info)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_dgesv_rbt with error code " << err << ", info " << info);

    auto best_sae = std::numeric_limits<double>::infinity();
    coot::mat best_solution = solved;
    size_t best_iter = 0;
UNROLL()
    for (size_t i = 1; i < iters; ++i) {
        const coot::mat error_mat = coot::abs(K * solved - rhs);
        const double this_sae = coot::accu(error_mat);
        if (this_sae < best_sae) {
            LOG4_TRACE("IRWLS iteration " << i << ", SAE " << this_sae << ", kernel dimensions " << coot::size(K) << ", best SAE " << best_sae);
            best_sae = this_sae;
            best_solution = solved;
            best_iter = i;
        }
        const coot::mat mult = coot::sqrt(error_mat + common::C_itersolve_delta / (double(i) * common::C_itersolve_range / double(iters)));
        const coot::mat left = (mult * coot::ones<coot::mat>(mult.n_cols, K.n_cols)) % K_epsco;
        solved = rhs % mult;
        coot::coot_synchronise();
        if ((err = magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, left.get_dev_mem(false).cuda_mem_ptr, m, solved.get_dev_mem(false).cuda_mem_ptr, m, &info)) < MAGMA_SUCCESS)
            LOG4_THROW("Failed calling magma_dgesv_rbt with error code " << err << ", info " << info);
        //magma_queue_sync(queue);
    }

    magma_queue_destroy(magma_queue);

    if ((err = magma_finalize()) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_finalize with error code " << err);
    LOG4_DEBUG("IRWLS best iteration " << best_iter << ", MAE " << best_sae / double(solved.n_elem) << ", kernel dimensions " << coot::size(K) <<
                                       ", delta " << common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << coot::size(solved));
    return best_solution;
}
#endif


void OnlineMIMOSVR::solve_opt(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const unsigned iters)
{
#if 1
    const auto loss_function = [&K, &rhs](CPTR(double) x, double *const f) {
#if 0
    const auto rhs_size = rhs.n_elem * sizeof(double);
        const auto loss_function = [rhs_size, &K, &rhs](CPTR(double) x, double *const f) {
        auto tmp = (double *) malloc(rhs_size);
        memcpy(tmp, rhs.memptr(), rhs_size);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K.n_rows, rhs.n_cols, K.n_cols, 0, K.memptr(), K.n_rows, x, rhs.n_cols, -1., tmp, K.n_rows);
        *f = cblas_dasum(rhs.n_elem, tmp, 1);
        free(tmp);
#else
        *f = arma::mean(arma::abs(arma::vectorise(K * arma::mat((double *) x, rhs.n_rows, rhs.n_cols, false, true) - rhs)));
#endif
    };
    arma::mat bounds(rhs.n_elem, 2);
    const auto lim = 10. * common::meanabs(rhs) / common::meanabs(K) / K.n_cols;
    bounds.col(0).fill(-lim);
    bounds.col(1).fill(lim);
    const optimizer::t_pprune_res res = optimizer::pprune(0, 100, bounds, loss_function, rhs.n_elem, 0, 0, solved);
    if (arma::size(rhs) != arma::size(solved)) solved.set_size(arma::size(rhs));
    std::copy(C_default_exec_policy, res.best_parameters.cbegin(), res.best_parameters.cend(), solved.begin());

#else // Solve using OSQP

    arma::sp_mat sp_K_epsco(K_epsco);
    const auto P_x = (OSQPFloat *) sp_K_epsco.values;
    OSQPInt P_nnz = sp_K_epsco.n_nonzero;
    const auto P_i = (OSQPInt *) sp_K_epsco.row_indices;
    const auto P_p = (OSQPInt *) sp_K_epsco.col_ptrs;
    OSQPFloat q[2] = {1.0, 1.0,};
    OSQPFloat A_x[4] = {1.0, 1.0, 1.0, 1.0,};
    OSQPInt A_nnz = 4;
    OSQPInt A_i[4] = {0, 1, 0, 2,};
    OSQPInt A_p[3] = {0, 2, 4,};
    OSQPFloat l[3] = {1.0, 0.0, 0.0,};
    OSQPFloat u[3] = {1.0, 0.7, 0.7,};
    OSQPInt n = 2;
    OSQPInt m = 3;

    /* Exitflag */
    OSQPInt exitflag = 0;

    /* Solver, settings, matrices */
    const auto P = (OSQPCscMatrix *) malloc(sizeof(OSQPCscMatrix));
    const auto A = (OSQPCscMatrix *) malloc(sizeof(OSQPCscMatrix));

    /* Populate matrices */
    csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
    csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);

    /* Set default settings */
    auto settings = (OSQPSettings *) malloc(sizeof(OSQPSettings));
    if (settings) {
        osqp_set_default_settings(settings);
        settings->alpha = 1.0; /* Change alpha parameter */
    }

    /* Setup solver */
    OSQPSolver *solver;
    exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

    /* Solve problem */
    if (!exitflag) exitflag = osqp_solve(solver);

    /* Cleanup */
    osqp_cleanup(solver);
    if (A) free(A);
    if (P) free(P);
    if (settings) free(settings);

    return (int) exitflag;
#endif
}


void OnlineMIMOSVR::solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const unsigned iters)
{
    double *d_K_epsco, *d_K, *d_labels, *d_solved, *d_rwork, *d_lwork, best_solve_score;
    const size_t K_size = K.n_elem * sizeof(double);
    const size_t rhs_size = rhs.n_elem * sizeof(double);
    unsigned best_iter = 0;
    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
/*    
    cu_errchk(cudaHostRegister((void *) K_epsco.memptr(), K_size, cudaHostRegisterReadOnly));
    cu_errchk(cudaHostRegister((void *) K.memptr(), K_size, cudaHostRegisterReadOnly));
    cu_errchk(cudaHostRegister((void *) rhs.memptr(), rhs_size, cudaHostRegisterReadOnly)); 
    */
    magma_queue_t maqueue;
    magma_queue_create(ctx.phy_id(), &maqueue);
    const cudaStream_t custream = magma_queue_get_cuda_stream(maqueue);
    const cublasHandle_t cublas_H = magma_queue_get_cublas_handle(maqueue);
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, K_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K_epsco, K_epsco.mem, K_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K, K_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K, K.mem, K_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync(&d_labels, rhs_size, custream));
    cu_errchk(cudaMemcpyAsync(d_labels, rhs.mem, rhs_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync(&d_solved, rhs_size, custream));
    cu_errchk(cudaMallocAsync(&d_rwork, rhs_size, custream));
    cu_errchk(cudaMallocAsync(&d_lwork, K_size, custream));
    const auto iters_mul = common::C_itersolve_range / double(iters);
    magma_int_t info;
    solvers::solve_hybrid(
            d_K_epsco, rhs.n_cols, K.n_rows, d_solved, C_rbt_iter, C_rbt_threshold, maqueue, PROPS.get_stabilize_iterations_count(), d_labels, rhs_size,
            d_rwork, custream, cublas_H, d_K, 1, rhs.n_elem, best_solve_score, best_iter, d_solved, K.n_elem, d_lwork, info, iters_mul, K.n_rows);
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K, custream));
    cu_errchk(cudaFreeAsync(d_labels, custream));
    cu_errchk(cudaFreeAsync(d_rwork, custream));
    cu_errchk(cudaFreeAsync(d_lwork, custream));
    if (arma::size(rhs) != arma::size(solved)) solved.set_size(arma::size(rhs));
//    cu_errchk(cudaHostRegister((void *) solved.memptr(), rhs_size, cudaHostRegisterDefault));
    cu_errchk(cudaMemcpyAsync(solved.memptr(), d_solved, rhs_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_solved, custream));
    /*
    cu_errchk(cudaHostUnregister((void *) K_epsco.memptr()));
    cu_errchk(cudaHostUnregister((void *) K.memptr()));
    cu_errchk(cudaHostUnregister((void *) rhs.memptr()));
    cu_errchk(cudaHostUnregister((void *) solved.memptr()));
    */
    cu_errchk(cudaStreamSynchronize(custream));
    magma_queue_destroy(maqueue);
}

// Buggy rewrite and test
std::deque<arma::mat> OnlineMIMOSVR::solve_batched_irwls(
        const std::deque<arma::mat> &K_epsco, const std::deque<arma::mat> &K, const std::deque<arma::mat> &rhs, const size_t iters,
        const magma_queue_t &magma_queue, const size_t gpu_phy_id)
{
    auto solved = rhs;
    const auto batch_size = K.size();
    const size_t m = K.front().n_rows, n = rhs.front().n_cols;

    if (K.size() != K_epsco.size() || K_epsco.size() != rhs.size()) LOG4_THROW("Sizes not correct " << K.size() << ", " << K_epsco.size() << ", " << rhs.size());
    LOG4_TRACE("Batch size " << K.size() << ", m " << m << ", n " << n << ", K front " << arma::size(K.front()) << ", K back " << arma::size(K.back()) << ", rhs front "
                             << arma::size(rhs.front()) <<
                             ", rhs back " << arma::size(rhs.back()) << ", solved front " << arma::size(solved.front()) << ", solved back " << arma::size(solved.back()));
    cu_errchk(cudaSetDevice(gpu_phy_id));
    auto [d_K, d_rhs] = solvers::init_magma_batch_solver(batch_size, m, n);

    solvers::iter_magma_batch_solve(m, n, K_epsco, rhs, solved, magma_queue, d_K, d_rhs, gpu_phy_id);

    std::vector<double> best_sae(batch_size, std::numeric_limits<double>::infinity());
    std::vector<size_t> best_iter(batch_size, 0);
    const auto range_iters = common::C_itersolve_range / double(iters);
    dtype(solved) best_solution(batch_size);
    std::deque<arma::mat> left(batch_size), right(batch_size);
    for (size_t i = 1; i < iters; ++i) {
        const auto delta_i_range_iters = common::C_itersolve_delta / (double(i) * range_iters);
// #pragma omp parallel for num_threads(adj_threads(batch_size)) schedule(static, 1)
UNROLL()
        for (size_t j = 0; j < batch_size; ++j) {
            const arma::mat error_mat = arma::abs(K[j] * solved[j] - rhs[j]);
            const auto this_sae = arma::accu(error_mat);
            if (this_sae < best_sae[j]) {
                LOG4_TRACE("IRWLS " << j << ", iteration " << i << ", SAE " << this_sae << ", kernel dimensions " << arma::size(K[j]) << ", best SAE " << best_sae[j]);
                best_sae[j] = this_sae;
                best_solution[j] = solved[j];
                best_iter[j] = i;
            }
            const arma::mat mult = arma::sqrt(error_mat + delta_i_range_iters);
            left[j] = (mult * arma::ones(mult.n_cols, K[j].n_cols)) % K_epsco[j];
            right[j] = rhs[j] % mult;
        }
        solvers::iter_magma_batch_solve(m, n, left, right, solved, magma_queue, d_K, d_rhs, gpu_phy_id);
    }
    solvers::uninit_magma_batch_solver(d_K, d_rhs);
#pragma omp for simd
    for (size_t j = 0; j < batch_size; ++j)
        LOG4_DEBUG("IRWLS " << j << ", best iteration " << best_iter[j] << ", MAE " << best_sae[j] / double(solved[j].n_elem) << ", kernel dimensions " <<
                            arma::size(K[j]) << ", delta " << common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution "
                            << arma::size(solved[j]));
    if (best_solution.size()) solved = best_solution;
    return solved;
}


arma::mat OnlineMIMOSVR::direct_solve(const arma::mat &a, const arma::mat &b)
{
    LOG4_THROW("Deprecated");
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));
    arma::mat solved = arma::zeros(arma::size(b));
//    PROFILE_EXEC_TIME(solvers::qrsolve(a.n_rows, b.n_cols, a.mem, b.mem, solved.memptr()), "qrsolve");
    return solved;
}

unsigned OnlineMIMOSVR::get_full_train_len(const unsigned n_rows, const unsigned decrement)
{
    return n_rows && n_rows < decrement ? n_rows - C_test_len : decrement;
}

size_t OnlineMIMOSVR::get_num_chunks(const unsigned n_rows, const unsigned chunk_size_)
{
    if (!n_rows || !chunk_size_) LOG4_THROW("Rows count " << n_rows << " or maximum chunk size " << chunk_size_ << " is zero!");
    if (n_rows <= chunk_size_) return 1;
    return cdiv(n_rows, chunk_size_ * C_chunk_offlap) - cdiv(1, C_chunk_offlap) + C_end_chunks;
}

size_t OnlineMIMOSVR::get_num_chunks()
{
    if (is_manifold())
        return 0;
    else if (!projection)
        return 1;
    else
        return get_num_chunks(
            get_full_train_len(p_labels ? p_labels->n_rows : 0, (**param_set.cbegin()).get_svr_decremental_distance()),
            max_chunk_size);
}

std::deque<arma::uvec>
generate_chunk_indexes(const unsigned n_rows_dataset, const unsigned decrement, const unsigned max_chunk_size)
{
    // Make sure we train on the latest data
    const auto n_rows_train = OnlineMIMOSVR::get_full_train_len(n_rows_dataset, decrement);
    const auto start_offset = n_rows_dataset - n_rows_train;
    const auto num_chunks = OnlineMIMOSVR::get_num_chunks(n_rows_train, max_chunk_size);
    std::deque<arma::uvec> indexes(num_chunks);
    if (num_chunks == 1) {
        indexes[0] = start_offset + arma::regspace<arma::uvec>(0, n_rows_train - 1);
        LOG4_DEBUG("Linear single chunk indexes size " << arma::size(indexes[0]));
        return indexes;
    }
    const size_t this_chunk_size = n_rows_train / ((num_chunks + 1. / C_chunk_offlap - C_end_chunks) * C_chunk_offlap);
    const size_t last_part_size = this_chunk_size * C_chunk_tail;
    const size_t first_part_chunk_size = this_chunk_size - last_part_size;
    const arma::uvec first_part = arma::regspace<arma::uvec>(0, n_rows_train - last_part_size - 1);
    const arma::uvec last_part = arma::regspace<arma::uvec>(n_rows_train - last_part_size, n_rows_train - 1);
    LOG4_DEBUG("Num rows is " << n_rows_dataset << ", decrement " << decrement << ", rows trained " << n_rows_train << ", num chunks " << num_chunks <<
                              ", max chunk size " << max_chunk_size << ", chunk size " << this_chunk_size << ", chunk first part size " << first_part_chunk_size
                              << ", last part size " << last_part_size << ", start offset " << start_offset << ", first part " << arma::size(first_part) <<
                              ", last part " << arma::size(last_part) << ", chunk offlap " << C_chunk_offlap);
OMP_FOR(num_chunks)
    for (size_t i = 0; i < num_chunks; ++i) {
        const size_t start_row = i * this_chunk_size * C_chunk_offlap;
        arma::uvec first_part_rows = arma::regspace<arma::uvec>(start_row, start_row + first_part_chunk_size - 1);
        // first_part_rows.rows(arma::find(first_part_rows >= first_part.n_rows)) -= first_part.n_rows;
        first_part_rows.shed_rows(arma::find(first_part_rows >= first_part.n_rows));
        indexes[i] = start_offset + arma::join_cols(first_part.rows(first_part_rows), last_part);
        if (!i || i == num_chunks - 1)
            LOG4_DEBUG("Chunk " << i << " first part " << first_part_rows.front() << ".." << first_part_rows.back() << ", last part " << last_part.front() << ".." <<
                                last_part.back() << ", start row " << start_row);
    }
    return indexes;
}

std::deque<arma::uvec>
OnlineMIMOSVR::generate_indexes(const unsigned n_rows_dataset, const unsigned decrement, const unsigned max_chunk_size)
{
    return generate_chunk_indexes(n_rows_dataset, decrement, max_chunk_size);
}

std::deque<arma::uvec> OnlineMIMOSVR::generate_indexes() const
{
    return projection ?
        generate_indexes(p_labels->n_rows, (**param_set.cbegin()).get_svr_decremental_distance(), max_chunk_size) :
           std::deque<arma::uvec>{arma::regspace<arma::uvec>(p_labels->n_rows - max_chunk_size, p_labels->n_rows - 1)};
}

arma::uvec
OnlineMIMOSVR::get_other_ixs(const unsigned i) const
{
    return arma::find(ixs[i] != arma::linspace<arma::uvec>(0, p_features->n_rows - 1, p_features->n_rows));
}


void mkl_dgesv(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    MKL_INT n = a.n_rows;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = n;
    MKL_INT ldb = n;
    solved = b;
    std::vector<MKL_INT> ipiv(a.n_rows);
    MKL_INT info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, const_cast<double *>(a.mem), lda, ipiv.data(), solved.memptr(), ldb);
    if (info > 0) LOG4_THROW("Something is wrong in solving LAPACKE_dgesv, error is  " << info << ". The solution could not be computed.");
}


void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    MKL_INT n = a.n_rows;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = n;
    MKL_INT ldb = n;
    solved = b;
    MKL_INT info = LAPACKE_dposv(
            LAPACK_COL_MAJOR, 'U', n, nrhs, const_cast<double *>(a.mem), lda, solved.memptr(), ldb);
    //if (info != 0) LOG4_THROW("DPOSV failed, the leading minor of order " << info << " is not positive definite.");
    if (info == 0) return;
    LOG4_WARN("DPOSV failed, the leading minor of order " << info << " is not positive definite, trying DGESV . . .");
    mkl_dgesv(a, b, solved);
}


void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    static std::mutex solve_mtx;
    const std::scoped_lock l(solve_mtx);
    MKL_INT m = a.n_rows;
    MKL_INT n = a.n_cols;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = m;
    MKL_INT ldb = m;
    solved = b;

    lapack_int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, const_cast<double *>(a.mem), lda, solved.memptr(), ldb);
    if (info > 0) LOG4_THROW("LAPACKE_dgels failed with " << info);
}

//Assuming objects are equal if sizes of the containers are the same
bool OnlineMIMOSVR::operator==(OnlineMIMOSVR const &o) const
{
    if (p_features->n_elem != o.p_features->n_elem
        || p_labels->n_elem != o.p_labels->n_elem
        || param_set != o.param_set
        || ixs.size() != o.ixs.size())
        return false;
    for (size_t i = 0; i < ixs.size(); ++i) {
        const auto &ith_vec = ixs[i] == o.ixs[i];
        if (any(ith_vec < 1)) return false;
    }
    return true;
}

void OnlineMIMOSVR::reset()
{
    samples_trained = 0;
    p_features = ptr<arma::mat>();
    p_labels = ptr<arma::mat>();
    p_last_knowns = ptr<arma::vec>();
    p_kernel_matrices = ptr<std::deque<arma::mat>>();

    weight_chunks.clear();
    all_weights.clear();
    ixs.clear();
    train_feature_chunks_t.clear();
    train_label_chunks.clear();
}

bool OnlineMIMOSVR::is_gradient() const
{
    return std::any_of(C_default_exec_policy, param_set.cbegin(), param_set.cend(), [](const auto &p) -> bool { return p->get_grad_level(); });
}

/* original features_t before matrix transposition
  = {   label 0 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ,
        label 1 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ..
        ...
        label 6000 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 }
     }
*/

mat_ptr OnlineMIMOSVR::all_cumulatives(const SVRParameters &p, const arma::mat &features_t)
{
    const auto lag = p.get_lag_count();
    const auto levels = features_t.n_rows / lag;
    auto cuml = ptr<arma::mat>(arma::size(features_t), arma::fill::none);
OMP_FOR(levels)
    for (size_t i = 0; i < levels; ++i)
        cuml->rows(i * lag, (i + 1) * lag - 1) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, from features t " << arma::size(features_t));
    return cuml;
}


std::shared_ptr<std::deque<arma::mat>> OnlineMIMOSVR::prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t)
{
    const auto lag = params.get_lag_count();
    const auto levels = features_t.n_rows / lag;
    const auto p_cums = otr<std::deque<arma::mat>>(levels);
    OMP_FOR_i(levels) p_cums->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, parameters " << params << ", from features_t " << arma::size(features_t));
    return p_cums;
}


double OnlineMIMOSVR::calc_epsco(const arma::mat &K)
{
    double K_mean;
    ip_errchk(ippsMean_64f(K.mem, K.n_elem, &K_mean));
    return 1. - K_mean;
}


double OnlineMIMOSVR::calc_gamma(const arma::mat &Z, const double train_L_m)
{
    const auto Z_m = common::mean(Z);
    const auto g = kernel::path::calc_g(Z.n_rows, Z_m, train_L_m);
    LOG4_TRACE("Mean Z " << Z_m << ", mean L " << train_L_m << ", n " << Z.n_rows << ", gamma " << g);
    return g;
}


mat_ptr OnlineMIMOSVR::prepare_Z(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    auto p_Z = ptr<arma::mat>(features_t.n_cols, features_t.n_cols, arma::fill::none);
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            kernel::path::cu_distances_xx(features_t.n_cols, features_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(),
                                          ccache.get_cached_cumulatives(params, features_t, time).mem, p_Z->memptr());
            LOG4_TRACE("Returning path Z " << common::present(*p_Z) << " for " << params << ", features " << arma::size(features_t));
            break;
        }
        default:
            IKernel<double>::get(params)->operator()(features_t.t(), *p_Z);
            break;
    }
    return p_Z;
}


mat_ptr OnlineMIMOSVR::prepare_Z(const SVRParameters &params, const arma::mat &features_t)
{
    const auto len = features_t.n_cols;
    auto p_Z = ptr<arma::mat>(len, len, arma::fill::none);
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            kernel::path::cu_distances_xx(len, features_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(),
                                          all_cumulatives(params, features_t)->mem, p_Z->memptr());
            LOG4_DEBUG("Returning path Z " << arma::size(*p_Z) << " for " << params << ", features " << arma::size(features_t));
            break;
        }
        default:
            IKernel<double>::get(params)->operator()(features_t.t(), *p_Z);
            break;
    }
    return p_Z;
}


mat_ptr OnlineMIMOSVR::prepare_Zy(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t,
                                  const bpt::ptime &time, const bpt::ptime &trained_time)
{
    assert(features_t.n_rows == predict_features_t.n_rows);
    mat_ptr p_Zy;
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            p_Zy = ptr<arma::mat>(predict_features_t.n_cols, features_t.n_cols, arma::fill::none);
            kernel::path::cu_distances_xy(features_t.n_cols, predict_features_t.n_cols, features_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(),
                                          ccache.get_cached_cumulatives(params, features_t, time).mem,
                                          ccache.get_cached_cumulatives(params, predict_features_t, trained_time).mem, p_Zy->memptr());

            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << ", features t " << arma::size(features_t) << ", predict features t " << arma::size(predict_features_t) <<
                                       " for parameters " << params);
            break;
        }
        case e_kernel_type::DEEP_PATH:
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }
    return p_Zy;
}


mat_ptr OnlineMIMOSVR::prepare_Zy(const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t)
{
    assert(features_t.n_rows == predict_features_t.n_rows);
    mat_ptr p_Zy;
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            p_Zy = ptr<arma::mat>(predict_features_t.n_cols, features_t.n_cols, arma::fill::none);
            kernel::path::cu_distances_xy(
                    features_t.n_cols, predict_features_t.n_cols, features_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(),
                    all_cumulatives(params, features_t)->mem, all_cumulatives(params, predict_features_t)->mem, p_Zy->memptr());
            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << ", features t " << arma::size(features_t) << ", predict features t " << arma::size(predict_features_t) <<
                                       " for parameters " << params);
            break;
        }
        case e_kernel_type::DEEP_PATH:
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }
    return p_Zy;
}


mat_ptr OnlineMIMOSVR::prepare_K(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &x_t, const bpt::ptime &time)
{
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            auto K = ptr<arma::mat>(x_t.n_cols, x_t.n_cols, arma::fill::none);
            const auto &cuml = ccache.get_cached_cumulatives(params, x_t, time);
            kernel::path::cu_kernel_xx(x_t.n_cols, x_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(), params.get_svr_kernel_param(), cuml.mem, K->memptr());
            // Calculating K from Z seems to lower precision, disabled temporarily
            // Original solvers::kernel_from_distances(K->memptr(), ccache.get_cached_Z(params, x_t, time).mem, K->n_rows, K->n_cols, params.get_svr_kernel_param());
            return K;
        }
        default:
            LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    return {};
}


mat_ptr OnlineMIMOSVR::prepare_K(const SVRParameters &params, const arma::mat &x_t)
{
    switch (params.get_kernel_type()) {
        case e_kernel_type::PATH: {
            const auto K = ptr<arma::mat>(x_t.n_cols, x_t.n_cols, arma::fill::none);
            const auto cuml = all_cumulatives(params, x_t);
            kernel::path::cu_kernel_xy(x_t.n_cols, x_t.n_cols, x_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(),
                                       params.get_svr_kernel_param(), cuml->mem, cuml->mem, K->memptr());
            LOG4_TRACE("Cuml " << common::present(*cuml) << ", K " << common::present(*K));
            return K;
        }
        default:
            LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    return {};
}


mat_ptr OnlineMIMOSVR::prepare_Ky(
        business::calc_cache &ccache, const datamodel::SVRParameters &params, const arma::mat &x_train_t, const arma::mat &x_predict_t,
        const bpt::ptime &predict_time, const bpt::ptime &trained_time)
{
    assert(x_train_t.n_rows == x_predict_t.n_rows);
    const auto Ky = ptr<arma::mat>(x_predict_t.n_cols, x_train_t.n_cols, arma::fill::none);
    switch (params.get_kernel_type()) {
        case datamodel::e_kernel_type::PATH: {
            // Calculating Ky from Zy seems to lower precision, disabled temporarily
            // solvers::kernel_from_distances(
            //        Ky->memptr(), ccache.get_cached_Zy(params, features_t, predict_features_t, time).mem, Ky->n_rows, Ky->n_cols, params.get_svr_kernel_param());
            const auto cuml_train = ccache.get_cached_cumulatives(params, x_train_t, trained_time);
            const auto cuml_predict = ccache.get_cached_cumulatives(params, x_predict_t, predict_time);
            kernel::path::cu_kernel_xy(
                    x_train_t.n_cols, x_predict_t.n_cols, x_train_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(), params.get_svr_kernel_param(),
                    cuml_train.mem, cuml_predict.mem, Ky->memptr());
            break;
        }
        default:
            LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    LOG4_DEBUG("Predict kernel matrix of size " << arma::size(*Ky) << ", trained features t " << arma::size(x_train_t) << ", predict features t "
                                                << arma::size(x_predict_t));
    return Ky;
}


mat_ptr OnlineMIMOSVR::prepare_Ky(const datamodel::SVRParameters &params, const arma::mat &x_train_t, const arma::mat &x_predict_t)
{
    assert(x_train_t.n_rows == x_predict_t.n_rows);
    switch (params.get_kernel_type()) {
        case datamodel::e_kernel_type::PATH: {
            auto Ky = ptr<arma::mat>(x_predict_t.n_cols, x_train_t.n_cols, arma::fill::none);
            auto cuml_train = all_cumulatives(params, x_train_t);
            auto cuml_predict = all_cumulatives(params, x_predict_t);
            kernel::path::cu_kernel_xy(
                    x_train_t.n_cols, x_predict_t.n_cols, x_train_t.n_rows, params.get_lag_count(), params.get_svr_kernel_param2(), params.get_svr_kernel_param(),
                    cuml_train->mem, cuml_predict->mem, Ky->memptr());
            return Ky;
        }
        default:
            LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    return {};
}

} // datamodel
} // svr
