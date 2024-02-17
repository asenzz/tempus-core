#include <armadillo>
#include </opt/intel/oneapi/mkl/latest/include/mkl_lapacke.h>
#include <cmath>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "onlinesvr.hpp"
#include "mat_solve_gpu.hpp"
#include "cuqrsolve.hpp"
#include "kernel_factory.hpp"
#include "model/Model.hpp"
#include "appcontext.hpp"
#include "SVRParametersService.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif


namespace svr {


OnlineMIMOSVR::OnlineMIMOSVR() : model_type(mimo_type_e::single), multistep_len(common::__multistep_len)
{
    init_model_base(0, mimo_type_e::single);
    LOG4_WARN("Created OnlineMIMOSVR object with default constructor and default multistep_len " << multistep_len);
}


OnlineMIMOSVR::OnlineMIMOSVR(
        const t_param_set_ptr &p_param_set_,
        const size_t multistep_len_,
        const size_t max_chunk_size_) :
        p_param_set(p_param_set_), multistep_len(multistep_len_), max_chunk_size(max_chunk_size_)
{
    const auto p = front(*p_param_set);
    gradient = p->get_grad_level();
    decon_level = p->get_decon_level();
    init_model_base(p->get_svr_epsilon(), p->get_model_type());
}


OnlineMIMOSVR::OnlineMIMOSVR(
        const t_param_set_ptr &p_param_set_,
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const matrices_ptr &kernel_matrices,
        const size_t multistep_len_,
        const size_t max_chunk_size_) :
        p_param_set(p_param_set_), multistep_len(multistep_len_), max_chunk_size(max_chunk_size_)
{
    if (p_param_set->empty()) LOG4_THROW("At least one parameter set should be supplied to construct an SVR model.");
    const auto p = front(*p_param_set);
    gradient = p->get_grad_level();
    decon_level = p->get_decon_level();
    init_model_base(p->get_svr_epsilon(), p->get_model_type());

    PROFILE_EXEC_TIME(
            batch_train(p_xtrain, p_ytrain, kernel_matrices),
            "Batch SVM train on " << arma::size(*p_ytrain) << " labels and " << arma::size(*p_xtrain) << " features, parameters " << *front(*p_param_set));
}

datamodel::t_param_set_ptr OnlineMIMOSVR::get_param_set_ptr() const
{
    return p_param_set;
}

datamodel::t_param_set_ptr &OnlineMIMOSVR::get_param_set_ptr()
{
    return p_param_set;
}

svr::datamodel::t_param_set OnlineMIMOSVR::get_param_set() const
{
    return *p_param_set;
}

svr::datamodel::t_param_set &OnlineMIMOSVR::get_param_set()
{
    return *p_param_set;
}

void OnlineMIMOSVR::set_param_set(const datamodel::t_param_set_ptr &p_svr_param_set)
{
    p_param_set = p_svr_param_set;
}

void OnlineMIMOSVR::set_param_set(const datamodel::t_param_set &svr_param_set_)
{
    *p_param_set = svr_param_set_;
}

void OnlineMIMOSVR::set_params(const datamodel::SVRParameters_ptr &p_svr_parameters_, const size_t chunk_ix)
{
    get_params_ptr(chunk_ix) = p_svr_parameters_;
}

void OnlineMIMOSVR::set_params(const datamodel::SVRParameters &svr_parameters_, const size_t chunk_ix)
{
    get_params(chunk_ix) = svr_parameters_;
}

datamodel::SVRParameters_ptr OnlineMIMOSVR::get_params_ptr(const size_t chunk_ix) const
{
    return business::SVRParametersService::find(*p_param_set, chunk_ix, gradient);
}

datamodel::SVRParameters &OnlineMIMOSVR::get_params(const size_t chunk_ix)
{
    return *business::SVRParametersService::find(*p_param_set, chunk_ix, gradient);
}

datamodel::SVRParameters OnlineMIMOSVR::get_params(const size_t chunk_ix) const
{
    return *business::SVRParametersService::find(*p_param_set, chunk_ix, gradient);
}

void OnlineMIMOSVR::init_model_base(const double epsilon_, const mimo_type_e model_type_)
{
    model_type = model_type_;
    switch (model_type) {
        case mimo_type_e::twin:
            LOG4_DEBUG("Create model of type twin");
            main_components.resize(2);
            main_components[0] = std::make_shared<MimoBase>();
            main_components[0]->epsilon = epsilon_;
            main_components[1] = std::make_shared<MimoBase>();
            main_components[1]->epsilon = -epsilon_;
            if (epsilon_ < std::numeric_limits<decltype(epsilon_)>::epsilon()) LOG4_ERROR("Epsilon is zero and twin type SVM selected!");
            break;
        case mimo_type_e::single:
        default:
            LOG4_DEBUG("Create model of type single");
            main_components.resize(1);
            main_components[0] = std::make_shared<MimoBase>();
            main_components[0]->epsilon = epsilon_;
    }
}

arma::mat &OnlineMIMOSVR::get_features()
{
    return *p_features;
}

arma::mat &OnlineMIMOSVR::get_labels()
{
    return *p_labels;
}

arma::mat &OnlineMIMOSVR::get_weights(uint8_t type)
{
    return main_components[type]->chunk_weights[0];
}


arma::mat OnlineMIMOSVR::get_weights(uint8_t type) const
{
    return main_components.at(type)->chunk_weights[0];
}

bool OnlineMIMOSVR::needs_tuning() const
{
    if (PROPS.get_tune_parameters()) return true;

    bool tune_parameters = false;
    for (auto &p: *p_param_set)
        tune_parameters |= p->get_svr_kernel_param() == 0;
    return tune_parameters;
}

arma::mat
OnlineMIMOSVR::init_kernel_matrix(const SVRParameters &params, const arma::mat &x)
{
    arma::mat K;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const arma::mat &Z = get_cached_Z(params, x.t(), x.n_rows);
            if (arma::size(K) != arma::size(Z)) K.set_size(arma::size(Z));
            solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, params.get_svr_kernel_param());
            break;
        }

        default:
            LOG4_THROW("Unhandled kernel.");
    }
    return K;
}

// sum(K.row(n)) = mean_L
// mean_K = mean_L / K.n_cols
// mean_K = 1. - mean_Z / (2. * gamma^2)
// mean_L / K.n_cols = 1. - mean_Z / (2. * gamma^2)
// mean_L / K.n_cols - 1 = mean_Z / (2. * gamma^2)
// (mean_L / K.n_cols - 1) / mean_Z = 1 / (2 * gamma^2)
// mean_Z / (mean_L / K.n_cols - 1) = 2 * gamma^2
// mean_Z / (mean_L / K.n_cols - 1) = 2 * gamma^2
// mean_Z / (2 * (mean_L / K.n_cols - 1)) = gamma^2
// mean_Z / (2 * mean_L / K.n_cols - 2) = gamma^2
// sqrt(mean_Z / (2 * mean_L / K.n_cols - 2)) = gamma


void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

// Unstable avoid!
arma::mat OnlineMIMOSVR::do_ocl_solve(const double *host_a, double *host_b, const int m, const int nrhs)
{
    LOG4_THROW("Deprecated!");
    return {};
#if 0
    if (m * (m + nrhs) * sizeof(double) > svr::common::gpu_handler::get().get_max_gpu_data_chunk_size())
        THROW_EX_FS(std::runtime_error,
                "Not enough memory on GPU, required " << m * nrhs * sizeof(double) << " bytes, available " <<
                     svr::common::gpu_handler::get().get_max_gpu_data_chunk_size() << " bytes.");

    auto context = svr::common::gpu_context();
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
        err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m * m * sizeof(double), host_a, 0, nullptr, nullptr);
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


void OnlineMIMOSVR::solve_irwls(const arma::mat &epsilon_eye_K, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters, const bool psd)
{
    if (K.n_rows != rhs.n_rows) LOG4_THROW("Illegal size K " << arma::size(K) << ", rhs " << arma::size(rhs));

    svr::common::gpu_context ctx;
    const size_t gpu_phy_id = ctx.phy_id();
#ifdef USE_MAGMA
    auto [magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, piv] = solvers::init_magma_solver(K.n_rows, rhs.n_cols, psd, gpu_phy_id);
#else
    auto [cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo] = solvers::init_cusolver(gpu_phy_id, K.n_rows, rhs.n_cols);
#endif

    if (solved.empty() || arma::size(solved) != arma::size(rhs))
        if (solved.n_rows != K.n_rows || solved.n_cols != rhs.n_cols) solved.set_size(K.n_rows, rhs.n_cols);

    {
        const arma::mat left = K + epsilon_eye_K;
#ifdef USE_MAGMA
//        svr::solvers::dyn_magma_solve(left.n_rows, rhs.n_cols, left.mem, rhs.mem, solved.memptr(), magma_queue, piv, d_K, d_rhs);
        svr::solvers::iter_magma_solve(left.n_rows, rhs.n_cols, left.mem, rhs.mem, solved.memptr(), magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, psd, gpu_phy_id);
#else
        svr::solvers::dyn_gpu_solve(gpu_phy_id, left.n_rows, rhs.n_cols, left.mem, rhs.mem, solved.memptr(), cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    }
    double best_sae = common::sum<double>(arma::abs(K * solved - rhs));
    arma::mat best_solution = solved;
    size_t best_iter = 0;
    const auto iters_delta = double(iters) * C_itersolve_delta;
    for (size_t i = 1; i < iters; ++i) {
        const double iter_add = iters_delta / (double(i) * C_itersolve_range);
        const arma::mat error_mat = arma::abs(K * solved - rhs);
        const auto this_sae = common::sum<double>(error_mat);
        if (this_sae < best_sae) {
            LOG4_TRACE("IRWLS iteration " << i << ", add " << iter_add << ", SAE " << this_sae << ", kernel dimensions " << arma::size(K) << ", best SAE " << best_sae << ", iter add " << iter_add);
            best_sae = this_sae;
            best_solution = solved;
            best_iter = i;
        }
        // const arma::mat mult = error_mat / double(K.n_rows); // TODO Test and implement and test conjugate
        const arma::mat mult = 1. / arma::sqrt(error_mat + iter_add);
        const arma::mat left = (arma::mean(mult, 1) * arma::ones(1, K.n_cols)) % K + epsilon_eye_K;
        const arma::mat right = rhs % mult;
#ifdef USE_MAGMA
//       svr::solvers::dyn_magma_solve(left.n_rows, right.n_cols, left.mem, right.mem, solved.memptr(), magma_queue, piv, d_K, d_rhs);
       svr::solvers::iter_magma_solve(left.n_rows, right.n_cols, left.mem, right.mem, solved.memptr(), magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, psd, gpu_phy_id);
#else
        svr::solvers::dyn_gpu_solve(gpu_phy_id, left.n_rows, right.n_cols, left.mem, right.mem, solved.memptr(), cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    }

#ifdef USE_MAGMA
    solvers::uninit_magma_solver(magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, piv, gpu_phy_id);
#else
    solvers::uninit_cusolver(gpu_phy_id, cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif

    LOG4_DEBUG("IRWLS best iteration " << best_iter << ", MAE " << best_sae / double(solved.n_elem) << ", kernel dimensions " << arma::size(K) <<
                ", delta " << C_itersolve_delta << ", range " << C_itersolve_range << ", kernel " << arma::size(K) << ", solution " << arma::size(solved));

    solved = best_solution;
}


arma::mat OnlineMIMOSVR::direct_solve(const arma::mat &a, const arma::mat &b)
{
    LOG4_THROW("Deprecated");
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));
    arma::mat solved = arma::zeros(arma::size(b));
//    PROFILE_EXEC_TIME(svr::solvers::qrsolve(a.n_rows, b.n_cols, a.mem, b.mem, solved.memptr()), "qrsolve");
    return solved;
}


void
OnlineMIMOSVR::solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, arma::mat &solved, const size_t iters, const bool psd)
{
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));
    PROFILE_EXEC_TIME(solve_irwls(epsilon_eye_K, a, b, solved, iters, false), "IRWLS solver " << arma::size(a) << ", " << arma::size(b)); // Emo's IRWLS solver
}


arma::mat
OnlineMIMOSVR::solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, const size_t iters, const bool psd)
{
    arma::mat r;
    solve_dispatch(epsilon_eye_K, a, b, r, iters, psd);
    return r;
}

size_t OnlineMIMOSVR::get_num_chunks(const size_t n_rows, const size_t chunk_size_)
{
    return std::max<size_t>(1, std::ceil(double(n_rows) / double(chunk_size_)));
}

size_t OnlineMIMOSVR::get_num_chunks() const
{
    return get_num_chunks(get_params().get_svr_decremental_distance(), max_chunk_size);
}

std::deque<arma::uvec> OnlineMIMOSVR::get_indexes() const
{
    return OnlineMIMOSVR::get_indexes(p_features->empty() ? p_labels->n_rows : p_features->n_rows, get_params());
}

std::deque<arma::uvec>
OnlineMIMOSVR::get_indexes(const size_t n_rows_dataset, const SVRParameters &svr_parameters) const
{
    return get_indexes(n_rows_dataset, svr_parameters, max_chunk_size);
}

std::deque<arma::uvec>
OnlineMIMOSVR::get_indexes(const size_t n_rows_dataset, const SVRParameters &svr_parameters, const size_t max_chunk_size)
{
    const auto n_rows_train = std::min<size_t>(n_rows_dataset, svr_parameters.get_svr_decremental_distance());
    const auto num_chunks = get_num_chunks(n_rows_train, max_chunk_size);
    std::deque<arma::uvec> indexes(num_chunks);
    if (num_chunks == 1) {
        indexes[0] = arma::linspace<arma::uvec>(0, n_rows_train - 1, n_rows_train);
        LOG4_DEBUG("Linear single chunk indexes size " << arma::size(indexes[0]));
        return indexes;
    }
    const size_t this_chunk_size = n_rows_train / num_chunks;
    const size_t last_part_size = this_chunk_size * .1;
    const size_t first_part_size = this_chunk_size - last_part_size;
    LOG4_DEBUG("Num rows is " << n_rows_dataset << ", decrement distance " << svr_parameters.get_svr_decremental_distance() << ", max indexes len " << n_rows_train <<
        ", num chunks " << num_chunks << ", chunk size " << this_chunk_size << ", first part " << first_part_size << ", last part size " << last_part_size);
    const arma::mat i_full = arma::cumsum(arma::ones(n_rows_train, 1)) - 1;
    const arma::mat first_part = i_full.rows(0, n_rows_train - last_part_size);
    const arma::mat last_part = i_full.rows(n_rows_train - last_part_size + 1, n_rows_train - 1);
    const arma::mat i_shuffle = svr::common::armd::fixed_shuffle(first_part);
#pragma omp parallel for num_threads(adj_threads(num_chunks))
    for (size_t i = 0; i < num_chunks; ++i) {
        const arma::mat m_11 = i_shuffle.rows(i * first_part_size, std::min((i + 1) * first_part_size, n_rows_train - last_part_size));
        indexes[i] = arma::conv_to<arma::uvec>::from(arma::join_cols(m_11, last_part));
    }

    return indexes;
}

arma::uvec
OnlineMIMOSVR::get_other_ixs(const size_t i) const
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
        || *p_param_set != *o.p_param_set
        || model_type != o.model_type
        || main_components.size() != o.main_components.size()
        || ixs.size() != o.ixs.size())
        return false;
    for (size_t i = 0; i < ixs.size(); ++i) {
        const auto &ith_vec = ixs[i] == o.ixs[i];
        if (any(ith_vec < 1)) return false;
    }
    return true;
}

void OnlineMIMOSVR::reset_model()
{
    samples_trained = 0;
    p_features = std::make_shared<arma::mat>();
    p_labels = std::make_shared<arma::mat>();
    p_kernel_matrices = std::make_shared<std::deque<arma::mat>>();
#pragma omp parallel for num_threads(adj_threads(main_components.size())) schedule(static, 1)
    for (auto &component : main_components) {
        component->chunk_weights.clear();
        component->weights_mask.clear();
        component->chunks_weight.clear();
        component->total_weights.clear();
        component->mae_chunk_values.clear();
    }
    ixs.clear();
}

bool OnlineMIMOSVR::is_gradient() const
{
    if (!p_param_set) return false;
    return std::any_of(std::execution::par_unseq, p_param_set->begin(), p_param_set->end(), [](const auto &p) -> bool { return p->get_grad_level(); });
}

} // svr
