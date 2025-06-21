#include <petscksp.h>
#include <petscsnes.h>
#include <algorithm>
#include <oneapi/mkl.hpp>

#define ARMA_DONT_USE_LAPACK
#undef ARMA_USE_LAPACK

#include <mpi.h>
#include <armadillo>
#include <ipp/ipp.h>
#include <cmath>
#include <deque>
#include <execution>
#include <magma_v2.h>
#include <memory>
#include <sys/mman.h>
#include <mpi.h>
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
#include "calc_cache.hpp"
#include "kernel_factory.hpp"
#include "pprune.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <osqp/osqp.h>
#include <matplotlibcpp.h>
#endif

namespace svr {
namespace datamodel {


class onlinesvr_lib_init {
public:
    onlinesvr_lib_init()
    {
        mlockall(MCL_CURRENT | MCL_FUTURE);
        ip_errchk(ippInit());

        ma_errchk(magma_init());
#ifdef USE_MPI
        static int zero = 0;
        int provided = 0;
        MPI_Init_thread(&zero, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided != MPI_THREAD_MULTIPLE) LOG4_ERROR("The MPI implementation " << provided << " does not support MPI_THREAD_MULTIPLE.");

        // Get MPI rank and size
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (rank == 0) LOG4_DEBUG("Running with " << size << " MPI processes.");
#endif
    }

    ~onlinesvr_lib_init()
    {
#ifdef USE_MPI
        MPI_Finalize();
#endif
        munlockall();
    }
};

const auto __lib_init = []() {
    return onlinesvr_lib_init();
}();

OnlineSVR::OnlineSVR() : Entity(0), multiout(PROPS.get_multiout()), max_chunk_size(PROPS.get_kernel_length()), chunk_offlap(1 - PROPS.get_chunk_overlap())
{
    LOG4_WARN("Created OnlineMIMOSVR object with default constructor and default multistep_len " << multiout);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

OnlineSVR::OnlineSVR(
        const bigint id,
        const bigint model_id,
        const t_param_set &param_set,
        const Dataset_ptr &p_dataset) :
        Entity(id), model_id(model_id), p_dataset(p_dataset), param_set(param_set), multiout(PROPS.get_multiout()), max_chunk_size(PROPS.get_kernel_length()), chunk_offlap(1 - PROPS.get_chunk_overlap())
{
    if (model_id) scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model_id);
    parse_params();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

OnlineSVR::OnlineSVR(
        const bigint id,
        const bigint model_id,
        const t_param_set &param_set,
        const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const bpt::ptime &last_value_time,
        const matrices_ptr &kernel_matrices,
        const Dataset_ptr &p_dataset) :
        Entity(id), model_id(model_id), p_dataset(p_dataset), param_set(param_set), multiout(PROPS.get_multiout()), max_chunk_size(PROPS.get_kernel_length()), chunk_offlap(1 - PROPS.get_chunk_overlap())
{
    if (model_id) scaling_factors = APP.dq_scaling_factor_service.find_all_by_model_id(model_id);
    parse_params();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    PROFILE_MSG(
            batch_train(p_xtrain, p_ytrain, nullptr, last_value_time, kernel_matrices),
            "Batch SVM train on " << arma::size(*p_ytrain) << " labels and " << arma::size(*p_xtrain) << " features, parameters " << *front(param_set));
}

void OnlineSVR::parse_params()
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

void OnlineSVR::set_dataset(const Dataset_ptr &p_dataset_)
{
    p_dataset = p_dataset_;
}

DTYPE(OnlineSVR::p_dataset) OnlineSVR::get_dataset() const
{
    return p_dataset;
}

DTYPE(OnlineSVR::p_dataset) &OnlineSVR::get_dataset()
{
    return p_dataset;
}

DTYPE(OnlineSVR::samples_trained) OnlineSVR::get_samples_trained_number() const noexcept
{
    return samples_trained;
}

void OnlineSVR::clear_kernel_matrix()
{
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_kernel_matrices->size()))
    for (auto &k: *p_kernel_matrices) k.clear();
}

DTYPE(OnlineSVR::gradient) OnlineSVR::get_gradient_level() const noexcept
{
    return gradient;
}

DTYPE(OnlineSVR::level) OnlineSVR::get_decon_level() const noexcept
{
    return level;
}

DTYPE(OnlineSVR::step) OnlineSVR::get_step() const noexcept
{
    return step;
}

DTYPE(OnlineSVR::multiout) OnlineSVR::get_multiout() const noexcept
{
    return multiout;
}

arma::uvec OnlineSVR::get_active_ixs() const
{
    arma::uvec active_ixs;
    for (const auto &ix: ixs) active_ixs.insert_rows(active_ixs.n_rows, ix);
    return arma::unique(active_ixs);
}

DTYPE(OnlineSVR::param_set) OnlineSVR::get_param_set() const noexcept
{
    return param_set;
}

DTYPE(OnlineSVR::param_set) &OnlineSVR::get_param_set() noexcept
{
    return param_set;
}

void OnlineSVR::set_param_set(const DTYPE(OnlineSVR::param_set) &param_set_)
{
    for (const auto &new_p: param_set_)
        set_params(new_p, new_p->get_chunk_index());
}

void OnlineSVR::set_params(const SVRParameters_ptr &p_svr_parameters_, const uint16_t chunk_ix)
{
    auto it_target_params = business::SVRParametersService::find(param_set, chunk_ix, gradient);
    if (it_target_params == param_set.cend())
        param_set.emplace(p_svr_parameters_);
    else
        **it_target_params = *p_svr_parameters_;
}

void OnlineSVR::set_params(const SVRParameters &param, const uint16_t chunk_ix)
{
    auto p_target_params = get_params_ptr(chunk_ix);
    if (p_target_params)
        *p_target_params = param;
    else
        param_set.emplace(ptr<SVRParameters>(param));
}

SVRParameters &OnlineSVR::get_params(const uint16_t chunk_ix) const
{
    return **business::SVRParametersService::find(param_set, chunk_ix, gradient);
}

SVRParameters_ptr OnlineSVR::get_params_ptr(const uint16_t chunk_ix) const
{
    return business::SVRParametersService::find_ptr(param_set, chunk_ix, gradient);
}

business::calc_cache &OnlineSVR::ccache()
{
    return p_dataset->get_calc_cache();
}

arma::mat &OnlineSVR::get_features()
{
    return *p_features;
}

arma::mat &OnlineSVR::get_labels()
{
    return *p_labels;
}


const dq_scaling_factor_container_t &OnlineSVR::get_scaling_factors() const
{
    return scaling_factors;
}

void OnlineSVR::set_scaling_factor(const DQScalingFactor_ptr &p_sf)
{
    business::DQScalingFactorService::add(scaling_factors, p_sf);
}

void OnlineSVR::set_scaling_factors(const dq_scaling_factor_container_t &new_scaling_factors)
{
    business::DQScalingFactorService::add(scaling_factors, new_scaling_factors);
}

bool OnlineSVR::needs_tuning(const t_param_set &param_set)
{
    if (PROPS.get_tune_parameters()) return true;
    for (const auto &p: param_set)
        if (p->get_svr_kernel_param() == 0) return true;
    return false;
}

bool OnlineSVR::needs_tuning() const
{
    return needs_tuning(param_set) || ixs.empty();
}

void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

#ifdef ENABLE_OPENCL

// Unstable avoid! OpenCL
arma::mat OnlineMIMOSVR::do_ocl_solve(CPTRd host_a, double *host_b, const int m, const uint32_t nrhs)
{
    if (m * (m + nrhs) * sizeof(double) > common::gpu_handler_1::get().get_max_gpu_data_chunk_size())
        THROW_EX_FS(std::runtime_error,
                "Not enough memory on GPU, required " << m * nrhs * sizeof(double) << " bytes, available " <<
                     common::gpu_handler_1::get().get_max_gpu_data_chunk_size() << " bytes.");

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
}

#endif

arma::mat OnlineSVR::self_predict(const arma::mat &K, const arma::mat &w, const arma::mat &rhs)
{
    arma::mat diff(arma::size(rhs), ARMA_DEFAULT_FILL);
    self_predict(K.n_rows, rhs.n_cols, K.mem, w.mem, rhs.mem, diff.memptr());
    return diff;
}

void OnlineSVR::self_predict(const uint32_t m, const uint32_t n, CRPTRd K, CRPTRd w, CRPTRd rhs, RPTR(double) diff)
{
    memcpy(diff, rhs, m * n * sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m, 1., K, m, w, m, -1., diff, m);
}

double OnlineSVR::score_weights(const uint32_t m, const uint32_t n, CRPTRd K, CRPTRd w, CRPTRd rhs)
{
    const auto mn = m * n;
    auto diff = (double *const) malloc(mn * sizeof(double));
    self_predict(m, n, K, w, rhs, diff);
    // const auto res =  common::medianabs(diff, mn);
    const auto res = cblas_dasum(mn, diff, 1);
    // const auto res = common::stdscore(diff, mn);
    free(diff);
    return res;
}


// TODO Buggy, rewrite and test
std::deque<arma::mat> OnlineSVR::solve_batched_irwls(
        const std::deque<arma::mat> &K_epsco, const std::deque<arma::mat> &K, const std::deque<arma::mat> &rhs, const size_t iters, const magma_queue_t &magma_queue,
        const size_t gpu_phy_id)
{
    auto solved = rhs;
    const auto batch_size = K.size();
    const size_t m = K.front().n_rows, n = rhs.front().n_cols;

    if (K.size() != K_epsco.size() || K_epsco.size() != rhs.size()) LOG4_THROW("Sizes not correct " << K.size() << ", " << K_epsco.size() << ", " << rhs.size());
    LOG4_TRACE("Batch size " << K.size() << ", m " << m << ", n " << n << ", K front " << arma::size(K.front()) << ", K back " << arma::size(K.back()) << ", rhs front "
                             << arma::size(rhs.front()) << ", rhs back " << arma::size(rhs.back()) << ", solved front " << arma::size(solved.front()) << ", solved back "
                             << arma::size(solved.back()));
    cu_errchk(cudaSetDevice(gpu_phy_id));
    auto [d_K, d_rhs] = solvers::init_magma_batch_solver(batch_size, m, n);

    solvers::iter_magma_batch_solve(m, n, K_epsco, rhs, solved, magma_queue, d_K, d_rhs, gpu_phy_id);

    std::vector<double> best_sae(batch_size, std::numeric_limits<double>::infinity());
    std::vector<size_t> best_iter(batch_size, 0);
    const auto range_iters = common::C_itersolve_range / double(iters);
    DTYPE(solved) best_solution(batch_size);
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


arma::mat OnlineSVR::direct_solve(const arma::mat &a, const arma::mat &b)
{
    LOG4_THROW("Deprecated");
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));
    arma::mat solved = arma::zeros(arma::size(b));
//    PROFILE_MSG(solvers::qrsolve(a.n_rows, b.n_cols, a.mem, b.mem, solved.memptr()), "qrsolve");
    return solved;
}

uint32_t OnlineSVR::get_full_train_len(const uint32_t n_rows, const uint32_t decrement)
{
    return n_rows > PROPS.get_shift_limit() ? std::min<uint32_t>(n_rows - PROPS.get_shift_limit(), decrement) : decrement;
}

uint32_t OnlineSVR::get_num_chunks(const uint32_t n_rows, const uint32_t chunk_size_)
{
    if (!n_rows || !chunk_size_) LOG4_THROW("Rows count " << n_rows << " or maximum chunk size " << chunk_size_ << " is zero!");
    if (n_rows <= chunk_size_) return 1;
    static const float chunk_offlap = 1 - PROPS.get_chunk_overlap();
    return cdiv(n_rows, chunk_size_ * chunk_offlap) - cdiv(1, chunk_offlap) + C_end_chunks;
}

uint32_t OnlineSVR::get_num_chunks() const
{
    return projection ? get_num_chunks(get_full_train_len(p_labels ? p_labels->n_rows : 0, (**param_set.cbegin()).get_svr_decremental_distance()), max_chunk_size) : 1;
}

std::deque<arma::uvec> OnlineSVR::generate_indexes() const
{
    const auto n_rows_dataset = p_labels->n_rows;
    const auto interleave_sqrt = std::sqrt(PROPS.get_interleave());
    if (is_manifold()) return {arma::regspace<arma::uvec>(0, interleave_sqrt, n_rows_dataset - 1)};
    const auto decrement = (*param_set.cbegin())->get_svr_decremental_distance();
    // Make sure we train on the latest data
    const auto n_rows_train = get_full_train_len(n_rows_dataset, decrement);
    assert(n_rows_dataset >= n_rows_train);
    const auto start_offset = n_rows_dataset - n_rows_train;
    const auto num_chunks = get_num_chunks(n_rows_train, max_chunk_size);
    assert(num_chunks);
    std::deque<arma::uvec> indexes(num_chunks);
    const uint32_t this_chunk_size = n_rows_train / ((num_chunks + 1 / chunk_offlap - C_end_chunks) * chunk_offlap);
    const uint32_t outlier_slack = PROPS.get_outlier_slack();
    const uint32_t skip = is_tft().get() == nullptr ? 1 : interleave_sqrt;
    LOG4_DEBUG("Num rows is " << n_rows_dataset << ", decrement " << decrement << ", rows trained " << n_rows_train << ", num chunks " << num_chunks << ", max chunk size " <<
            max_chunk_size << ", chunk size " << this_chunk_size << ", start offset " << start_offset << ", chunk offlap " << chunk_offlap << ", outlier slack " << outlier_slack <<
            ", skip " << skip);
    OMP_FOR_i(num_chunks) {
        const uint32_t start_row = start_offset + i * this_chunk_size * chunk_offlap;
        if (const auto end_row = start_row + this_chunk_size; end_row >= n_rows_dataset)
            indexes[i] = arma::regspace<arma::uvec>(start_row, skip, n_rows_dataset - 1);
        else
            indexes[i] = arma::join_cols(arma::regspace<arma::uvec>(start_row, skip, end_row - 1), arma::regspace<arma::uvec>(end_row, end_row + outlier_slack - 1));
        if (!i || i == DTYPE(i)(num_chunks - 1)) LOG4_DEBUG("Chunk " << i << ", start row " << start_row << ", chunk len " << this_chunk_size << ", indexes " << common::present(indexes[i]));
    }
    return indexes;
}

arma::uvec
OnlineSVR::get_other_ixs(const uint16_t i) const
{
    return ixs[i](arma::find(ixs[i] != arma::linspace<arma::uvec>(0, p_features->n_rows - 1, p_features->n_rows)));
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
    MKL_INT info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', n, nrhs, const_cast<double *>(a.mem), lda, solved.memptr(), ldb);
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
bool OnlineSVR::operator==(OnlineSVR const &o) const
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

void OnlineSVR::reset()
{
    samples_trained = 0;
    p_features = ptr<arma::mat>();
    p_labels = ptr<arma::mat>();
    p_kernel_matrices = ptr<std::deque<arma::mat>>();

    weight_chunks.clear();
    all_weights.clear();
    ixs.clear();
    train_feature_chunks_t.clear();
    train_label_chunks.clear();
}

bool OnlineSVR::is_gradient() const
{
    return std::any_of(C_default_exec_policy, param_set.cbegin(), param_set.cend(), [](const auto &p) -> bool { return p->get_grad_level(); });
}


double OnlineSVR::calc_epsco(const arma::mat &K, const arma::mat &labels)
{
    return arma::mean(arma::mean(labels, 1) - arma::sum(K, 1) - arma::sum(arma::vectorise(labels)));
}


std::tuple<double, double> OnlineSVR::calc_gamma(const arma::mat &Z, const arma::mat &L)
{
    const auto ref = kernel::get_reference_Z(L);
    const auto mean = common::mean(Z) - common::mean(ref);
    arma::mat Z_ = mean != 0 ? Z - mean : Z;
    const auto gamma = common::meanabs(Z_) / common::meanabs(ref);
    return {gamma, mean};
}

arma::vec OnlineSVR::calc_gammas(const arma::mat &Z, const arma::mat &L)
{
    arma::vec mean_Z, min_Z, max_Z, min_L, max_L, mean_L;

#pragma omp parallel ADJ_THREADS(6)
#pragma omp single
    {
#pragma omp task
        mean_Z = arma::mean(Z, 1);
#pragma omp task
        min_Z = arma::min(Z, 1);
#pragma omp task
        max_Z = arma::max(Z, 1);
#pragma omp task
        min_L = arma::min(L, 1);
#pragma omp task
        max_L = arma::max(L, 1);
#pragma omp task
        mean_L = arma::mean(L, 1);
    }

    arma::vec g_row(Z.n_rows, ARMA_DEFAULT_FILL);
    OMP_FOR_i(Z.n_rows) {
        // const auto min_qgamma = kernel::path::calc_qgamma(mean_Z(i), min_Z(i), mean_L(i), min_L(i), Z.n_cols);
        // g_row(i) = bias * (kernel::path::calc_qgamma(mean_Z(i), max_Z(i), mean_L(i), max_L(i), Z.n_cols) - min_qgamma) + min_qgamma;
    }
    LOG4_DEBUG("Xh-row " << common::present(g_row) << ", Z " << arma::size(Z) << ", L " << arma::size(L));
    return g_row;
}

datamodel::SVRParameters_ptr OnlineSVR::is_manifold() const
{
    return business::SVRParametersService::is_manifold(param_set);
}

datamodel::SVRParameters_ptr OnlineSVR::is_tft() const
{
    return business::SVRParametersService::is_tft(param_set);
}

} // datamodel
} // svr
