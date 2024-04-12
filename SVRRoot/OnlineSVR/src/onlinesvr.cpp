#include <armadillo>
#include </opt/intel/oneapi/mkl/latest/include/mkl_lapacke.h>
#include <cmath>
#include <execution>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "appcontext.hpp"
#include "SVRParametersService.hpp"
#include "onlinesvr.hpp"
#include "mat_solve_gpu.hpp"
#include "cuqrsolve.hpp"
#include "model/Dataset.hpp"
#include "model/Entity.hpp"
#include "model/SVRParameters.hpp"
#include "util/math_utils.hpp"
#include "cuda_path.hpp"
#include "calc_cache.hpp"
#include "kernel_factory.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif


namespace svr {
namespace datamodel {

OnlineMIMOSVR::OnlineMIMOSVR() : Entity(0), multistep_len(common::C_default_multistep_len)
{
    LOG4_WARN("Created OnlineMIMOSVR object with default constructor and default multistep_len " << multistep_len);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}


void OnlineMIMOSVR::parse_params()
{
    if (param_set.empty()) LOG4_THROW("At least one parameter set should be supplied to construct an SVR model.");

    const auto p = *param_set.cbegin();
    gradient = p->get_grad_level();
    decon_level = p->get_decon_level();
    if (p_dataset) {
        multistep_len = p_dataset->get_multiout();
        max_chunk_size = p_dataset->get_max_chunk_size();
    }
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

Dataset_ptr OnlineMIMOSVR::get_dataset() const
{
    return p_dataset;
}

Dataset_ptr &OnlineMIMOSVR::get_dataset()
{
    return p_dataset;
}

ssize_t OnlineMIMOSVR::get_samples_trained_number() const
{
    return samples_trained;
}

void OnlineMIMOSVR::clear_kernel_matrix()
{
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_kernel_matrices->size()))
    for (auto &k: *p_kernel_matrices) k.clear();
}

size_t OnlineMIMOSVR::get_gradient_level() const
{
    return gradient;
}

size_t OnlineMIMOSVR::get_decon_level() const
{
    return decon_level;
}

size_t OnlineMIMOSVR::get_multistep_len() const
{
    return multistep_len;
}

arma::uvec OnlineMIMOSVR::get_active_ixs() const
{
    arma::uvec active_ixs;
    for (const auto &ix: ixs) active_ixs.insert_rows(active_ixs.n_rows, ix);
    return arma::unique(active_ixs);
}

t_param_set OnlineMIMOSVR::get_param_set() const
{
    return param_set;
}

t_param_set &OnlineMIMOSVR::get_param_set()
{
    return param_set;
}

void OnlineMIMOSVR::set_param_set(const t_param_set &param_set_)
{
    for (const auto &new_p: param_set_)
        set_params(new_p, new_p->get_chunk_ix());
}

void OnlineMIMOSVR::set_params(const SVRParameters_ptr &p_svr_parameters_, const size_t chunk_ix)
{
    auto p_target_params = get_params_ptr(chunk_ix);
    if (p_target_params)
        *p_target_params = *p_svr_parameters_;
    else
        param_set.emplace(p_svr_parameters_);
}

void OnlineMIMOSVR::set_params(const SVRParameters &svr_parameters_, const size_t chunk_ix)
{
    auto p_target_params = get_params_ptr(chunk_ix);
    if (p_target_params)
        *p_target_params = svr_parameters_;
    else
        param_set.emplace(ptr<SVRParameters>(svr_parameters_));
}

SVRParameters_ptr OnlineMIMOSVR::get_params_ptr(const size_t chunk_ix) const
{
    return business::SVRParametersService::find(param_set, chunk_ix, gradient);
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
    return needs_tuning(param_set);
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


void OnlineMIMOSVR::solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters, const bool psd)
{
    const svr::common::gpu_context ctx;
    const auto gpu_phy_id = ctx.phy_id();
#ifdef USE_MAGMA
    auto [magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, piv] = solvers::init_magma_solver(K.n_rows, rhs.n_cols, psd, gpu_phy_id);
#else
    auto [cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo] = solvers::init_cusolver(gpu_phy_id, K.n_rows, rhs.n_cols);
#endif
    if (arma::size(solved) != arma::size(rhs)) solved = rhs;

#ifdef USE_MAGMA
    svr::solvers::iter_magma_solve(K_epsco.n_rows, rhs.n_cols, K_epsco.mem, rhs.mem, solved.memptr(), magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, psd, gpu_phy_id);
#else
    svr::solvers::dyn_gpu_solve(gpu_phy_id, K_epsco.n_rows, rhs.n_cols, K_epsco.mem, rhs.mem, solved.memptr(), cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    auto best_sae = std::numeric_limits<double>::infinity();
    arma::mat best_solution = solved;
    size_t best_iter = 0;
    for (size_t i = 1; i < iters; ++i) {
        const arma::mat error_mat = arma::abs(K * solved - rhs);
        const double this_sae = arma::accu(error_mat);
        if (this_sae < best_sae) {
            LOG4_TRACE("IRWLS iteration " << i << ", SAE " << this_sae << ", kernel dimensions " << arma::size(K) << ", best SAE " << best_sae);
            best_sae = this_sae;
            best_solution = solved;
            best_iter = i;
        }
        const arma::mat mult = arma::sqrt(error_mat + C_itersolve_delta / (double(i) * C_itersolve_range / double(iters)));
        const arma::mat left = (mult * arma::ones(mult.n_cols, K.n_cols)) % K_epsco;
        const arma::mat right = rhs % mult;
#ifdef USE_MAGMA
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
                                       ", delta " << C_itersolve_delta << ", range " << C_itersolve_range << ", solution " << arma::size(solved));
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
OnlineMIMOSVR::solve_dispatch(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, arma::mat &w, const size_t iters, const bool psd)
{
    if (K.n_cols != K.n_rows || K.n_rows != y.n_rows || arma::size(K_epsco) != arma::size(K)) LOG4_THROW("Incorrect sizes K " << arma::size(K) << ", y " << arma::size(y));
    PROFILE_EXEC_TIME(solve_irwls(K_epsco, K, y, w, iters, false), "IRWLS solver " << arma::size(K) << ", " << arma::size(y)); // Emo's IRWLS solver
}


arma::mat
OnlineMIMOSVR::solve_dispatch(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &y, const size_t iters, const bool psd)
{
    arma::mat w;
    solve_dispatch(K_epsco, K, y, w, iters, psd);
    return w;
}

size_t OnlineMIMOSVR::get_num_chunks(const size_t n_rows, const size_t chunk_size_)
{
    if (!n_rows || !chunk_size_) {
        LOG4_WARN("Rows count " << n_rows << " or maximum chunk size " << chunk_size_ << " is zero!");
        return 0;
    }
    return std::max<size_t>(1, std::ceil( double(n_rows) / (chunk_size_ * C_chunk_offlap) ));
}


size_t OnlineMIMOSVR::get_num_chunks()
{
    if (is_manifold()) return 0;
    const auto decrement = (**param_set.cbegin()).get_svr_decremental_distance();
    const auto n_rows_train = p_labels && p_labels->n_rows ? std::min<size_t>(p_labels->n_rows, decrement) : decrement;
    return get_num_chunks(n_rows_train, max_chunk_size);
}

std::deque<arma::uvec> OnlineMIMOSVR::generate_indexes() const
{
    return OnlineMIMOSVR::generate_indexes(p_labels->n_rows, (**param_set.cbegin()).get_svr_decremental_distance(), max_chunk_size);
}

std::deque<arma::uvec> // TODO Remove n_rows_datasets
OnlineMIMOSVR::generate_indexes(const size_t n_rows_dataset, const size_t decrement, const size_t max_chunk_size)
{
    size_t n_rows_train, start_offset;
    // Make sure we train on the latest data
    if (n_rows_dataset && n_rows_dataset < decrement) {
        n_rows_train = n_rows_dataset;
        start_offset = 0;
    } else {
        n_rows_train = decrement;
        start_offset = n_rows_dataset - decrement;
    }
    const auto num_chunks = get_num_chunks(n_rows_train, max_chunk_size);
    std::deque<arma::uvec> indexes(num_chunks);
    if (num_chunks == 1) {
        indexes[0] = start_offset + arma::regspace<arma::uvec>(0, n_rows_train - 1);
        LOG4_DEBUG("Linear single chunk indexes size " << arma::size(indexes[0]));
        return indexes;
    }
    const size_t this_chunk_size = n_rows_train / (num_chunks * C_chunk_offlap);
#if 1
    const size_t last_part_size = this_chunk_size * C_chunk_tail;
    const size_t first_part_chunk_size = this_chunk_size - last_part_size;
    const arma::uvec first_part = arma::regspace<arma::uvec>(0, n_rows_train - last_part_size - 1);
    const arma::uvec last_part = arma::regspace<arma::uvec>(n_rows_train - last_part_size, n_rows_train - 1);
    LOG4_DEBUG("Num rows is " << n_rows_dataset << ", decrement " << decrement << ", rows trained " << n_rows_train << ", num chunks " << num_chunks <<
              ", chunk size " << this_chunk_size << ", chunk first part size " << first_part_chunk_size << ", last part size " << last_part_size <<
              ", start offset " << start_offset << ", first part " << arma::size(first_part) << ", last part " << arma::size(last_part) << ", chunk offlap " << C_chunk_offlap);
#pragma omp parallel for num_threads(adj_threads(num_chunks)) schedule(static, 1)
    for (size_t i = 0; i < num_chunks; ++i) {
        const size_t start_row = i * this_chunk_size * C_chunk_offlap;
        arma::uvec first_part_rows = arma::regspace<arma::uvec>(start_row, start_row + first_part_chunk_size - 1);
        first_part_rows.rows(arma::find(first_part_rows >= first_part.n_rows)) -= first_part.n_rows;
        indexes[i] = start_offset + arma::join_cols(first_part.rows(first_part_rows), last_part);
    }
#else
#pragma omp parallel for num_threads(adj_threads(num_chunks)) schedule(static, 1)
    for (size_t i = 0; i < num_chunks; ++i)
        indexes[i] = start_offset + arma::regspace<arma::uvec>(i * this_chunk_size, i < num_chunks - 1 ? (i + 1) * this_chunk_size - 1 : n_rows_train - 1);
#endif
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
    return std::any_of(std::execution::par_unseq, param_set.begin(), param_set.end(), [](const auto &p) -> bool { return p->get_grad_level(); });
}

/* original features_t before matrix transposition
  = {   label 0 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ,
        label 1 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ..
        ...
        label 6000 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 }
     }
*/

std::shared_ptr<std::deque<arma::mat>> OnlineMIMOSVR::prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t)
{
    const auto lag = params.get_lag_count();
    const auto levels = features_t.n_rows / lag;
    const auto p_cums = otr<std::deque<arma::mat>>(levels);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(levels))
    for (size_t i = 0; i < levels; ++i)
        p_cums->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, parameters " << params << ", from features_t " << arma::size(features_t));
    return p_cums;
}


double OnlineMIMOSVR::calc_epsco(const arma::mat &K)
{
    return arma::mean(arma::vectorise(K));

    constexpr ssize_t diags_ct = 1;
    double avg_diag_diff = 0;
    for (ssize_t i = -1; i > -1 - diags_ct; --i)
        avg_diag_diff += arma::mean(arma::vectorise(K.diag(i))) - arma::mean(arma::vectorise(K.diag(i - 1)));
    avg_diag_diff /= double(diags_ct);
    return arma::mean(arma::vectorise(K.diag(-1))) - avg_diag_diff;
}


double OnlineMIMOSVR::calc_gamma(const arma::mat &Z, const double mean_L)
{
    const double mean_Z = arma::mean(arma::vectorise(Z));
    const auto res = std::sqrt(double(Z.n_cols) * mean_Z / (2. * (double(Z.n_cols) - mean_L)));
    LOG4_TRACE("Mean Z " << mean_Z << ", mean L " << mean_L << ", ncols " << Z.n_cols << ", gamma " << res);
    return res;
}


mat_ptr OnlineMIMOSVR::prepare_Z(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const auto len = features_t.n_cols;
    mat_ptr p_Z;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto &cums_X = ccache.get_cached_cumulatives(params, features_t, time);
            if (cums_X.empty()) LOG4_THROW("Failed preparing cumulatives matrices.");
            p_Z = ptr<arma::mat>(len, len);
            OMP_LOCK(prepare_Z_l)
#pragma omp parallel for num_threads(adj_threads(cums_X.size())) schedule(static, 1)
            for (const auto &c_X: cums_X) {
                arma::mat z(arma::size(*p_Z));
                kernel::path::cu_distances_xx(c_X.n_rows, len, len, len, c_X.mem, params.get_svr_kernel_param2(), z.memptr());
                omp_set_lock(&prepare_Z_l);
                *p_Z += z;
                omp_unset_lock(&prepare_Z_l);
            }
            *p_Z /= double(cums_X.size());
            LOG4_DEBUG("Returning path Z " << arma::size(*p_Z) << " for " << params << ", cumulative matrices " << cums_X.size() << ", of dimensions " << arma::size(cums_X.front()) << ", features " << arma::size(features_t));
            break;
        }
        default:
            const auto kf = IKernel<double>::get(params);
            p_Z = ptr<arma::mat>(len, len);
            kf->operator()(features_t.t(), *p_Z);
            break;
    }
    return p_Z;
}


mat_ptr OnlineMIMOSVR::prepare_Z(const SVRParameters &params, const arma::mat &features_t)
{
    const auto len = features_t.n_cols;
    mat_ptr p_Z;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto p_cums_X = prepare_cumulatives(params, features_t);
            if (p_cums_X->empty()) LOG4_THROW("Failed preparing cumulatives matrices.");
            p_Z = otr<arma::mat>(len, len);
            OMP_LOCK(Z_l)
#pragma omp parallel for num_threads(adj_threads(p_cums_X->size())) schedule(static, 1)
            for (const auto &c_X: *p_cums_X) {
                arma::mat z(arma::size(*p_Z));
                kernel::path::cu_distances_xx(c_X.n_rows, len, len, len, c_X.mem, params.get_svr_kernel_param2(), z.memptr());
                omp_set_lock(&Z_l);
                *p_Z += z;
                omp_unset_lock(&Z_l);
            }
            *p_Z /= double(p_cums_X->size());
            LOG4_DEBUG("Returning path Z " << arma::size(*p_Z) << " for " << params << ", cumulative matrices " << p_cums_X->size() << ", of dimensions " <<
                        arma::size(p_cums_X->front()) << ", features " << arma::size(features_t));
            break;
        }
        default:
            const auto kf = IKernel<double>::get(params);
            p_Z = otr<arma::mat>(len, len);
            kf->operator()(features_t.t(), *p_Z);
            break;
    }
    return p_Z;
}


mat_ptr OnlineMIMOSVR::prepare_Zy(business::calc_cache &ccache, const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &time)
{
    mat_ptr p_Zy;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto &cums_X = ccache.get_cached_cumulatives(params, features_t, time);
            if (cums_X.empty()) LOG4_THROW("Failed preparing feature cumulative matrices.");
            const auto &cums_Xy = ccache.get_cached_cumulatives(params, predict_features_t, time);
            if (cums_Xy.empty()) LOG4_THROW("Failed preparing predict cumulative matrices.");
            OMP_LOCK(add_Zy_l)
            p_Zy = otr<arma::mat>(predict_features_t.n_cols, features_t.n_cols);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(cums_X.size()))
            for (size_t i = 0; i < cums_X.size(); ++i) {
                arma::mat z(arma::size(*p_Zy));
                kernel::path::cu_distances_xy(
                        cums_X.at(i).n_rows, features_t.n_cols, predict_features_t.n_cols, features_t.n_cols, predict_features_t.n_cols,
                        cums_X.at(i).mem, cums_Xy.at(i).mem, params.get_svr_kernel_param2(), z.memptr());
                if (z.has_nonfinite()) LOG4_THROW("Z " << i << " not sane");
                omp_set_lock(&add_Zy_l);
                *p_Zy += z;
                omp_unset_lock(&add_Zy_l);
            }
            *p_Zy /= double(cums_X.size());
            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << ", features t " << arma::size(features_t) << ", predict features t " << arma::size(predict_features_t) <<
                                       " for parameters " << params << ", cumulative matrices " << cums_X.size() << " and " << cums_Xy.size() << ", of dimensions " << arma::size(cums_X.front()));
            break;
        }
        case kernel_type_e::DEEP_PATH:
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }

    return p_Zy;
}


mat_ptr OnlineMIMOSVR::prepare_Zy(const SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t)
{
    mat_ptr p_Zy;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            auto p_cums_X = prepare_cumulatives(params, features_t);
            if (p_cums_X->empty()) LOG4_THROW("Failed preparing feature cumulative matrices.");
            auto p_cums_Xy = prepare_cumulatives(params, predict_features_t);
            if (p_cums_Xy->empty()) LOG4_THROW("Failed preparing predict cumulative matrices.");
            OMP_LOCK(add_Zy_l)
            p_Zy = ptr<arma::mat>(predict_features_t.n_cols, features_t.n_cols);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_cums_X->size()))
            for (size_t i = 0; i < p_cums_X->size(); ++i) {
                arma::mat z(arma::size(*p_Zy));
                kernel::path::cu_distances_xy(
                        p_cums_X->at(i).n_rows, features_t.n_cols, predict_features_t.n_cols, features_t.n_cols, predict_features_t.n_cols,
                        p_cums_X->at(i).mem, p_cums_Xy->at(i).mem, params.get_svr_kernel_param2(), z.memptr());
                p_cums_Xy->at(i).clear();
                if (z.has_nonfinite()) LOG4_THROW("Z " << i << " not sane");
                omp_set_lock(&add_Zy_l);
                *p_Zy += z;
                omp_unset_lock(&add_Zy_l);
            }
            *p_Zy /= double(p_cums_X->size());
            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << ", features t " << arma::size(features_t) << ", predict features t " << arma::size(predict_features_t) <<
                " for parameters " << params << ", cumulative matrices " << p_cums_X->size() << " and " << p_cums_Xy->size() << ", of dimensions " << arma::size(p_cums_X->front()));
            break;
        }
        case kernel_type_e::DEEP_PATH:
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }

    return p_Zy;
}


arma::mat OnlineMIMOSVR::prepare_K(const SVRParameters &params, const arma::mat &x_t, const bpt::ptime &time)
{
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto &Z = ccache().get_cached_Z(params, x_t, time);
            arma::mat K(arma::size(Z));
            solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, params.get_svr_kernel_param());
            return K;
        }
        default: LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    return {};
}

arma::mat OnlineMIMOSVR::prepare_K(const SVRParameters &params, const arma::mat &x_t)
{
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto p_Z = prepare_Z(params, x_t);
            arma::mat K(arma::size(*p_Z));
            solvers::kernel_from_distances(K.memptr(), p_Z->mem, p_Z->n_rows, p_Z->n_cols, params.get_svr_kernel_param());
            return K;
        }
        default: LOG4_THROW("Unhandled kernel " << int(params.get_kernel_type()));
    }
    return {};
}


arma::mat OnlineMIMOSVR::prepare_Ky(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train_t, const arma::mat &x_predict_t, const bpt::ptime &time)
{
    arma::mat Ky(x_predict_t.n_cols, x_train_t.n_cols);
    switch (svr_parameters.get_kernel_type()) {
        case datamodel::kernel_type_e::PATH: {
            mat_ptr p_Zy;
            PROFILE_EXEC_TIME(p_Zy = prepare_Zy(ccache(), svr_parameters, x_train_t, x_predict_t, time), "Prepare Zy, params " << svr_parameters);
            solvers::kernel_from_distances(Ky.memptr(), p_Zy->mem, p_Zy->n_rows, p_Zy->n_cols, svr_parameters.get_svr_kernel_param());
            break;
        }
        default: LOG4_THROW("Unhandled kernel " << int(svr_parameters.get_kernel_type()));
    }
    LOG4_DEBUG("Predict kernel matrix of size " << arma::size(Ky) << ", trained features t " << arma::size(x_train_t) <<
                                                ", predict features t " << arma::size(x_predict_t));
    return Ky;
}


arma::mat OnlineMIMOSVR::prepare_Ky(const datamodel::SVRParameters &svr_parameters, const arma::mat &x_train_t, const arma::mat &x_predict_t)
{
    switch (svr_parameters.get_kernel_type()) {
        case datamodel::kernel_type_e::PATH: {
            arma::mat Ky(x_predict_t.n_cols, x_train_t.n_cols);
            mat_ptr p_Zy;
            PROFILE_EXEC_TIME(p_Zy = prepare_Zy(svr_parameters, x_train_t, x_predict_t), "Prepare Zy, params " << svr_parameters);
            solvers::kernel_from_distances(Ky.memptr(), p_Zy->mem, p_Zy->n_rows, p_Zy->n_cols, svr_parameters.get_svr_kernel_param());
            LOG4_DEBUG("Predict kernel matrix of size " << arma::size(Ky) << ", trained features t " << arma::size(x_train_t) <<
                ", predict features t " << arma::size(x_predict_t));
            return Ky;
        }
        default: LOG4_THROW("Unhandled kernel " << int(svr_parameters.get_kernel_type()));
    }
    return {};
}

} // datamodel
} // svr
