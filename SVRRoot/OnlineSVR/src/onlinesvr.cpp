#include <armadillo>
#include <mkl_lapacke.h>

#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "onlinesvr.hpp"
#include "onlinesvr_persist.hpp"
#include "mat_solve_gpu.hpp"
#include "cuqrsolve.hpp"
#include "cuda_path.hpp"
#include "kernel_factory.hpp"

#ifdef EXPERIMENTAL_FEATURES
#include <matplotlibcpp.h>
#endif

#define GU_PART_TRAIN       (1. - VALIDATION_SIZE)
#define TESTING_SIZE        (CHUNK_SIZE * VALIDATION_SIZE)


namespace svr {

using namespace arma;


OnlineMIMOSVR::OnlineMIMOSVR(
        const SVRParameters_ptr &p_svr_parameters_, const MimoType type, const size_t multistep_len_) :
        p_svr_parameters(p_svr_parameters_), mimo_type(type), multistep_len(multistep_len_)
{
    init_mimo_base(0 /*std::numeric_limits<double>::min()*/, type);
}


OnlineMIMOSVR::OnlineMIMOSVR(
        const SVRParameters_ptr &p_svr_parameters_,
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const bool update_r_matrix,
        const matrices_ptr &kernel_matrices,
        const bool pseudo_online,
        const MimoType mimo_type_,
        const size_t multistep_len_) :
        p_svr_parameters(p_svr_parameters_), mimo_type(mimo_type_), multistep_len(multistep_len_)
{
    init_mimo_base(0, mimo_type_);

    PROFILE_EXEC_TIME(
            batch_train(p_xtrain, p_ytrain, update_r_matrix, kernel_matrices, pseudo_online),
                "Batch SVM train on " << arma::size(*p_ytrain) << " labels and " << arma::size(*p_xtrain) << " features, parameters " << p_svr_parameters->to_string());
}


void OnlineMIMOSVR::init_mimo_base(const double &epsilon_, const MimoType &mimo_type_)
{
    if (mimo_type == MimoType::single) {
        LOG4_DEBUG("Create model of type single");
        main_components[0].epsilon = 0.;
    } else if (mimo_type == MimoType::twin) {
        LOG4_DEBUG("Create model of type twin");
        main_components[0].epsilon = epsilon_;
        main_components[1].epsilon = -epsilon_;
        if (epsilon_ < std::numeric_limits<double>::epsilon()) LOG4_ERROR("Epsilon is zero and twin type SVM selected!");
    } else {
        LOG4_ERROR("Unsupported MIMO type was selected. Creating as single.");
        main_components[0].epsilon = 0.;
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
    return main_components[type].chunk_weights[0];
}


arma::mat OnlineMIMOSVR::get_weights(uint8_t type) const
{
    return main_components.at(type).chunk_weights[0];
}


void
OnlineMIMOSVR::init_kernel_matrix(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train,
        arma::mat &kernel_matrix,
        const OnlineMIMOSVR_ptr &p_manifold)
{
    switch (svr_parameters.get_kernel_type()) {
        case kernel_type_e::RBF_EXPONENTIAL: {
            auto kernel = IKernel<double>::get_kernel(svr_parameters.get_kernel_type(), svr_parameters);
            kernel->operator()(x_train, kernel_matrix);
            return;
        }

        case kernel_type_e::DEEP_PATH2:
        case kernel_type_e::DEEP_PATH: {
            prepare_manifold_kernel(x_train, y_train, kernel_matrix, p_manifold);
            return;
        }

        default: {
            // TODO Not working correctly test first!
            //const arma::mat &Z = get_cached_Z(svr_parameters, x_train.t(), x_train.n_rows);
            //kernel_matrix.set_size(arma::size(Z));
            //solvers::kernel_from_distances(kernel_matrix.memptr(), Z.memptr(), Z.n_rows, svr_parameters.get_svr_kernel_param());
            kernel_matrix = 1. - get_cached_Z(svr_parameters, x_train.t(), y_train, x_train.n_rows) / (2. * std::pow(svr_parameters.get_svr_kernel_param(), 2.));
        }
    }
}
// k = (1. - get_cached_Z(svr_parameters, x_train.t(), x_train.n_rows) / (2. * gamma^2)

void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved);

// Unstable avoid!
arma::mat OnlineMIMOSVR::do_ocl_solve(const double *host_a, double *host_b, const int m, const int nrhs)
{
    if (m * (m + nrhs) * sizeof(double) > svr::common::gpu_handler::get_instance().get_max_gpu_data_chunk_size())
        THROW_EX_FS(std::runtime_error,
                "Not enough memory on GPU, required " << m * nrhs * sizeof(double) << " bytes, available " <<
                     svr::common::gpu_handler::get_instance().get_max_gpu_data_chunk_size() << " bytes.");

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
}


arma::mat OnlineMIMOSVR::call_gpu_oversolve(const arma::mat &Left, const arma::mat &Right)
{
    // assert(arma::size(Left, 0) == arma::size(Right, 0));

    const size_t Nrows = arma::size(Left,0);
    const size_t Ncols = arma::size(Left,1);
    const size_t Nrhs = arma::size(Right,1);

    arma::mat output = arma::zeros(Ncols, Nrhs);
    svr::solvers::call_gpu_overdetermined(Nrows, Ncols, Nrhs, Left.memptr(), Right.memptr(), output.memptr());

    return output;
}


arma::mat OnlineMIMOSVR::call_gpu_dynsolve(const arma::mat &left, const arma::mat &right)
{
    arma::mat output(left.n_rows, right.n_cols);
#pragma omp parallel for schedule(static, 1) num_threads(right.n_cols)
    for (size_t i = 0; i < right.n_cols; ++i)
        /*PROFILE_EXEC_TIME(*/svr::solvers::dyn_gpu_solve(left.n_rows, left.memptr(), right.colptr(i), output.colptr(i))/*, "Dynamic GPU solve")*/;
    return output;
}


void OnlineMIMOSVR::call_gpu_dynsolve(const arma::mat &left, const arma::mat &right, arma::Mat<double> &output)
{
    if (output.n_rows != left.n_rows || output.n_cols != right.n_cols) output.set_size(left.n_rows, right.n_cols);
#pragma omp parallel for schedule(static, 1) num_threads(right.n_cols)
    for (size_t i = 0; i < right.n_cols; ++i)
        PROFILE_EXEC_TIME(svr::solvers::dyn_gpu_solve(left.n_rows, left.memptr(), right.colptr(i), output.colptr(i)), "Dynamic GPU solve");
}


void OnlineMIMOSVR::call_gpu_dynsolve(const arma::mat &left, const arma::mat &right, arma::subview<double> output)
{
#pragma omp parallel for schedule(static, 1) num_threads(right.n_cols)
    for (size_t i = 0; i < right.n_cols; ++i)
        PROFILE_EXEC_TIME(svr::solvers::dyn_gpu_solve(left.n_rows, left.memptr(), right.colptr(i), output.colptr(i)), "Dynamic GPU solve");
}

// Best
void OnlineMIMOSVR::solve_irwls(const arma::mat &epsilon_eye_K, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters)
{
#if defined(SINGLE_SOLVE)
    return call_gpu_dynsolve(K + epsilon_eye_K, rhs);
#else
    if (solved.empty()) call_gpu_dynsolve(K + epsilon_eye_K, rhs, solved);
    arma::mat best_solution = solved;
    size_t best_it = std::numeric_limits<size_t>::max();
    double best_sae = std::numeric_limits<double>::max();
    for (size_t i = 0; i < iters; ++i) {
        arma::mat error_mat(arma::size(solved));
        double sae = 0;
#pragma omp parallel for reduction(+:sae) schedule(static, 1)
        for (size_t col_ix = 0; col_ix < solved.n_cols; ++col_ix) {
            error_mat.col(col_ix) = arma::abs(K * solved.col(col_ix) - rhs.col(col_ix));
            const double this_sae = common::sum<double>(error_mat.col(col_ix));
            sae += this_sae;
        }
        if (sae < best_sae) {
            best_sae = sae;
            best_solution = solved;
            best_it = i;
        }
        LOG4_TRACE("IRWLS iteration " << i << ", SAE " << sae << ", kernel dimensions " << arma::size(K) << ", best SAE " << best_sae);
        // const arma::mat mult = error_mat / double(K.n_rows);
        const arma::mat mult = 1. / arma::sqrt(error_mat + common::C_singlesolve_delta / ((i + 1.) * IRWLS_ITER / double(iters)));
#pragma omp parallel for schedule(static, 1) num_threads(solved.n_cols)
        for (size_t col_ix = 0; col_ix < solved.n_cols; ++col_ix)
            call_gpu_dynsolve((mult.col(col_ix) * arma::ones(1, K.n_cols)) % K + epsilon_eye_K, rhs.col(col_ix) % mult.col(col_ix), solved.col(col_ix));
    }
    const double sae = common::sumabs<double>(K * solved - rhs);
    if (sae < best_sae) {
        best_sae = sae;
        best_solution = solved;
        best_it = iters - 1;
    }
    LOG4_DEBUG("IRWLS best SAE " << best_sae << ", best iteration " << best_it << ", kernel dimensions " << arma::size(K) << ", delta " << common::C_singlesolve_delta);
    solved = best_solution;
#endif
}

#if 0
// ?!?! Emo's oddball solver :)
arma::mat OnlineMIMOSVR::arma_multisolve(const arma::mat &epsilon_eye_K, const arma::mat &Z, const arma::mat &rhs)
{
#if 0
    const size_t Nrows = arma::size(Z, 1);
    const size_t multinum = 16; // or 4
    arma::mat x;
    1. / double(multinum) * solve_irwls(epsilon_eye_K, Z, rhs, x);
    for (size_t i = 0; i < multinum - 1; ++i) {
        arma::mat ZZ = Z.rows(i + 1, Nrows - 1);
        arma::mat y = solve_irwls(epsilon_eye_K, ZZ.cols(i + 1, Nrows - 1), rhs.rows(i + 1, Nrows - 1));
        x.rows(i + 1, Nrows - 1) += y * 1. / double(multinum);
    }
    return x;
#else
    LOG4_THROW("Do not use!");
    return {};
#endif
}
#endif

arma::mat OnlineMIMOSVR::direct_solve(const arma::mat &a, const arma::mat &b)
{
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));
    arma::mat solved = arma::zeros(arma::size(b));
    PROFILE_EXEC_TIME(svr::solvers::qrsolve(a.n_rows, b.n_cols, a.memptr(), b.memptr(), solved.memptr()), "qrsolve");
    return solved;
}


void
OnlineMIMOSVR::solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, arma::mat &solved, const size_t iters)
{
    if (a.n_rows != b.n_rows) LOG4_THROW("Incorrect sizes a " << arma::size(a) << ", b " << arma::size(b));

    PROFILE_EXEC_TIME(solve_irwls(epsilon_eye_K, a, b, solved, iters), "IRWLS solver " << arma::size(a) << ", " << arma::size(b)); // Emo's dynamic (slowest) IRWLS solver
    // PROFILE_EXEC_TIME(call_gpu_oversolve(a, b, solved), "call_gpu_oversolve " << arma::size(a) << ", " << arma::size(b)); // Emo's solver, best
    // PROFILE_EXEC_TIME(call_gpu_dynsolve(a, b, solved), "call_gpu_oversolve " << arma::size(a) << ", " << arma::size(b)); // call_gpu_dynsolve
    // PROFILE_EXEC_TIME(svr::solvers::qrsolve(a.n_rows, b.n_cols, a.memptr(), b.memptr(), solved.memptr()), "qrsolve"); // direct CUDA solver, fastest, least precise
    // PROFILE_EXEC_TIME(do_mkl_solve(a, b, solved), "lock1 without wait");
}


arma::mat
OnlineMIMOSVR::solve_dispatch(const arma::mat &epsilon_eye_K, const arma::mat &a, const arma::mat &b, const size_t iters)
{
    arma::mat result;
    solve_dispatch(epsilon_eye_K, a, b, result, iters);
    return result;
}


double OnlineMIMOSVR::go_for_variable_gu(const arma::mat &kernel, const arma::mat &rhs, double &gu, const bool fine_search)
{
    LOG4_BEGIN();

    const auto avg_abs_label = common::meanabs(rhs);
    const size_t total_size = rhs.n_rows;
    const size_t train_len = GU_PART_TRAIN * total_size;
    const auto S1 = arma::span(0, train_len - 1);
    const auto S2 = arma::span(train_len, total_size - 1);
    const auto S3 = arma::span(total_size - train_len, total_size - 1);
    const auto S4 = arma::span(0, total_size - train_len - 1);
    const arma::mat partial_kernel = kernel(S1, S1);
    const arma::mat partial_kernel2 = kernel(S3, S3);
    const arma::mat partial_rhs = rhs.rows(S1);
    const arma::mat partial_rhs2 = rhs.rows(S3);
    const arma::mat test_rhs = rhs.rows(S2);
    const arma::mat test_rhs2 = rhs.rows(S4);
    const arma::mat partial_eye = arma::eye(arma::size(partial_kernel));
    const arma::mat partial_eye2 = arma::eye(arma::size(partial_kernel));
    const arma::mat validation_kernel = kernel(S2, S1);
    const arma::mat validation_kernel2 = kernel(S4, S3);
    auto mae = std::numeric_limits<double>::max();
    LOG4_DEBUG("Validation size " << total_size - train_len << ", training size " << train_len);

    std::mutex mae_mx;
    std::vector<double> gus;
    auto validate_gu = [&](const double new_test_gu) {
        const double mae_new =
                common::meanabs<double>(validation_kernel * OnlineMIMOSVR::call_gpu_dynsolve(new_test_gu * partial_eye + partial_kernel, partial_rhs) - test_rhs) / avg_abs_label
                + common::meanabs<double>(validation_kernel2 * OnlineMIMOSVR::call_gpu_dynsolve(new_test_gu * partial_eye2 + partial_kernel2, partial_rhs2) - test_rhs2) / avg_abs_label;

        LOG4_TRACE("MAE " << mae_new << ", test GU " << new_test_gu << ", GU " << gu);
#pragma omp critical(validate_gu)
        {
            if (mae_new < mae || isnan(mae)) {
                gu = new_test_gu;
                mae = mae_new;
            }
        }
    };

    for (const auto test_gu: common::C_tune_crass_epscost)
        validate_gu(test_gu);
#if 0
    if (fine_search)
        for (double test_gu = (gu / MULTIPLE_EPSCO) / MULTIPLE_EPSCO; test_gu < (gu * MULTIPLE_EPSCO) * MULTIPLE_EPSCO; test_gu *= MULTIPLE_EPSCO_FINE)
            validate_gu(test_gu);
#endif
    LOG4_DEBUG("Final gu " << gu << ", MAE " << mae);

    return mae;
}


void OnlineMIMOSVR::do_over_train_zero_epsilon(const arma::mat &xx_train, const arma::mat &yy_train, const size_t chunk_idx)
{
    const arma::mat x_train = xx_train.rows(ixs[chunk_idx]);
    const arma::mat y_train = yy_train.rows(ixs[chunk_idx]);
    if (p_kernel_matrices->at(chunk_idx).empty()) {
        PROFILE_EXEC_TIME(init_kernel_matrix(*p_svr_parameters, x_train, y_train, p_kernel_matrices->at(chunk_idx), p_manifold), "Init kernel matrix");
    } else
        LOG4_DEBUG("Using pre-calculated kernel matrix for " << chunk_idx);

    if (!std::isnormal(auto_gu[chunk_idx])) {
        PROFILE_EXEC_TIME(go_for_variable_gu(p_kernel_matrices->at(chunk_idx), y_train, auto_gu[chunk_idx]), "go_for_variable_gu");
        p_svr_parameters->set_svr_C(1. / (2. * auto_gu[chunk_idx]));
    }
    for (auto &kv: main_components) {
        auto &component = kv.second;
        // The regression parameters
        component.chunk_weights[chunk_idx] = arma::ones(y_train.n_rows, y_train.n_cols);
        // Prediction error
        calc_weights(chunk_idx, y_train, 1. / (2. * auto_gu[chunk_idx]), component, IRWLS_ITER);
    }
}


void OnlineMIMOSVR::calc_weights(const size_t chunk_ix, const mat &y_train, const double C, OnlineMIMOSVR::MimoBase &component, const size_t iters)
{
    arma::mat E(arma::size(y_train));
#pragma omp parallel for schedule(static, 1) num_threads(std::min<size_t>(std::thread::hardware_concurrency(), y_train.n_cols))
    for (size_t col_ix = 0; col_ix < y_train.n_cols; ++col_ix)
        E.col(col_ix) = y_train.col(col_ix) + component.epsilon - p_kernel_matrices->at(chunk_ix) * component.chunk_weights[chunk_ix].col(col_ix);
    component.mae_chunk_values[chunk_ix] = ixs.size() < 2 ? 1 : common::meanabs<double>(E);
    LOG4_DEBUG("Initial MAE " << component.mae_chunk_values[chunk_ix] << " for chunk " << chunk_ix);
#if 0
    // RMSE with no weights
    arma::mat L = zeros(size(u, 0), 1);
    arma::mat Lp = zeros(2, 1);
    // optimize outside epsilon
    L = pow(u, 2);
    // minimize
    Lp(0, 0) = C * accu(L) / 2.;
    const arma::mat Beta_a = component.chunk_weights[chunk_ix];
    //E_a = E;
    const arma::mat u_a = u;
#endif
    //arma::cube eye_K_epscost(p_kernel_matrices->at(chunk_ix).n_rows, p_kernel_matrices->at(chunk_ix).n_cols, 1);
    solve_dispatch(eye(size(p_kernel_matrices->at(chunk_ix))) * 1. / (2. * C), p_kernel_matrices->at(chunk_ix), y_train, component.chunk_weights[chunk_ix], iters);
#pragma omp parallel for schedule(static, 1) num_threads(y_train.n_cols)
    for (size_t col_ix = 0; col_ix < y_train.n_cols; ++col_ix)
        E.col(col_ix) = y_train.col(col_ix) + component.epsilon - p_kernel_matrices->at(chunk_ix) * component.chunk_weights[chunk_ix].col(col_ix);
    // E = y_train + component.epsilon - p_kernel_matrices->at(chunk_ix) * component.chunk_weights[chunk_ix];
    component.mae_chunk_values[chunk_ix] = ixs.size() < 2 ? 1 : common::meanabs<double>(E);
    LOG4_DEBUG("Final MAE " << component.mae_chunk_values[chunk_ix] << " for chunk " << chunk_ix);
}


std::vector<arma::uvec> OnlineMIMOSVR::get_indexes() const
{
    return OnlineMIMOSVR::get_indexes(p_features->empty() ? p_labels->n_rows : p_features->n_rows, *p_svr_parameters);
}


std::vector<arma::uvec>
OnlineMIMOSVR::get_indexes(const size_t n_rows_dataset, const SVRParameters &svr_parameters)
{
    const auto n_rows_train = std::min<size_t>(n_rows_dataset, svr_parameters.get_svr_decremental_distance());
    const auto num_chunks = get_chunk_num(n_rows_train);
    std::vector<arma::uvec> indexes(num_chunks);
    if (num_chunks == 1) {
        indexes[0] = arma::linspace<arma::uvec>(0, n_rows_train - 1, n_rows_train);
        LOG4_DEBUG("Linear single chunk indexes size " << arma::size(indexes[0]));
        return indexes;
    }

    const size_t chunk_size = n_rows_train / num_chunks;
    LOG4_DEBUG("Num rows is " << n_rows_dataset << ", decrement distance " << svr_parameters.get_svr_decremental_distance() <<
                              ", max indexes len " << n_rows_train << ", num chunks " << num_chunks << ", chunk size " << chunk_size);
    const size_t last_part_size = chunk_size * .1;
    const size_t first_part_size = chunk_size - last_part_size;
    const arma::mat i_full = arma::cumsum(arma::ones(n_rows_train, 1)) - 1;
    const arma::mat first_part = i_full.rows(0, n_rows_train - last_part_size);
    const arma::mat last_part = i_full.rows(n_rows_train - last_part_size + 1, n_rows_train - 1);
    const arma::mat i_shuffle = svr::common::armd::fixed_shuffle(first_part);
    __tbb_pfor_i(0, num_chunks,
        const arma::mat m_11 = i_shuffle.rows(i * first_part_size, std::min((i + 1) * first_part_size - 1, n_rows_train - last_part_size - 1));
        indexes[i] = arma::conv_to<arma::uvec>::from(arma::join_cols(m_11, last_part));
    )

    return indexes;
}

arma::uvec
OnlineMIMOSVR::get_other_ixs(const size_t i) const
{
    return arma::find(ixs[i] != arma::linspace<arma::uvec>(0, p_features->n_rows - 1, p_features->n_rows));
}


matrices_ptr
OnlineMIMOSVR::produce_kernel_matrices(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train,
        const OnlineMIMOSVR_ptr &p_manifold)
{
    //viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);
    const auto indexes = get_indexes(x_train.n_rows, svr_parameters);
    auto p_res_kernel_matrices = std::make_shared<std::vector<arma::mat>>(indexes.size());
    __tbb_pfor_i (
        0, indexes.size(),
        PROFILE_EXEC_TIME(init_kernel_matrix(svr_parameters, x_train.rows(indexes[i]), y_train.rows(indexes[i]), p_res_kernel_matrices->at(i), p_manifold),
        "Init kernel matrix");
    )
    return p_res_kernel_matrices;
}

std::vector<arma::mat>
OnlineMIMOSVR::produce_labels(const arma::mat &y_train, const SVRParameters &svr_parameters)
{
    const auto indexes = get_indexes(y_train.n_rows, svr_parameters);
    std::vector<arma::mat> labels(indexes.size());
    __tbb_pfor_i(0, indexes.size(), labels[i] = y_train.rows(indexes[i]) )
    return labels;
}


void
OnlineMIMOSVR::batch_train(
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const bool update_r_matrix,
        const matrices_ptr &kernel_matrices,
        const bool pseudo_online)
{
    LOG4_DEBUG(
            "Training on X " << common::present(*p_xtrain) << ", Y " << common::present(*p_ytrain) << ", update R matrix " << update_r_matrix << ", pre-calculated kernel matrices " << (kernel_matrices ? kernel_matrices->size() : 0) <<
                             ", pseudo online " << pseudo_online << ", parameters " << p_svr_parameters->to_string());
    const std::scoped_lock learn_lock(learn_mx);

#ifdef MANIFOLD_TEST_DEBUG
    if (!p_svr_parameters->get_decon_level() || p_svr_parameters->get_decon_level() == 28 || p_svr_parameters->get_decon_level() == 2) {
        p_xtrain->save(common::formatter() << "/tmp/xtrain_" << p_svr_parameters->get_decon_level() << ".csv", arma::csv_ascii);
        p_ytrain->save(common::formatter() << "/tmp/ytrain_" << p_svr_parameters->get_decon_level() << ".csv", arma::csv_ascii);
#ifdef EXPERIMENTAL_FEATURES
        matplotlibcpp::plot(std::vector<double>{(double *)p_ytrain->memptr(), (double *)p_ytrain->memptr() + p_ytrain->n_elem}, {{"label", "Level 0 labels"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
        matplotlibcpp::cla();
        matplotlibcpp::clf();
        matplotlibcpp::close();
        matplotlibcpp::plot(std::vector<double>{(double *)p_xtrain->memptr(), (double *)p_xtrain->memptr() + p_ytrain->n_cols}, {{"label", "Level 0 first features"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
        matplotlibcpp::cla();
        matplotlibcpp::clf();
        matplotlibcpp::close();
        matplotlibcpp::plot(std::vector<double>{(double *)p_xtrain->memptr() + p_xtrain->n_elem - p_ytrain->n_cols, (double *)p_xtrain->memptr() + p_xtrain->n_elem}, {{"label", "Level 0 last features"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
#endif
    }
#endif

    reset_model(pseudo_online);

    if (p_svr_parameters->get_svr_kernel_param() == 0) {
        if (kernel_matrices) {
            LOG4_ERROR("Kernel matrices provided will be cleared because SVR kernel parameters are not initialized!");
            kernel_matrices->clear();
        }
//        PROFILE_EXEC_TIME(tune_kernel_params(p_svr_parameters, *p_xtrain, *p_ytrain), "Tune kernel params for model " << p_svr_parameters->get_decon_level());
    }
    if (p_xtrain->n_rows > p_svr_parameters->get_svr_decremental_distance())
        p_xtrain->shed_rows(0, p_xtrain->n_rows - p_svr_parameters->get_svr_decremental_distance() - 1);
    if (p_ytrain->n_rows > p_svr_parameters->get_svr_decremental_distance())
        p_ytrain->shed_rows(0, p_ytrain->n_rows - p_svr_parameters->get_svr_decremental_distance() - 1);

    if (not pseudo_online) {
        p_features = p_xtrain;
        p_labels = p_ytrain;
        if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty() || !p_xtrain->n_cols || !p_ytrain->n_cols)
            LOG4_THROW("Invalid data dimensions X train " << arma::size(*p_xtrain) << ", Y train " << arma::size(*p_ytrain) << ", level " << p_svr_parameters->get_decon_level());
    }

    if (!p_manifold) init_manifold(*p_xtrain, *p_ytrain);

    if (ixs.empty()) ixs = get_indexes();
    if (kernel_matrices && kernel_matrices->size() == ixs.size()) {
        p_kernel_matrices = kernel_matrices;
        LOG4_DEBUG("Using " << ixs.size() << " pre-calculated matrices.");
    } else if (kernel_matrices && !kernel_matrices->empty()) {
        LOG4_ERROR("External kernel matrices do not match needed chunks count!");
    } else {
        LOG4_DEBUG("Initializing kernel matrices from scratch.");
    }

    for (auto &kv : main_components) {
        auto &component = kv.second;
        // the regression parameters
        //component.total_weights = arma::zeros(n_m, n_k);
        if (component.chunk_weights.size() != ixs.size()) component.chunk_weights.resize(ixs.size());
        if (component.mae_chunk_values.size() != ixs.size()) component.mae_chunk_values.resize(ixs.size(), 0.);
    }

    if (update_r_matrix && r_matrix.size() != ixs.size()) r_matrix.resize(ixs.size());
    if (p_kernel_matrices->size() != ixs.size()) p_kernel_matrices->resize(ixs.size());
    if (auto_gu.size() != ixs.size()) auto_gu.resize(ixs.size(), 1. / (2. * p_svr_parameters->get_svr_C()));

    __tbb_pfor_i(0, ixs.size(), do_over_train_zero_epsilon(*p_features, *p_labels, i)) // do_over_train(*p_features, *p_labels, i)
    update_total_weights();
    samples_trained = p_features->n_rows;

    if (update_r_matrix) PROFILE_EXEC_TIME(init_r_matrix(), "init_r_matrix");

#ifdef PSEUDO_ONLINE // Clear kernel matrices once weight is calculated
    p_kernel_matrices = std::make_shared<std::vector<arma::mat>>(ixs.size());
#endif
}

void OnlineMIMOSVR::add_to_r_matrix(const size_t inner_ix, const size_t outer_ix, const double correction)
{
    const size_t last_elem = p_kernel_matrices->at(outer_ix).n_rows - 1;
    arma::uvec others = arma::linspace<arma::uvec>(0, last_elem, p_kernel_matrices->at(outer_ix).n_rows);
    others.shed_row(inner_ix);
    arma::uvec element(1);
    element(0) = inner_ix;
    arma::mat a11 = p_kernel_matrices->at(outer_ix).submat(others, others) + arma::eye(last_elem, last_elem) * correction;
    arma::mat a12 = p_kernel_matrices->at(outer_ix).submat(others, element);
    arma::mat a21 = p_kernel_matrices->at(outer_ix).submat(element, others);

    const double a22 = p_kernel_matrices->at(outer_ix)(inner_ix, inner_ix) + correction;
    const arma::mat f22 = a22 - a21 * r_matrix[outer_ix] * a12;
    const arma::mat r_f22 = 1. / f22;
    const arma::mat prod_a11m1_a12 = r_matrix[outer_ix] * a12;
    const arma::mat prod_a21_a11m1 = a21 * r_matrix[outer_ix];
    r_matrix[outer_ix] = r_matrix[outer_ix] + prod_a11m1_a12 * r_f22 * prod_a21_a11m1;
    const arma::mat new_a12 = -(r_matrix[outer_ix] * a12) / a22;
    arma::mat new_a21 = -(a21 * r_matrix[outer_ix]) / a22;
    r_matrix[outer_ix].insert_cols(inner_ix, new_a12);
    new_a21.insert_cols(inner_ix, r_f22);
    r_matrix[outer_ix].insert_rows(inner_ix, new_a21);
}


void OnlineMIMOSVR::remove_from_r_matrix(const size_t inner_ix, const size_t outer_ix)
{
    std::scoped_lock l(r_mx);
    arma::uvec rows_and_cols = linspace<uvec>(0, r_matrix[outer_ix].n_rows - 1, r_matrix[outer_ix].n_rows);
    rows_and_cols.shed_row(inner_ix);
    arma::mat sub_matrix = r_matrix[outer_ix].submat(rows_and_cols, rows_and_cols);
    arma::mat no_row = r_matrix[outer_ix].rows(rows_and_cols);
    arma::mat no_col = r_matrix[outer_ix].cols(rows_and_cols);
    r_matrix[outer_ix] = sub_matrix - no_row.col(inner_ix) * (no_col.row(inner_ix) / r_matrix[outer_ix](inner_ix, inner_ix));
}


void mkl_dgesv(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    MKL_INT n = a.n_rows;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = n;
    MKL_INT ldb = n;
    solved = b;
    std::vector<double> a_(a.n_elem);
    std::memcpy(a_.data(), a.memptr(), sizeof(double) * a.n_elem);
    std::vector<MKL_INT> ipiv(a.n_rows);
    MKL_INT info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a_.data(), lda, ipiv.data(), const_cast<double *>(solved.memptr()), ldb);
    if (info > 0) LOG4_THROW("Something is wrong in solving LAPACKE_dgesv, error is  " << info << ". The solution could not be computed.");
}


void do_mkl_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    MKL_INT n = a.n_rows;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = n;
    MKL_INT ldb = n;
    solved = b;
//    static std::mutex solve_mtx;
//    std::scoped_lock lock(solve_mtx);
//    mkl_set_num_threads_local(40);
    MKL_INT info = LAPACKE_dposv(
            LAPACK_COL_MAJOR, 'U', n, nrhs, const_cast<double *>(a.memptr()), lda, solved.memptr(), ldb);
    //if (info != 0) LOG4_THROW("DPOSV failed, the leading minor of order " << info << " is not positive definite.");
    if (info == 0) return;
    LOG4_WARN("DPOSV failed, the leading minor of order " << info << " is not positive definite, trying DGESV . . .");
    mkl_dgesv(a, b, solved);
}


void do_mkl_over_solve(const arma::mat &a, const arma::mat &b, arma::mat &solved)
{
    static std::mutex solve_mtx;
    std::scoped_lock l(solve_mtx);
    MKL_INT m = a.n_rows;
    MKL_INT n = a.n_cols;
    MKL_INT nrhs = b.n_cols;
    MKL_INT lda = m;
    MKL_INT ldb = m;
    std::vector<double> a_(a.n_elem);
    std::vector<double> b_(b.n_elem);
    std::memcpy(a_.data(), a.memptr(), sizeof(double) * a.n_elem);
    std::memcpy(b_.data(), b.memptr(), sizeof(double) * b.n_elem);
    solved = zeros(n, nrhs);
//extern lapack_int LAPACKE_dgels(int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb);

    lapack_int info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, a_.data(), lda, b_.data(), ldb);
    if (info > 0) LOG4_THROW("LAPACKE_dgels failed with " << info);
    arma::mat x(b_);
    x.reshape(b.n_rows, b.n_cols);
    solved = x.rows(0, n - 1);
}


void OnlineMIMOSVR::init_r_matrix()
{
    r_matrix.resize(p_kernel_matrices->size());
    //const auto cost_factor = 1. / (2. * p_svr_parameters->get_svr_C());
    __omp_pfor_i(0, p_kernel_matrices->size(),
                 const arma::mat singular = arma::eye(p_kernel_matrices->at(i).n_rows, p_kernel_matrices->at(i).n_cols);
        //solve_dispatch(cost_factor, p_kernel_matrices->at(i) + cost_factor * singular, singular, r_matrix[i]);
    )
}


void OnlineMIMOSVR::update_total_weights()
{
    for (auto &kv: main_components) {
        kv.second.total_weights = zeros(size(*p_labels));
#if 1
        arma::mat divisors = zeros(p_labels->n_rows, 1);
        for (size_t ix = 0; ix < ixs.size(); ++ix) {
            divisors.rows(ixs[ix]) += 1;
            kv.second.total_weights.rows(ixs[ix]) += kv.second.chunk_weights[ix];
        }
        kv.second.total_weights /= divisors;
#else
        for (size_t ix = 0; ix < ixs.size(); ++ix)
            kv.second.total_weights.rows(ixs[ix]) += kv.second.chunk_weights[ix] / double(ixs.size());
#endif
        LOG4_TRACE("Weights for level " << p_svr_parameters->get_decon_level() << " " << arma::max(arma::sum(kv.second.total_weights, 1)));
    }
}

//Assuming objects are equal if sizes of the containers are the same
bool OnlineMIMOSVR::operator==(OnlineMIMOSVR const &other) const
{
    if (p_features->n_elem != other.p_features->n_elem
        || p_labels->n_elem != other.p_labels->n_elem
        || *p_svr_parameters != *other.p_svr_parameters
        || mimo_type != other.mimo_type
        || main_components.size() != other.main_components.size()
        || ixs.size() != other.ixs.size())
        return false;
    for (size_t i = 0; i < ixs.size(); ++i) {
        const auto &ith_vec = ixs[i] == other.ixs[i];
        if (any(ith_vec < 1)) return false;
    }
    return true;
}

OnlineMIMOSVR::OnlineMIMOSVR(std::stringstream &input_stream) : mimo_type(MimoType::single), multistep_len(1) {}

void OnlineMIMOSVR::reset_model(const bool pseudo_online)
{
    samples_trained = 0;
    if (not pseudo_online) {
        p_features = std::make_shared<arma::mat>();
        p_labels = std::make_shared<arma::mat>();
        auto_gu.clear();
    }
    p_kernel_matrices = std::make_shared<std::vector<arma::mat>>();
    r_matrix.clear();
    for (auto &component : main_components) {
        component.second.chunk_weights.clear();
        component.second.total_weights.clear();
        component.second.mae_chunk_values.clear();
    }
    ixs.clear();
}


size_t OnlineMIMOSVR::get_chunk_num(const size_t decrement)
{
    return std::max<size_t>(1, std::ceil(double(decrement) / double(CHUNK_SIZE)));
}


} // svr
