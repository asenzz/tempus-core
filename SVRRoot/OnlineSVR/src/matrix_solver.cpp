//
// Created by zarko on 27/03/2025.
//
#include <semaphore>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <armadillo>
#ifdef USE_PETSC_SOLVER
#include <petscksp.h>
#endif
#include "appcontext.hpp"
#include "matrix_solver.hpp"
#include "common/logging.hpp"
#include "cuqrsolve.cuh"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"


namespace svr {
namespace solvers {
antisymmetric_solver::antisymmetric_solver(
    const Ti m, const Ti n, const Ti iter, const Tv *const x0_, const Tv *const A_, const Tv *const b_, const bool direct_solve,
    const uint16_t irwls_iter) : direct_solve(direct_solve), x0_(x0_), A_(A_), b_(b_), iter(iter), irwls_iter(irwls_iter), m(m), n(n), indices(m)
{
    std::iota(indices.begin(), indices.end(), 0);
}

antisymmetric_solver::~antisymmetric_solver()
{
}

antisymmetric_solver::Tv antisymmetric_solver::operator()(Tv *const sol) const
{
    LOG4_BEGIN();

    Tv rnorm;

    double *x0;
    arma::vec x0_vec;
    if (direct_solve) {
        const arma::Mat<Tv> b_vec((double *) b_, m, n, false, true);
        const arma::Mat<Tv> K((double *) A_, m, m, false, true);
        solvers::solve_irwls(K, b_vec, x0_vec, irwls_iter, PROPS.get_weight_layers());
        x0 = x0_vec.memptr();
        // LOG4_TRACE("Direct solve " << common::present(x0_vec) << ", labels " << common::present(b_vec) << ", K " << common::present(K));
    } else
        x0 = (double *) x0_;

#ifdef USE_PETSC_SOLVER

    assert(n == 1);

    static tbb::mutex petsc_mx;
    tbb::mutex::scoped_lock lk(petsc_mx);
    common::init_petsc(); // PETSc is needed for calc_weights

    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;

    // Initialize PETSc objects
    PetscCallCXXAbort(PETSC_COMM_SELF, MatCreate(PETSC_COMM_WORLD, &A));
    PetscCallCXXAbort(PETSC_COMM_SELF, MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
    // PetscCallCXXAbort(PETSC_COMM_SELF, MatSetType(A, MATSEQAIJ));
    PetscCallCXXAbort(PETSC_COMM_SELF, MatSetFromOptions(A));
    PetscCallCXXAbort(PETSC_COMM_SELF, MatSetUp(A));

    PetscCallCXXAbort(PETSC_COMM_SELF, VecCreate(PETSC_COMM_WORLD, &b));
    // PetscCallCXXAbort(PETSC_COMM_SELF, VecSetType(b, VECSEQ));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecSetSizes(b, PETSC_DECIDE, m));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecSetFromOptions(b));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecSetUp(b));

    PetscCallCXXAbort(PETSC_COMM_SELF, VecDuplicate(b, &x));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecSetValues(b, m, indices.data(), b_, INSERT_VALUES));

    PetscCallCXXAbort(PETSC_COMM_SELF, VecSetUp(x));
    if (x0)
        PetscCallCXXAbort(PETSC_COMM_SELF, VecSetValuesBlocked(x, m, indices.data(), x0, INSERT_VALUES));
    else
        PetscCallCXXAbort(PETSC_COMM_SELF, VecSet(x, 0.));

#if 1
    for (Ti i = 0; i < m; ++i)
        for (Ti j = 0; j < m; ++j)
            PetscCallCXXAbort(PETSC_COMM_SELF, MatSetValue(A, i, j, A_[i + j * m], INSERT_VALUES));
#else
    std::vector<Ti> col(m);
    for (DTYPE(m) i = 0; i < m; ++i) {
        std::ranges::fill(col, i);
        PetscCallCXXAbort(PETSC_COMM_SELF, MatSetValuesBlocked(A, m, indices.data(), m, col.data(), A_ + i * m, INSERT_VALUES));
    }
#endif

    PetscCallCXXAbort(PETSC_COMM_SELF, MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallCXXAbort(PETSC_COMM_SELF, MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecAssemblyBegin(b));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecAssemblyEnd(b));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecAssemblyBegin(x));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecAssemblyEnd(x));

    // Create the linear solver context
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGMRESSetRestart(ksp, std::min<PetscInt>(1e4, iter - 1)));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPGMRES));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGMRESSetCGSRefinementType(ksp, KSP_GMRES_CGS_REFINE_ALWAYS));
    // PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

    // Configure the solver for antisymmetric matrices
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
    PetscCallCXXAbort(PETSC_COMM_SELF, PCJacobiSetFixDiagonal(pc, PETSC_FALSE));
    // PetscCallCXXAbort(PETSC_COMM_SELF, PCJacobiSetUseAbs(pc, PETSC_TRUE));

    // Set more options
    // PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_gmres_preallocate", nullptr));
    // PetscCallCXXAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_pipefgmres_shift", "10"));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, 1e-9, PETSC_DEFAULT, PETSC_DEFAULT, iter));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetFromOptions(ksp));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A, A));

#if 0

    // Monitor convergence by checking the cost function
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPMonitorSet(ksp, [](KSP ksp, Ti it, Tv rnorm, void *ctx) {
        /* antisymmetric_solver *solver = (antisymmetric_solver *) ctx;
        Vec x;
        KSPGetSolution(ksp, &x);
        const double cost = solver->costFunction(x);
         */
        if (it % 100 == 0) LOG4_TRACE("Iteration " << it << " residual norm " << rnorm);
        return 0;
    }, this, nullptr));

#endif

    // Solve the system
    PROFIL3(KSPSolve(ksp, b, x));

    // Get convergence information
    KSPConvergedReason reason;
    Ti iterations;
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &reason));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetResidualNorm(ksp, &rnorm));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &iterations));
    // Vec vsol;
    // PetscCallCXXAbort(PETSC_COMM_SELF, KSPGetSolution(ksp, &vsol));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecGetValues(x, m, indices.data(), sol));

    // Cleanup
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallCXXAbort(PETSC_COMM_SELF, MatDestroy(&A));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&b));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&x));

    common::uninit_petsc();
    lk.release();

    LOG4_DEBUG("Solver " << (reason > 0 ? "converged" : "diverged") << " after " << iterations << " iterations, rnorm " << rnorm);

#else

    // Create executor (use OmpExecutor for CPU, CudaExecutor for GPU if available)
    auto exec = gko::OmpExecutor::create();

    // Convert column-major dense A to Ginkgo's dense format
    auto A_dense = gko::matrix::Dense<Tv>::create(exec, gko::dim<2>{m, m});
    OMP_FOR_(m * m, SSIMD collapse(2))
    for (DTYPE(m) col = 0; col < m; ++col)
        for (DTYPE(m) row = 0; row < m; ++row)
            A_dense->at(row, col) = A_[col * m + row]; // col-major access

    // Convert to CSR (Ginkgo solvers require sparse formats)
    auto A_csr = gko::share(gko::matrix::Csr<Tv>::create(exec));
    A_dense->convert_to(A_csr.get());

    // Create vectors
    auto b = gko::matrix::Dense<Tv>::create(exec, gko::dim<2>{m, n});
    auto x = gko::matrix::Dense<Tv>::create(exec, gko::dim<2>{m, n});

    OMP_FOR_(m * m, SSIMD collapse(2))
    for (DTYPE(m) col = 0; col < n; ++col)
        for (DTYPE(m) row = 0; row < m; ++row) {
            b->at(row, col) = b_[col * m + row];
            x->at(row, col) = x0[col * m + row]; // initial guess
        }

    // Setup Jacobi preconditioner
    /*auto jacobi_factory = gko::preconditioner::Jacobi<Tv>::build()
            .with_max_block_size(32u)
            .on(exec); */

    // Create solver (GMRES here, suitable for nonsymmetric matrices)
    auto solver = gko::solver::CbGmres<Tv>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(iter).on(exec),
                gko::stop::ResidualNorm<Tv>::build()
                    .with_reduction_factor(1e-6)
                    .with_baseline(gko::stop::mode::rhs_norm)
                    .on(exec)
            )
            // .with_preconditioner(std::move(jacobi_factory))
            // .with_ortho_method(gko::solver::gmres::ortho_method::mgs)
            .with_krylov_dim(100)
            .with_storage_precision(gko::solver::cb_gmres::storage_precision::keep)
            .on(exec);
    auto gmres_solver = solver->generate(A_csr);
    gmres_solver->apply(b, x);
    exec->synchronize();
    OMP_FOR_(m * m, SSIMD collapse(2))
    for (DTYPE(m) col = 0; col < n; ++col)
        for (DTYPE(m) row = 0; row < m; ++row)
            sol[row + col * m] = x->at(row, col);

#endif

    const auto tmp = (double *) ALIGNED_ALLOC_(MEM_ALIGN, m * sizeof(double));
    const auto L_mean_mask = common::mean_mask(arma::mat((double *)b, m * n, false, true), PROPS.get_solve_radius() * m);
    const auto score = datamodel::OnlineSVR::score_weights(m, n, PROPS.get_weight_layers(), L_mean_mask.mem, A_, sol, tmp);
    ALIGNED_FREE_(tmp);
    return score;
}
}
}
