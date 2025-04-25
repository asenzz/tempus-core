//
// Created by zarko on 27/03/2025.
//
#include <cmath>
#include <iostream>
#include <algorithm>
#include <armadillo>
#include "appcontext.hpp"
#include "matrix_solver.hpp"

namespace svr {
namespace solvers {

antisymmetric_solver::antisymmetric_solver(
        const PetscInt m, const PetscInt iter, const PetscScalar *const x0_, const PetscScalar *const A_, const PetscScalar *const b_, const bool direct_solve,
        const uint16_t irwls_iter) : m(m), indices(m) // , p_b(b_), p_A(A_)
{
    LOG4_BEGIN();
    double *x0;
    arma::vec x0_vec;
    if (direct_solve) {
        const arma::vec b_vec((double *) b_, m, false, true);
        // const arma::Mat<PetscScalar> K((double *) A_, m, m, false, true);
        // solvers::solve_irwls(K, K, b_vec, x0_vec, irwls_iter);
        x0_vec.set_size(m);
        x0_vec.ones();
        x0 = x0_vec.memptr();
        // LOG4_TRACE("Direct solve " << common::present(x0_vec) << ", labels " << common::present(b_vec) << ", K " << common::present(K));
    } else
        x0 = (double *) x0_;

    std::iota(indices.begin(), indices.end(), 0);

    // Initialize PETSc objects
    PetscCallAbort(PETSC_COMM_SELF, MatCreate(PETSC_COMM_WORLD, &A));
    PetscCallAbort(PETSC_COMM_SELF, MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
    PetscCallAbort(PETSC_COMM_SELF, MatSetType(A, MATDENSE));
    PetscCallAbort(PETSC_COMM_SELF, MatSetFromOptions(A));
    PetscCallAbort(PETSC_COMM_SELF, MatSetUp(A));

    PetscCallAbort(PETSC_COMM_SELF, VecCreate(PETSC_COMM_WORLD, &b));
    PetscCallAbort(PETSC_COMM_SELF, VecSetSizes(b, PETSC_DECIDE, m));
    PetscCallAbort(PETSC_COMM_SELF, VecSetFromOptions(b));
    PetscCallAbort(PETSC_COMM_SELF, VecSetUp(b));

    PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(b, &x));
    PetscCallAbort(PETSC_COMM_SELF, VecSetUp(x));
    PetscCallAbort(PETSC_COMM_SELF, VecSetValuesBlocked(x, m, indices.data(), x0, INSERT_VALUES));
    OMP_FOR_(m * m, SSIMD collapse(2))
    for (PetscInt i = 0; i < m; ++i)
        for (PetscInt j = 0; j < m; ++j)
            PetscCallAbort(PETSC_COMM_SELF, MatSetValue(A, i, j, A_[i + j * m], INSERT_VALUES));

    // Create the linear solver context
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A, A));

    // Configure the solver for antisymmetric matrices
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));
    PetscCallAbort(PETSC_COMM_SELF, PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
    PetscCallAbort(PETSC_COMM_SELF, PCJacobiSetFixDiagonal(pc, PETSC_FALSE));
    PetscCallAbort(PETSC_COMM_SELF, KSPGMRESSetRestart(ksp, 3000));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPGMRES));

    // Set more options
    PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_gmres_preallocate", nullptr));
    PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_gmres_modifiedgramschmidt", nullptr));
    PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_gmres_cgs_refinement_type", "refine_always"));
    // PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(nullptr, "-ksp_pipefgmres_shift", "10"));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, iter));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetFromOptions(ksp));

    PetscCallAbort(PETSC_COMM_SELF, VecSetValuesBlocked(b, m, indices.data(), b_, INSERT_VALUES));
    // PetscCallAbort(PETSC_COMM_SELF, MatSetValuesBlocked(A, m, indices.data(), m, indices.data(), A_, INSERT_VALUES));

    PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCallAbort(PETSC_COMM_SELF, VecAssemblyBegin(b));
    PetscCallAbort(PETSC_COMM_SELF, VecAssemblyEnd(b));

    LOG4_END();
}

antisymmetric_solver::~antisymmetric_solver()
{
    // Cleanup
    PetscCallCXXAbort(PETSC_COMM_SELF, MatDestroy(&A));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&b));
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&x));
    PetscCallCXXAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
}

// Retrieve solution vector
PetscReal antisymmetric_solver::solve()
{
    LOG4_BEGIN();

#ifndef NDEBUG
    // Monitor convergence by checking the cost function
    PetscCallAbort(PETSC_COMM_SELF, KSPMonitorSet(ksp, [](KSP ksp, PetscInt it, PetscReal rnorm, void *ctx) {
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
    PROFILE_(KSPSolve(ksp, b, x));

    // Get convergence information
    KSPConvergedReason reason;
    PetscInt iterations;
    PetscReal rnorm;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &reason));
    PetscCallAbort(PETSC_COMM_SELF, KSPGetResidualNorm(ksp, &rnorm));
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &iterations));
    LOG4_DEBUG("Solver " << (reason > 0 ? "converged" : "diverged") << " after " << iterations << " iterations, score " << rnorm);
    return rnorm;
}

Vec antisymmetric_solver::get_solution()
{
    return x;
}

void antisymmetric_solver::get_solution(double *const sol)
{
    PetscCallAbort(PETSC_COMM_SELF, VecGetValues(x, m, indices.data(), sol));
}

}
}
