//
// Created by zarko on 06/03/2025.
//

#ifndef SVR_PSOLVER_HPP
#define SVR_PSOLVER_HPP

#include <petsc.h>
#include <petscksp.h>
#include <functional>
#include <cmath>
#include <iostream>

namespace svr {

class antisymmetric_solver {
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;
    PetscInt n;
    std::vector<PetscInt> indices;

public:
    antisymmetric_solver(const PetscInt size, const PetscInt iter, const PetscScalar *const x0, const PetscScalar *const A_, const PetscScalar *const b_)
            : n(size)
    {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0);

        // Initialize PETSc objects
        PetscCallAbort(PETSC_COMM_SELF, MatCreate(PETSC_COMM_WORLD, &A));
        PetscCallAbort(PETSC_COMM_SELF, MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
        PetscCallAbort(PETSC_COMM_SELF, MatSetType(A, MATDENSE));
        PetscCallAbort(PETSC_COMM_SELF, MatSetFromOptions(A));
        PetscCallAbort(PETSC_COMM_SELF, MatSetUp(A));

        PetscCallAbort(PETSC_COMM_SELF, VecCreate(PETSC_COMM_WORLD, &b));
        PetscCallAbort(PETSC_COMM_SELF, VecSetSizes(b, PETSC_DECIDE, n));
        PetscCallAbort(PETSC_COMM_SELF, VecSetFromOptions(b));
        PetscCallAbort(PETSC_COMM_SELF, VecSetUp(b));

        PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(b, &x));
        PetscCallAbort(PETSC_COMM_SELF, VecSetUp(x));
        PetscCallAbort(PETSC_COMM_SELF, VecSetValuesBlocked(x, n, indices.data(), x0, INSERT_VALUES));
        OMP_FOR_(n * n, SSIMD collapse(2))
        for (PetscInt i = 0; i < n; ++i)
            for (PetscInt j = 0; j < n; ++j)
                PetscCallAbort(PETSC_COMM_SELF, MatSetValue(A, i, j, A_[i + j * n], INSERT_VALUES));

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

        PetscCallAbort(PETSC_COMM_SELF, VecSetValuesBlocked(b, n, indices.data(), b_, INSERT_VALUES));
        // PetscCallAbort(PETSC_COMM_SELF, MatSetValuesBlocked(A, n, indices.data(), n, indices.data(), A_, INSERT_VALUES));

        PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
        PetscCallAbort(PETSC_COMM_SELF, VecAssemblyBegin(b));
        PetscCallAbort(PETSC_COMM_SELF, VecAssemblyEnd(b));
    }

    antisymmetric_solver()
    {
        // Cleanup
        PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&A));
        PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b));
        PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x));
        PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    }

    PetscReal solve()
    {
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

    // Retrieve solution vector
    Vec get_solution()
    {
        return x;
    }

    void get_solution(double *sol)
    {
        PetscCallAbort(PETSC_COMM_SELF, VecGetValues(x, n, indices.data(), sol));
    }
};

}

#endif //SVR_PSOLVER_HPP
