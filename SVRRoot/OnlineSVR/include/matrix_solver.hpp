//
// Created by zarko on 06/03/2025.
//

#ifndef SVR_MATRIX_SOLVER_HPP
#define SVR_MATRIX_SOLVER_HPP

#include <mpi.h>
#include <petsc.h>
#include <petscksp.h>

namespace svr {
namespace solvers {

class antisymmetric_solver {
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;
    PetscInt m;
//    const double *const p_b, *const p_A;
    std::vector<PetscInt> indices;

public:
    antisymmetric_solver(const PetscInt m, const PetscInt iter, const PetscScalar *const x0_, const PetscScalar *const A_,
                         const PetscScalar *const b_, const bool direct_solve, const uint16_t irwls_iter);

    ~antisymmetric_solver();

    PetscReal solve();

    // Retrieve solution vector
    Vec get_solution();

    void get_solution(double *const sol);
};

}
}

#endif //SVR_MATRIX_SOLVER_HPP
