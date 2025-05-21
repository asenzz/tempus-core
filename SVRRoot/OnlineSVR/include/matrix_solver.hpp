//
// Created by zarko on 06/03/2025.
//

#ifndef SVR_MATRIX_SOLVER_HPP
#define SVR_MATRIX_SOLVER_HPP

#define USE_PETSC_SOLVER

#include <mpi.h>
#ifdef USE_PETSC_SOLVER
#include <petsc.h>
#else
#include <ginkgo/ginkgo.hpp>
#endif

namespace svr {
namespace solvers {

class antisymmetric_solver {
public:

#ifdef USE_PETSC_SOLVER

    using Tv = PetscScalar;
    using Ti = PetscInt;

#else

    using Tv = double;
    using Ti = uint32_t;

#endif

    static constexpr Ti C_gpu_solve_threshold = 3e3;

private:
    const bool direct_solve;
    const Tv *const x0_, *const A_, *const b_;

    const Ti iter;
    const uint16_t irwls_iter;
    Ti m, n;
    std::vector<Ti> indices;


public:

    antisymmetric_solver(const Ti m, const Ti n, const Ti iter, const Tv *const x0_, const Tv *const A_, const Tv *const b_, const bool direct_solve, const uint16_t irwls_iter);

    ~antisymmetric_solver();

    Tv operator()(Tv *const sol) const;
};

}
}

#endif //SVR_MATRIX_SOLVER_HPP
