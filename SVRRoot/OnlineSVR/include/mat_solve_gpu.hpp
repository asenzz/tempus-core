//
// Created by jarko on 18/01/19.
//
#if 0 // Deprecated!
#ifndef SVR_MAT_SOLVE_GPU_HPP
#define SVR_MAT_SOLVE_GPU_HPP

#include <CL/cl.h>

#include "common/gpu_handler.hpp"

namespace svr {

typedef enum
{
    MagmaUpper = 121,
    MagmaLower = 122,
    MagmaFull = 123   /* lascl, laset */
} uplo_t;

typedef enum
{
    MagmaNoTrans = 111,
    MagmaTrans = 112,
    MagmaConjTrans = 113
} trans_t;

typedef enum
{
    MagmaLeft = 141,
    MagmaRight = 142,
} side_t;

typedef enum
{
    MagmaNonUnit = 131,
    MagmaUnit = 132
} magma_diag_t;

typedef enum {
    MagmaForward       = 391,  /* larfb */
    MagmaBackward      = 392
} direct_t;

typedef enum {
    MagmaColumnwise    = 401,  /* larfb */
    MagmaRowwise       = 402
} storev_t;


int
dposv_gpu(
        const uplo_t uplo, const int n, const int nrhs,
        cl_mem dA, const int ldda,
        cl_mem dB, const int lddb,
        cl_command_queue queue,
        int *p_info);

int
dgesv_gpu(
        const int n, const int nrhs,
        cl_mem dA, const int ldda,
        int *ipiv,
        cl_mem dB, const int lddb,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dgeqrf_gpu(
        int m, int n,
        cl_mem dA, size_t dA_offset,  int ldda,
        double *tau, cl_mem dT, size_t dT_offset,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dgels_gpu(
        trans_t trans, int m, int n, int nrhs,
        cl_mem dA, size_t dA_offset,  int ldda,
        cl_mem dB, size_t dB_offset,  int lddb,
        double *hwork_, int lwork,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dgeqrs_gpu(
        const int m, const int n, const int nrhs,
        cl_mem dA, const size_t dA_offset, const int ldda,
        double *tau,   cl_mem dT, size_t dT_offset,
        cl_mem dB, const size_t dB_offset, const int lddb,
        double *hwork, const int lwork,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dgetrf_gpu(
        const int m, const int n,
        cl_mem dA, size_t dA_offset, const int ldda,
        int *p_ipiv,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dgetrs_gpu(
        trans_t trans, int n, int nrhs,
        cl_mem dA, size_t dA_offset, int ldda,
        int *p_ipiv,
        cl_mem dB, size_t dB_offset, int lddb,
        cl_command_queue queue,
        int *info);

int
dlarfb_gpu(
        side_t side, trans_t trans, direct_t direct, storev_t storev,
        int m, int n, int k,
        cl_mem dV, size_t dV_offset, int ldv,
        cl_mem dT, size_t dT_offset, int ldt,
        cl_mem dC, size_t dC_offset, int ldc,
        cl_mem dwork, size_t dwork_offset, int ldwork,
        cl_command_queue queue);


int
dormqr_gpu(
        side_t side, trans_t trans,
        int m, int n, int k,
        cl_mem dA, size_t dA_offset, int ldda,
        double *tau,
        cl_mem dC, size_t dC_offset, int lddc,
        double *hwork_, int lwork,
        cl_mem dT, size_t dT_offset, int nb,
        viennacl::ocl::context &ctx,
        int *p_info);

int
dpotrf_gpu(
        uplo_t uplo,
        int n,
        cl_mem dA,
        size_t dA_offset,
        int ldda,
        cl_command_queue queue,
        int *p_info);

int
dpotrs_gpu(
        uplo_t uplo, int n, int nrhs,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dB, size_t dB_offset, int lddb,
        cl_command_queue queue,
        int *info);

void
dlaswp(
        const int n,
        cl_mem dAT, size_t dAT_offset, const int ldda,
        const int k1, const int k2,
        int *ipiv, const int inci,
        viennacl::ocl::context &ctx);


} /* namespace svr */

#endif //SVR_MAT_SOLVE_GPU_HPP
#endif