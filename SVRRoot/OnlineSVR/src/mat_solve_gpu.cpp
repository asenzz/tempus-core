//
// Created by jarko on 18/01/19.
//
#if 0
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "mat_solve_gpu.hpp"
#include <clblast.h>
#include <mkl_lapacke.h>
#include <mkl.h>
#include "common.hpp"
#include "kernel_base.hpp"

namespace svr {

static cl_event *g_event = nullptr;

#define NX 32
#define NB 16
#define NY 8

#define MAGMA_D_MAKE(r, i)         (r)
#define MAGMA_ERR_HOST_ALLOC       -112
#define MAGMA_SUCCESS               0

#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_NEG_ONE           (-1.0)


const int magma2amdblas_constants[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,                      // 100
        (int) clblast::Layout::kRowMajor,         // 101: MagmaRowMajor
        (int) clblast::Layout::kColMajor,      // 102: MagmaColMajor
        0, 0, 0, 0, 0, 0, 0, 0,
        (int) clblast::Transpose::kNo,          // 111: MagmaNoTrans
        (int) clblast::Transpose::kYes,            // 112: MagmaTrans
        (int) clblast::Transpose::kConjugate,        // 113: MagmaConjTrans
        0, 0, 0, 0, 0, 0, 0,
        (int) clblast::Triangle::kUpper,            // 121: MagmaUpper
        (int) clblast::Triangle::kLower,            // 122: MagmaLower
        0, 0, 0, 0, 0, 0, 0, 0,
        (int) clblast::Diagonal::kNonUnit,          // 131: MagmaNonUnit
        (int) clblast::Diagonal::kUnit,             // 132: MagmaUnit
        0, 0, 0, 0, 0, 0, 0, 0,
        (int) clblast::Side::kLeft,             // 141: MagmaLeft
        (int) clblast::Side::kRight,            // 142: MagmaRight
        0, 0, 0, 0, 0, 0, 0, 0
};


void
magma_malloc(viennacl::ocl::context &ctx, cl_mem *ptrPtr, size_t size)
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if (size == 0) size = sizeof(cl_double2);
    cl_int err;
    *ptrPtr = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE, size, nullptr, &err);
    CL_CHECK(err);
}


static inline void
magma_dmalloc(viennacl::ocl::context &ctx, cl_mem *ptr_ptr, const size_t n)
{
    magma_malloc(ctx, (cl_mem *) ptr_ptr, n * sizeof(double));
}


int
magma_malloc_cpu(void **ptr_ptr, size_t size)
{
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if (size == 0)
        size = sizeof(cl_double2);
#if 1
#if defined( _WIN32 ) || defined( _WIN64 )
    *ptr_ptr = _aligned_malloc( size, 32 );
    if ( *ptr_ptr == nullptr ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#else
    int err = posix_memalign(ptr_ptr, 32, size);
    if (err != 0) {
        *ptr_ptr = nullptr;
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
#else
    *ptr_ptr = malloc( size );
    if ( *ptr_ptr == nullptr ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    return MAGMA_SUCCESS;
}


int
magma_free(cl_mem ptr)
{
    cl_int err = clReleaseMemObject(ptr);
    if (err != CL_SUCCESS) LOG4_ERROR("Failed freeing pointer at " << ptr << " error " << err);
    return err;
}


int
magma_free_cpu(void *ptr)
{
#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free(ptr);
#endif
    return MAGMA_SUCCESS;
}


inline int
magma_dmalloc_cpu(double **ptrPtr, const size_t n)
{
    return magma_malloc_cpu((void **) ptrPtr, n * sizeof(double));
}


clblast::Side clblast_side_const(const side_t magma_const)
{
    assert(magma_const >= MagmaLeft);
    assert(magma_const <= MagmaRight);
    return (clblast::Side) magma2amdblas_constants[magma_const];
}


clblast::Diagonal clblast_diag_const(const magma_diag_t magma_const)
{
    assert(magma_const >= MagmaNonUnit);
    assert(magma_const <= MagmaUnit);
    return (clblast::Diagonal) magma2amdblas_constants[magma_const];
}


clblast::Transpose clblast_trans_const(const trans_t magma_const)
{
    assert(magma_const >= MagmaNoTrans);
    assert(magma_const <= MagmaConjTrans);
    return (clblast::Transpose) magma2amdblas_constants[magma_const];
}


clblast::Triangle clblast_uplo_const(const uplo_t magma_const)
{
    assert(magma_const >= MagmaUpper);
    assert(magma_const <= MagmaLower);
    return (clblast::Triangle) magma2amdblas_constants[magma_const];
}


void
magma_dgetmatrix(
        int m, int n,
        cl_mem dA_src, size_t dA_offset, int ldda,
        double *hB_dst, int ldhb,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0) return;

    size_t buffer_origin[3] = {dA_offset * sizeof(double), 0, 0};
    size_t host_orig[3] = {0, 0, 0};
    size_t region[3] = {m * sizeof(double), (size_t) n, 1};
    cl_int err = clEnqueueReadBufferRect(
            queue, dA_src, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            ldda * sizeof(double), 0,
            ldhb * sizeof(double), 0,
            hB_dst, 0, nullptr, g_event);
    CL_CHECK(err);
}


void
magma_dsetmatrix(
        int m, int n,
        double const *hA_src, int ldha,
        cl_mem dB_dst, size_t dB_offset, int lddb,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0) return;

    size_t buffer_origin[3] = {dB_offset * sizeof(double), 0, 0};
    size_t host_orig[3] = {0, 0, 0};
    size_t region[3] = {m * sizeof(double), (size_t) n, 1};
    cl_int err = clEnqueueWriteBufferRect(
            queue, dB_dst, CL_TRUE,  // blocking
            buffer_origin, host_orig, region,
            lddb * sizeof(double), 0,
            ldha * sizeof(double), 0,
            hA_src, 0, nullptr, g_event);
    CL_CHECK(err);
}

int
magma_get_dpotrf_nb(int m)
{
    if (m <= 4256) return 128;
    else return 256;
}


void
magma_dsetmatrix_async(
        int m, int n,
        double const *hA_src, int ldha,
        cl_mem dB_dst, size_t dB_offset, int lddb,
        cl_command_queue queue, cl_event *event)
{
    if (m <= 0 || n <= 0) return;

    size_t buffer_origin[3] = {dB_offset * sizeof(double), 0, 0};
    size_t host_orig[3] = {0, 0, 0};
    size_t region[3] = {m * sizeof(double), (size_t) n, 1};
    cl_int err = clEnqueueWriteBufferRect(
            queue, dB_dst, CL_FALSE,  // non-blocking
            buffer_origin, host_orig, region,
            lddb * sizeof(double), 0,
            ldha * sizeof(double), 0,
            hA_src, 0, nullptr, event);
    clFlush(queue);
    CL_CHECK(err);
}


void
magma_dsyrk(
        uplo_t uplo, trans_t trans,
        int n, int k,
        double alpha,
        cl_mem dA, size_t dA_offset, int ldda,
        double beta,
        cl_mem dC, size_t dC_offset, int lddc,
        cl_command_queue queue)
{
    clblast::StatusCode err = clblast::Syrk<double>(
            clblast::Layout::kColMajor,
            clblast_uplo_const(uplo),
            clblast_trans_const(trans),
            (size_t) n, (size_t) k,
            alpha, dA, dA_offset, (size_t) ldda,
            beta, dC, dC_offset, (size_t) lddc,
            &queue, g_event);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}


void
magma_dgemv(
        trans_t transA,
        int m, int n,
        double alpha,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dx, size_t dx_offset, int incx,
        double beta,
        cl_mem dy, size_t dy_offset, int incy,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0)
        return;

    const auto err = clblast::Gemv(
            clblast::Layout::kColMajor,
            clblast_trans_const(transA),
            m, n,
            alpha, dA, dA_offset, ldda,
            dx, dx_offset, incx,
            beta, dy, dy_offset, incy,
            &queue, g_event);
    clFlush(queue);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}


void
magma_dcopymatrix(
        int m, int n,
        cl_mem dA_src, size_t dA_offset, int ldda,
        cl_mem dB_dst, size_t dB_offset, int lddb,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0) return;

    size_t src_origin[3] = {dA_offset * sizeof(double), 0, 0};
    size_t dst_orig[3] = {dB_offset * sizeof(double), 0, 0};
    size_t region[3] = {m * sizeof(double), (size_t) n, 1};
    cl_int err = clEnqueueCopyBufferRect(
            queue, dA_src, dB_dst,
            src_origin, dst_orig, region,
            ldda * sizeof(double), 0,
            lddb * sizeof(double), 0,
            0, NULL, g_event);
    CL_CHECK(err);
}


void
magma_dgemm(
        const trans_t transA, const trans_t transB,
        const int m, const int n, const int k,
        const double alpha,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dB, size_t dB_offset, int lddb,
        const double beta,
        cl_mem dC, size_t dC_offset, int lddc,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0 || k <= 0) return;
    const auto err = clblast::Gemm<double>(
            clblast::Layout::kColMajor,
            clblast_trans_const(transA),
            clblast_trans_const(transB),
            m, n, k,
            alpha, dA, dA_offset, ldda,
            dB, dB_offset, lddb,
            beta, dC, dC_offset, lddc,
            &queue, g_event);
    clFlush(queue);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}


void
magma_dtrmm(
        side_t side, uplo_t uplo, trans_t trans, magma_diag_t diag,
        int m, int n,
        double alpha,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dB, size_t dB_offset, int lddb,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0) return;

    const auto err = clblast::Trmm(
            clblast::Layout::kColMajor,
            clblast_side_const(side),
            clblast_uplo_const(uplo),
            clblast_trans_const(trans),
            clblast_diag_const(diag),
            m, n,
            alpha, dA, dA_offset, ldda,
            dB, dB_offset, lddb,
            &queue, g_event);
    clFlush(queue);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}

void
magma_dtrsm(
        const side_t side, const uplo_t uplo, const trans_t trans, const magma_diag_t diag,
        const int m, const int n,
        const double alpha,
        cl_mem dA, size_t dA_offset, const int ldda,
        cl_mem dB, size_t dB_offset, const int lddb,
        cl_command_queue queue)
{
    if (m <= 0 || n <= 0) return;
    const auto err = clblast::Trsm<double>(
            clblast::Layout::kColMajor,
            clblast_side_const(side),
            clblast_uplo_const(uplo),
            clblast_trans_const(trans),
            clblast_diag_const(diag),
            m, n,
            alpha, dA, dA_offset, ldda,
            dB, dB_offset, lddb,
            &queue, g_event);
    clFlush(queue);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}


void
magma_dtrsv(
        const uplo_t uplo, const trans_t trans, const magma_diag_t diag,
        const int n,
        cl_mem dA, size_t dA_offset, const int ldda,
        cl_mem dx, size_t dx_offset, const int incx,
        cl_command_queue queue)
{
    if (n <= 0) return;

    const auto err = clblast::Trsv<double>(
            clblast::Layout::kColMajor,
            clblast_uplo_const(uplo),
            clblast_trans_const(trans),
            clblast_diag_const(diag),
            n,
            dA, dA_offset, ldda,
            dx, dx_offset, incx,
            &queue, g_event);
    clFlush(queue);
    if (err != clblast::StatusCode::kSuccess) LOG4_THROW("Call failed with error " << (int) err);
}


void
magma_dgetmatrix_async(
        int m, int n,
        cl_mem dA_src, size_t dA_offset, int ldda,
        double *hB_dst, int ldhb,
        cl_command_queue queue, cl_event *event)
{
    if (m <= 0 || n <= 0)
        return;

    size_t buffer_origin[3] = {dA_offset * sizeof(double), 0, 0};
    size_t host_orig[3] = {0, 0, 0};
    size_t region[3] = {m * sizeof(double), (size_t) n, 1};
    cl_int err = clEnqueueReadBufferRect(
            queue, dA_src, CL_FALSE,  // non-blocking
            buffer_origin, host_orig, region,
            ldda * sizeof(double), 0,
            ldhb * sizeof(double), 0,
            hB_dst, 0, nullptr, event);
    clFlush(queue);
    CL_CHECK(err);
}


int
magma_event_sync(cl_event event)
{
    cl_int err = clWaitForEvents(1, &event);
    CL_CHECK(err);
    return err;
}


int
magma_queue_sync(cl_command_queue queue)
{
    cl_int err = clFinish(queue);
    clFlush(queue);
    CL_CHECK(err);
    return err;
}


int
dpotrf_gpu(
        uplo_t uplo,
        int n,
        cl_mem dA,
        size_t dA_offset,
        int ldda,
        cl_command_queue queue,
        int *p_info)
{
// produces pointer and offset as two args to magmaBLAS routines
#define dA_dpotrf(i, j)  dA, ( (dA_offset) + (i) + (j)*ldda )

// produces pointer as single arg to BLAS routines
#define A_dpotrf(i, j)  &A[ (i) + (j)*lda ]

    int j, jb, nb;
    double z_one = MAGMA_D_MAKE(1.0, 0.0);
    double mz_one = MAGMA_D_MAKE(-1.0, 0.0);
    double one = 1.0;
    double m_one = -1.0;
    double *work;
    int err;

    *p_info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) {
        *p_info = -1;
    } else if (n < 0) {
        *p_info = -2;
    } else if (ldda < std::max(1, n)) {
        *p_info = -4;
    }
    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    }

    nb = magma_get_dpotrf_nb(n);

    err = magma_dmalloc_cpu(&work, nb * nb);
    if (err != MAGMA_SUCCESS) {
        *p_info = MAGMA_ERR_HOST_ALLOC;
        return *p_info;
    }

    cl_event event = nullptr;
    if (nb <= 1 || nb >= n) {
        // use unblocked code
        magma_dgetmatrix(n, n, dA, dA_offset, ldda, work, n, queue);
        *p_info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo == MagmaLower ? 'L' : 'U', n, work, n);
        if (*p_info != 0) {
            LOG4_DEBUG("LAPACKE_dpotrf returned1 " << " n " << n << " nb " << nb << " ldda " << ldda << " work  0 "
                                                   << work[0] << " " << "  " << *p_info);
            magma_queue_sync(queue);
            magma_free_cpu(work);
            return *p_info;
        }
        magma_dsetmatrix(n, n, work, n, dA, dA_offset, ldda, queue);
    } else {
        if (uplo == MagmaUpper) {
            // --------------------
            // compute Cholesky factorization A = U'*U
            // using the left looking algorithm
            for (j = 0; j < n; j += nb) {
                // apply all previous updates to diagonal block
                jb = std::min(nb, n - j);
                if (j > 0)
                    magma_dsyrk(
                            MagmaUpper, MagmaConjTrans, jb, j,
                            m_one, dA_dpotrf(0, j), ldda,
                            one, dA_dpotrf(j, j), ldda, queue);

                // start asynchronous data transfer
                magma_dgetmatrix_async(jb, jb, dA_dpotrf(j, j), ldda, work, jb, queue, &event);

                // apply all previous updates to block row right of diagonal block
                if (j + jb < n)
                    magma_dgemm(MagmaConjTrans, MagmaNoTrans,
                                jb, n - j - jb, j,
                                mz_one, dA_dpotrf(0, j), ldda,
                                dA_dpotrf(0, j + jb), ldda,
                                z_one, dA_dpotrf(j, j + jb), ldda, queue);

                // simultaneous with above dgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                magma_event_sync(event);
                *p_info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', jb, work, jb);
                if (*p_info != 0) {
                    magma_queue_sync(queue);
                    magma_free_cpu(work);
                    LOG4_ERROR("LAPACKE_dpotrf returned2 " << *p_info);
                    return *p_info;
                }
                magma_dsetmatrix_async(jb, jb, work, jb, dA_dpotrf(j, j), ldda, queue, &event);

                // apply diagonal block to block row right of diagonal block
                if (j + jb < n) {
                    magma_event_sync(event);
                    magma_dtrsm(
                            MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                            jb, n - j - jb,
                            z_one, dA_dpotrf(j, j), ldda,
                            dA_dpotrf(j, j + jb), ldda, queue);
                }
            }
        } else {
            // --------------------
            // compute Cholesky factorization A = L*L'
            // using the left looking algorithm
            for (j = 0; j < n; j += nb) {
                // apply all previous updates to diagonal block
                jb = std::min(nb, n - j);
                if (j > 0)
                    magma_dsyrk(
                            MagmaLower, MagmaNoTrans, jb, j,
                            m_one, dA_dpotrf(j, 0), ldda,
                            one, dA_dpotrf(j, j), ldda, queue);

                // start asynchronous data transfer
                magma_dgetmatrix_async(jb, jb, dA_dpotrf(j, j), ldda, work, jb, queue, &event);

                // apply all previous updates to block column below diagonal block
                if (j + jb < n)
                    magma_dgemm(
                            MagmaNoTrans, MagmaConjTrans,
                            n - j - jb, jb, j,
                            mz_one, dA_dpotrf(j + jb, 0), ldda,
                            dA_dpotrf(j, 0), ldda,
                            z_one, dA_dpotrf(j + jb, j), ldda, queue);

                // simultaneous with above dgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                magma_event_sync(event);
                *p_info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', jb, work, jb);
                if (*p_info != 0) {
                    magma_queue_sync(queue);
                    magma_free_cpu(work);
                    LOG4_ERROR("LAPACKE_dpotrf returned3 " << *p_info);
                    return *p_info;
                }
                magma_dsetmatrix_async(jb, jb, work, jb, dA_dpotrf(j, j), ldda, queue, &event);

                // apply diagonal block to block column below diagonal
                if (j + jb < n) {
                    magma_event_sync(event);
                    magma_dtrsm(
                            MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                            n - j - jb, jb,
                            z_one, dA_dpotrf(j, j), ldda,
                            dA_dpotrf(j + jb, j), ldda, queue);
                }
            }
        }
    }
    magma_queue_sync(queue);
    magma_free_cpu(work);
    return *p_info;
}


int
dpotrs_gpu(
        uplo_t uplo, int n, int nrhs,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dB, size_t dB_offset, int lddb,
        cl_command_queue queue,
        int *info)
{
    double z_one = MAGMA_D_MAKE(1.0, 0.0);

    *info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower) *info = -1;
    if (n < 0)
        *info = -2;
    if (nrhs < 0)
        *info = -3;
    if (ldda < fmax(1, n))
        *info = -5;
    if (lddb < fmax(1, n))
        *info = -7;
    if (*info != 0) {
        LOG4_ERROR("Call failed with " << *info);
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) return *info;

    if (uplo == MagmaUpper) {
        if (nrhs == 1) {
            magma_dtrsv(MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
            magma_dtrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
        } else {
            magma_dtrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
            magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
        }
    } else {
        if (nrhs == 1) {
            magma_dtrsv(MagmaLower, MagmaNoTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
            magma_dtrsv(MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
        } else {
            magma_dtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
            magma_dtrsm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
        }
    }
    magma_queue_sync(queue);
    return *info;
}

int
dposv_gpu(
        uplo_t uplo, const int n, const int nrhs,
        cl_mem dA, const int ldda,
        cl_mem dB, const int lddb,
        cl_command_queue queue,
        int *p_info)
{
    *p_info = 0;
    if (uplo != MagmaUpper && uplo != MagmaLower)
        *p_info = -1;
    if (n < 0)
        *p_info = -2;
    if (nrhs < 0)
        *p_info = -3;
    if (ldda < std::max(1, n))
        *p_info = -5;
    if (lddb < std::max(1, n))
        *p_info = -7;
    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) return *p_info;

    dpotrf_gpu(uplo, n, dA, 0, ldda, queue, p_info);
    if (*p_info == 0) dpotrs_gpu(uplo, n, nrhs, dA, 0, ldda, dB, 0, lddb, queue, p_info);
    else {
        LOG4_DEBUG("magma_dpotrs_gpu returned " << *p_info);
        return *p_info;
    }

    return *p_info;
}


int magma_get_dgetrf_nb(const int m)
{
    if (m <= 2048) return 64;
    else if (m < 7200) return 192;
    else return 256;
}


void
magmablas_dtranspose_inplace(
        const int n,
        cl_mem dA, size_t dA_offset, int ldda,
        viennacl::ocl::context &ctx)
{
    cl_int err;

    int info = 0;
    if (n < 0)
        info = -1;
    else if (ldda < n)
        info = -3;

    if (info != 0) {
        LOG4_ERROR("Call failed with " << info);
        return;  //info;
    }

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());

    size_t threads[2] = {NB, NB};
    int nblock = (n + NB - 1) / NB;

    // need 1/2 * (nblock+1) * nblock to cover lower triangle and diagonal of matrix.
    // block assignment differs depending on whether nblock is odd or even.
    if (nblock % 2 == 1) {
        size_t grid[2] = {(size_t) nblock, (size_t) (nblock + 1) / 2};
        grid[0] *= threads[0];
        grid[1] *= threads[1];
        svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "dtranspose_inplace_odd", "dtranspose_inplace");
        svr::cl12::ocl_kernel dtranspose_kernel(
                ctx.get_kernel("dtranspose_inplace", "dtranspose_inplace_odd").handle());

        dtranspose_kernel.set_args(n, dA, dA_offset, ldda);
        err = cq_command_queue.finish();
        CL_CHECK(err);

        err = dtranspose_kernel.enqueue(
                cq_command_queue,
                cl::NullRange, svr::cl12::ndrange(grid), svr::cl12::ndrange(threads),
                nullptr, nullptr);
        CL_CHECK(err);

        err = cq_command_queue.finish();
        CL_CHECK(err);
    } else {
        size_t grid[2] = {(size_t) nblock + 1, (size_t) nblock / 2};
        grid[0] *= threads[0];
        grid[1] *= threads[1];

        svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "dtranspose_inplace_even", "dtranspose_inplace");
        svr::cl12::ocl_kernel dtranspose_kernel(
                ctx.get_kernel("dtranspose_inplace", "dtranspose_inplace_even").handle());
        dtranspose_kernel.set_args(n, dA, dA_offset, ldda);
        err = cq_command_queue.finish();
        CL_CHECK(err);

        err = dtranspose_kernel.enqueue(
                cq_command_queue,
                cl::NullRange, svr::cl12::ndrange(grid), svr::cl12::ndrange(threads),
                nullptr, nullptr);
        CL_CHECK(err);

        err = cq_command_queue.finish();
        CL_CHECK(err);
    }
}


void
magmablas_dtranspose(
        const int m, const int n,
        cl_mem dA, const size_t dA_offset, const int ldda,
        cl_mem dAT, const size_t dAT_offset, const int lddat,
        viennacl::ocl::context &ctx)
{
    cl_int err;

    int info = 0;
    if (m < 0)
        info = -1;
    else if (n < 0)
        info = -2;
    else if (ldda < m)
        info = -4;
    else if (lddat < n)
        info = -6;

    if (info != 0) {
        LOG4_ERROR("Call failed with " << info);
        return;
    }

    /* Quick return */
    if (m == 0 || n == 0) return;

    size_t threads[2] = {NX, NY};
    size_t grid[2] = {(size_t) (m + NB - 1) / NB, (size_t) (n + NB - 1) / NB};
    grid[0] *= threads[0];
    grid[1] *= threads[1];

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "dtranspose_kernel", "dtranspose");
    svr::cl12::ocl_kernel dtranspose_kernel(ctx.get_kernel("dtranspose", "dtranspose_kernel").handle());

    dtranspose_kernel.set_args(m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat);
    err = cq_command_queue.finish();
    CL_CHECK(err);

    err = dtranspose_kernel.enqueue(
            cq_command_queue,
            cl::NullRange, svr::cl12::ndrange(grid), svr::cl12::ndrange(threads),
            nullptr, nullptr);
    CL_CHECK(err);

    err = cq_command_queue.finish();
    CL_CHECK(err);
}

#define MAX_PIVOTS 128
#define NTHREADS   64

typedef struct
{
    int npivots;
    int ipiv[MAX_PIVOTS];
} dlaswp_params_t; // Verify this is the struct type int size that is used inside the OpenCL kernel

void
dlaswp(
        const int n,
        cl_mem dAT, size_t dAT_offset, const int ldda,
        const int k1, const int k2,
        int *ipiv, const int inci,
        viennacl::ocl::context &ctx)
{
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_int err;

    int info = 0;
    if (n < 0)
        info = -1;
    else if (k1 < 1 || k1 > n)
        info = -4;
    else if (k2 < 1 || k2 > n)
        info = -5;
    else if (inci <= 0)
        info = -7;

    if (info != 0) {
        LOG4_ERROR("Call failed with " << info);
        return;  //info;
    }

    size_t grid[1] = {(size_t) (n + NTHREADS - 1) / NTHREADS};
    size_t threads[1] = {NTHREADS};
    grid[0] *= threads[0];
    dlaswp_params_t params;

    svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "dlaswp_kernel", "dlaswp");
    svr::cl12::ocl_kernel dlaswp_kernel(ctx.get_kernel("dlaswp", "dlaswp_kernel").handle());

    for (int k = k1 - 1; k < k2; k += MAX_PIVOTS) {
        int npivots = std::min(MAX_PIVOTS, k2 - k);
        params.npivots = npivots;
        for (int j = 0; j < npivots; ++j) params.ipiv[j] = ipiv[(k + j) * inci] - k - 1;
        const size_t k_offset = dAT_offset + k * ldda;
        dlaswp_kernel.set_args(n, dAT, k_offset, ldda, params);
        err = cq_command_queue.finish();
        CL_CHECK(err);

        err = dlaswp_kernel.enqueue(
                cq_command_queue,
                cl::NullRange, cl::NDRange(grid[0]), cl::NDRange(threads[0]),
                nullptr, nullptr);
        CL_CHECK(err);

        err = cq_command_queue.finish();
        CL_CHECK(err);
    }
}


void
copy_to_int(int *dest, const lapack_int *src, const size_t count)
{
    for(size_t ix = 0; ix < count; ++ix) dest[ix] = src[ix];
}


void
copy_to_lapackint(lapack_int *dest, const int *src, const size_t count)
{
    for (size_t ix = 0; ix < count; ++ix) dest[ix] = src[ix];
}


int
dgetrf_gpu(
        const int m, const int n,
        cl_mem dA, size_t dA_offset, const int ldda,
        int *p_ipiv,
        viennacl::ocl::context &ctx,
        int *p_info)
{
#define dA_dgetrf(i_, j_) dA,   dA_offset  + (i_)*nb       + (j_)*nb*ldda
#define dAT_dgetrf(i_, j_) dAT,  dAT_offset + (i_)*nb*lddat + (j_)*nb
#define dAP_dgetrf(i_, j_) dAP,               (i_)          + (j_)*maxm
#define work_dgetrf(i_)   (work + (i_))

    double c_one = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_command_queue queue = cq_command_queue();

    int iinfo, nb;
    int maxm, maxn, mindim;
    int i, j, rows, s, lddat, ldwork;
    cl_mem dAT, dAP;
    double *work;
    size_t dAT_offset;

    /* Check arguments */
    *p_info = 0;
    if (m < 0)
        *p_info = -1;
    else if (n < 0)
        *p_info = -2;
    else if (ldda < std::max(1, m))
        *p_info = -4;

    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) return *p_info;

    /* Function Body */
    mindim = std::min(m, n);
    nb = magma_get_dgetrf_nb(m);
    s = mindim / nb;

    if (nb <= 1 || nb >= std::min(m, n)) {
        /* Use CPU code. */
        if (MAGMA_SUCCESS != magma_dmalloc_cpu(&work, m * n)) {
            *p_info = MAGMA_ERR_HOST_ALLOC;
            return *p_info;
        }
        magma_dgetmatrix(m, n, dA_dgetrf(0, 0), ldda, work_dgetrf(0), m, queue);
        lapack_int lapack_ipiv[MAX_PIVOTS];
        //EMO - not needed copy(lapack_ipiv, p_ipiv, MAX_PIVOTS);
        const auto rc = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, work, m, &lapack_ipiv[0]);
        if (rc != 0) {
            LOG4_DEBUG("Call to LAPACKE_dgetrf failed with " << rc);
            return rc;
        }
        copy_to_int(p_ipiv, &lapack_ipiv[0], std::min(m, n));

        magma_dsetmatrix(m, n, work_dgetrf(0), m, dA_dgetrf(0, 0), ldda, queue);
        magma_free_cpu(work);
    } else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31) / 32) * 32;
        maxn = ((n + 31) / 32) * 32;
        magma_dmalloc(ctx, &dAP, nb * maxm);


        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if (m == n) {
            dAT = dA;
            dAT_offset = dA_offset;
            lddat = ldda;
            magmablas_dtranspose_inplace(m, dAT_dgetrf(0, 0), lddat, ctx);
        } else {
            lddat = maxn;  // N-by-M
            dAT_offset = 0;
            try { magma_dmalloc(ctx, &dAT, lddat * maxm); }
            catch (...) {
                throw;
                magma_free(dAP);
            }
            magmablas_dtranspose(m, n, dA_dgetrf(0, 0), ldda, dAT_dgetrf(0, 0), lddat, ctx);
        }

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_dmalloc_cpu(&work, ldwork * nb)) {
            magma_free(dAP);
            if (dA != dAT)
                magma_free(dAT);

            *p_info = MAGMA_ERR_HOST_ALLOC;
            return *p_info;
        }

        for (j = 0; j < s; j++) {
            // download j-th panel
            magmablas_dtranspose(nb, m - j * nb, dAT_dgetrf(j, j), lddat, dAP_dgetrf(0, 0), maxm, ctx);
            magma_dgetmatrix(m - j * nb, nb, dAP_dgetrf(0, 0), maxm, work_dgetrf(0), ldwork, queue);

            if (j > 0) {
                magma_dtrsm(
                        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                        n - (j + 1) * nb, nb,
                        c_one, dAT_dgetrf(j - 1, j - 1), lddat,
                        dAT_dgetrf(j - 1, j + 1), lddat, queue);
                magma_dgemm(
                        MagmaNoTrans, MagmaNoTrans,
                        n - (j + 1) * nb, m - j * nb, nb,
                        c_neg_one, dAT_dgetrf(j - 1, j + 1), lddat,
                        dAT_dgetrf(j, j - 1), lddat,
                        c_one, dAT_dgetrf(j, j + 1), lddat, queue);
            }

            // do the cpu part
            rows = m - j * nb;
            lapack_int lapack_piv[MAX_PIVOTS];
            //copy(&lapack_piv[0], p_ipiv, MAX_PIVOTS);
            iinfo = LAPACKE_dgetrf(LAPACK_COL_MAJOR, rows, nb, work, ldwork, &lapack_piv[0]);
            if (*p_info == 0 && iinfo > 0) *p_info = iinfo + j * nb;
            copy_to_int(p_ipiv + j * nb, &lapack_piv[0], std::min(rows, nb));

            for (i = j * nb; i < j * nb + nb; ++i) p_ipiv[i] += j * nb;

            dlaswp(n, dAT_dgetrf(0, 0), lddat, j * nb + 1, j * nb + nb, p_ipiv, 1, ctx);

            // upload j-th panel
            magma_dsetmatrix(m - j * nb, nb, work_dgetrf(0), ldwork, dAP_dgetrf(0, 0), maxm, queue);
            magmablas_dtranspose(m - j * nb, nb, dAP_dgetrf(0, 0), maxm, dAT_dgetrf(j, j), lddat, ctx);

            // do the small non-parallel computations (next panel update)
            if (s > j + 1) {
                magma_dtrsm(
                        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                        nb, nb,
                        c_one, dAT_dgetrf(j, j), lddat,
                        dAT_dgetrf(j, j + 1), lddat, queue);
                magma_dgemm(
                        MagmaNoTrans, MagmaNoTrans,
                        nb, m - (j + 1) * nb, nb,
                        c_neg_one, dAT_dgetrf(j, j + 1), lddat,
                        dAT_dgetrf(j + 1, j), lddat,
                        c_one, dAT_dgetrf(j + 1, j + 1), lddat, queue);
            } else {
                magma_dtrsm(
                        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                        n - s * nb, nb,
                        c_one, dAT_dgetrf(j, j), lddat,
                        dAT_dgetrf(j, j + 1), lddat, queue);
                magma_dgemm(
                        MagmaNoTrans, MagmaNoTrans,
                        n - (j + 1) * nb, m - (j + 1) * nb, nb,
                        c_neg_one, dAT_dgetrf(j, j + 1), lddat,
                        dAT_dgetrf(j + 1, j), lddat,
                        c_one, dAT_dgetrf(j + 1, j + 1), lddat, queue);
            }
        }

        int nb0 = std::min(m - s * nb, n - s * nb);
        if (nb0 > 0) {
            rows = m - s * nb;

            magmablas_dtranspose(nb0, rows, dAT_dgetrf(s, s), lddat, dAP_dgetrf(0, 0), maxm, ctx);
            magma_dgetmatrix(rows, nb0, dAP_dgetrf(0, 0), maxm, work_dgetrf(0), ldwork, queue);

            // do the cpu part
            lapack_int lapack_ipiv[MAX_PIVOTS];
            //DANGER - maybe MAX_PIVOTS is too small?
            //EMO - not needed copy(&lapack_ipiv[0], p_ipiv, MAX_PIVOTS);

            iinfo = LAPACKE_dgetrf(LAPACK_COL_MAJOR, rows, nb0, work, ldwork, &lapack_ipiv[0 /* TODO Verify */]);
            if (*p_info == 0 && iinfo > 0)
                *p_info = iinfo + s * nb;
            copy_to_int(p_ipiv + s * nb, &lapack_ipiv[0], std::min(rows, nb0));

            for (i = s * nb; i < s * nb + nb0; ++i) p_ipiv[i] += s * nb;
            dlaswp(n, dAT_dgetrf(0, 0), lddat, s * nb + 1, s * nb + nb0, p_ipiv, 1, ctx);

            // upload j-th panel
            magma_dsetmatrix(rows, nb0, work_dgetrf(0), ldwork, dAP_dgetrf(0, 0), maxm, queue);
            magmablas_dtranspose(rows, nb0, dAP_dgetrf(0, 0), maxm, dAT_dgetrf(s, s), lddat, ctx);

            magma_dtrsm(
                    MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    n - s * nb - nb0, nb0,
                    c_one, dAT_dgetrf(s, s), lddat,
                    dAT_dgetrf(s, s) + nb0, lddat, queue);
        }

        // undo transpose
        if (dA == dAT) {
            magmablas_dtranspose_inplace(m, dAT_dgetrf(0, 0), lddat, ctx);
        } else {
            magmablas_dtranspose(n, m, dAT_dgetrf(0, 0), lddat, dA_dgetrf(0, 0), ldda, ctx);
            magma_free(dAT);
        }

        magma_free(dAP);
        magma_free_cpu(work);
    }

    return *p_info;
}


int
dgetrs_gpu(
        trans_t trans, int n, int nrhs,
        cl_mem dA, size_t dA_offset, int ldda,
        int *p_ipiv,
        cl_mem dB, size_t dB_offset, int lddb,
        cl_command_queue queue,
        int *info)
{
    double c_one = MAGMA_D_ONE;
    double *work = nullptr;
    int notran = (trans == MagmaNoTrans);
    int i1, i2, inc;

    *info = 0;
    if (!notran && trans != MagmaTrans && trans != MagmaConjTrans) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < std::max(1, n)) {
        *info = -5;
    } else if (lddb < std::max(1, n)) {
        *info = -8;
    }
    if (*info != 0) {
        LOG4_ERROR("Call failed with " << *info);
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_dmalloc_cpu(&work, n * nrhs);
    if (work == nullptr) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    i1 = 1;
    i2 = n;
    if (notran) {
        inc = 1;

        /* Solve A * X = B. */
        magma_dgetmatrix(n, nrhs, dB, dB_offset, lddb, work, n, queue);
        std::vector<lapack_int> lapack_ipiv(i2 * std::abs(inc));
        copy_to_lapackint(&lapack_ipiv[0], p_ipiv, i2 * std::abs(inc));
        const auto err = LAPACKE_dlaswp(LAPACK_COL_MAJOR, nrhs, work, n, i1, i2, &lapack_ipiv[0], inc);
        if (err != 0) {
            LOG4_DEBUG("LAPACKE_dlaswp failed with " << err);
            return err;
        }
        //copy(p_ipiv, &lapack_ipiv[0], MAX_PIVOTS);
        magma_dsetmatrix(n, nrhs, work, n, dB, dB_offset, lddb, queue);

        if (nrhs == 1) {
            magma_dtrsv(MagmaLower, MagmaNoTrans, MagmaUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
            magma_dtrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
        } else {
            magma_dtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
            magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB,
                        dB_offset, lddb, queue);
        }
    } else {
        inc = -1;

        /* Solve A' * X = B. */
        if (nrhs == 1) {
            magma_dtrsv(MagmaUpper, trans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
            magma_dtrsv(MagmaLower, trans, MagmaUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue);
        } else {
            magma_dtrsm(MagmaLeft, MagmaUpper, trans, MagmaNonUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset,
                        lddb, queue);
            magma_dtrsm(MagmaLeft, MagmaLower, trans, MagmaUnit, n, nrhs, c_one, dA, dA_offset, ldda, dB, dB_offset,
                        lddb, queue);
        }

        magma_dgetmatrix(n, nrhs, dB, dB_offset, lddb, work, n, queue);
        std::vector<lapack_int> lapack_ipiv(i2 * std::abs(inc));
        copy_to_lapackint(&lapack_ipiv[0], p_ipiv, i2 * std::abs(inc));
        const auto err = LAPACKE_dlaswp(LAPACK_COL_MAJOR, nrhs, work, n, i1, i2, &lapack_ipiv[0], inc);
        if (err != 0) {
            LOG4_DEBUG("LAPACKE_dlaswp failed with " << err);
            return err;
        }
        //copy(p_ipiv, &lapack_ipiv[0], MAX_PIVOTS);
        magma_dsetmatrix(n, nrhs, work, n, dB, dB_offset, lddb, queue);
    }
    magma_free_cpu(work);

    return *info;
}

int
dgesv_gpu(
        const int n, const int nrhs,
        cl_mem dA, const int ldda,
        int *ipiv,
        cl_mem dB, const int lddb,
        viennacl::ocl::context &ctx,
        int *p_info)
{
    *p_info = 0;
    if (n < 0) {
        *p_info = -1;
    } else if (nrhs < 0) {
        *p_info = -2;
    } else if (ldda < std::max(1, n)) {
        *p_info = -4;
    } else if (lddb < std::max(1, n)) {
        *p_info = -7;
    }
    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) return *p_info;

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    dgetrf_gpu(n, n, dA, 0, ldda, ipiv, ctx, p_info);
    if (*p_info == 0)
        dgetrs_gpu(
                MagmaNoTrans, n, nrhs, dA, 0, ldda, ipiv, dB, 0, lddb, cq_command_queue(), p_info);

    return *p_info;
}


int magma_get_dgeqrf_nb(const int m)
{
    if (m <= 2048) return 64;
    else return 128;
}


int
dlarfb_gpu(
        side_t side, trans_t trans, direct_t direct, storev_t storev,
        int m, int n, int k,
        cl_mem dV, size_t dV_offset, int ldv,
        cl_mem dT, size_t dT_offset, int ldt,
        cl_mem dC, size_t dC_offset, int ldc,
        cl_mem dwork, size_t dwork_offset, int ldwork,
        cl_command_queue queue)
{
#define dV(i)       dV, (i)
#define dT(i)       dT, (i)
#define dC(i)       dC, (i)
#define dwork(i) dwork, (i)

    double c_zero = MAGMA_D_MAKE(0.0, 0.0);
    double c_one = MAGMA_D_MAKE(1.0, 0.0);
    double c_neg_one = MAGMA_D_MAKE(-1.0, 0.0);

    if (m <= 0 || n <= 0) return MAGMA_SUCCESS;

    trans_t transt;
    if (trans == MagmaNoTrans)
        transt = MagmaConjTrans;
    else
        transt = MagmaNoTrans;

    if (side == MagmaLeft) {

        if (storev == MagmaColumnwise) {
            magma_dgemm(MagmaConjTrans, MagmaNoTrans,
                        n, k, m,
                        c_one, dC(dC_offset), ldc,
                        dV(dV_offset), ldv,
                        c_zero, dwork(dwork_offset), ldwork, queue);

            if (direct == MagmaForward)
                magma_dtrmm(MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                            n, k,
                            c_one, dT(dT_offset), ldt,
                            dwork(dwork_offset), ldwork, queue);
            else
                magma_dtrmm(MagmaRight, MagmaLower, transt, MagmaNonUnit,
                            n, k,
                            c_one, dT(dT_offset), ldt,
                            dwork(dwork_offset), ldwork, queue);

            magma_dgemm(MagmaNoTrans, MagmaConjTrans,
                        m, n, k,
                        c_neg_one, dV(dV_offset), ldv,
                        dwork(dwork_offset), ldwork,
                        c_one, dC(dC_offset), ldc, queue);
        } else {
            magma_dgemm(MagmaNoTrans, MagmaConjTrans,
                        m, k, n,
                        c_one, dC(dC_offset), ldc,
                        dV(dV_offset), ldv,
                        c_zero, dwork(dwork_offset), ldwork, queue);

            magma_dtrmm(MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                        m, k,
                        c_one, dT(dT_offset), ldt,
                        dwork(dwork_offset), ldwork, queue);

            magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                        m, n, k,
                        c_neg_one, dwork(dwork_offset), ldwork,
                        dV(dV_offset), ldv,
                        c_one, dC(dC_offset), ldc, queue);
        }
    } else {

        /* Case side == 'R' */
        if (storev == MagmaColumnwise) {
            magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                        m, k, n,
                        c_one, dC(dC_offset), ldc,
                        dV(dV_offset), ldv,
                        c_zero, dwork(dwork_offset), ldwork, queue);
            // ??? ldwork replaced by k for case n < k

            if (direct == MagmaForward)
                magma_dtrmm(MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                            m, k,
                            c_one, dT(dT_offset), ldt,
                            dwork(dwork_offset), ldwork, queue);
            else
                magma_dtrmm(MagmaRight, MagmaLower, transt, MagmaNonUnit,
                            m, k,
                            c_one, dT(dT_offset), ldt,
                            dwork(dwork_offset), ldwork, queue);

            magma_dgemm(MagmaNoTrans, MagmaConjTrans,
                        m, n, k,
                        c_neg_one, dwork(dwork_offset), ldwork,
                        dV(dV_offset), ldv,
                        c_one, dC(dC_offset), ldc, queue);
        } else {
            magma_dgemm(MagmaNoTrans, MagmaConjTrans,
                        m, k, n,
                        c_one, dC(dC_offset), ldc,
                        dV(dV_offset), ldv,
                        c_zero, dwork(dwork_offset), ldwork, queue);

            magma_dtrmm(MagmaRight, MagmaUpper, transt, MagmaNonUnit,
                        m, k,
                        c_one, dT(dT_offset), ldt,
                        dwork(dwork_offset), ldwork, queue);

            magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                        m, n, k,
                        c_neg_one, dwork(dwork_offset), ldwork,
                        dV(dV_offset), ldv,
                        c_one, dC(dC_offset), ldc, queue);
        }
    }

    return MAGMA_SUCCESS;
} /* magma_dlarfb */


void
dsplit_diag_block(int ib, double *a, int lda, double *work)
{
    double *cola, *colw;
    double c_zero = MAGMA_D_ZERO;
    double c_one = MAGMA_D_ONE;

    for (int i = 0; i < ib; ++i) {
        cola = a + i * lda;
        colw = work + i * ib;
        for(int j = 0; j < i; ++j){
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', ib, work, ib);
}


int
dgeqrf_gpu(
        int m, int n,
        cl_mem dA, size_t dA_offset, int ldda,
        double *tau, cl_mem dT, size_t dT_offset,
        viennacl::ocl::context &ctx,
        int *p_info)
{
#define a_ref_dgeqrf(a_1, a_2) dA, (dA_offset + (a_1) + (a_2)*(ldda))
#define t_ref_dgeqrf(a_1) dT, (dT_offset + (a_1)*nb)
#define d_ref_dgeqrf(a_1) dT, (dT_offset + (minmn + (a_1))*nb)
#define dd_ref_dgeqrf(a_1)dT, (dT_offset + (2*minmn+(a_1))*nb)
#define work_ref_dgeqrf(a_1)  ( work + (a_1))
#define hwork_dgeqrf  ( work + (nb)*(m))

    int i, k, minmn, old_i, old_ib, rows, cols;
    int ib, nb;
    int ldwork, lddwork, lwork, lhwork;
    double *work, *ut;

    /* check arguments */
    *p_info = 0;
    if (m < 0) {
        *p_info = -1;
    } else if (n < 0) {
        *p_info = -2;
    } else if (ldda < std::max(1, m)) {
        *p_info = -4;
    }
    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    }

    k = minmn = std::min(m, n);
    if (k == 0)
        return *p_info;

    nb = magma_get_dgeqrf_nb(m);

    lwork = (m + n + nb) * nb;
    lhwork = lwork - m * nb;

    if (MAGMA_SUCCESS != magma_dmalloc_cpu(&work, lwork)) {
        *p_info = MAGMA_ERR_HOST_ALLOC;
        return *p_info;
    }

    ut = hwork_dgeqrf + nb * (n);
    memset(ut, 0, nb * nb * sizeof(double));

    cl_event event[2] = {NULL, NULL};
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_command_queue queue = cq_command_queue();

    ldwork = m;
    lddwork = n;

    if ((nb > 1) && (nb < k)) {
        /* Use blocked code initially */
        old_i = 0;
        old_ib = nb;
        for (i = 0; i < k - nb; i += nb) {
            ib = std::min(k - i, nb);
            rows = m - i;
            magma_dgetmatrix_async(
                    rows, ib,
                    a_ref_dgeqrf(i, i), ldda,
                    work_ref_dgeqrf(i), ldwork, queue, &event[1]);
            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                cols = n - old_i - 2 * old_ib;
                dlarfb_gpu(
                        MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                        m - old_i, cols, old_ib,
                        a_ref_dgeqrf(old_i, old_i), ldda, t_ref_dgeqrf(old_i), nb,
                        a_ref_dgeqrf(old_i, old_i + 2 * old_ib), ldda, dd_ref_dgeqrf(0), lddwork, queue);

                /* store the diagonal */
                magma_dsetmatrix_async(old_ib, old_ib,
                                       ut, old_ib,
                                       d_ref_dgeqrf(old_i), old_ib, queue, &event[0]);
            }

            magma_event_sync(event[1]);
            *p_info = LAPACKE_dgeqrf_work(
                    LAPACK_COL_MAJOR, rows, ib, work_ref_dgeqrf(i), ldwork, tau + i, hwork_dgeqrf,
                    lhwork); // TODO Verify call
            if (*p_info != 0) {
                LOG4_ERROR("Call to LAPACKE_dgeqrf_work failed with " << *p_info);
                return *p_info;
            }

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            *p_info = LAPACKE_dlarft(
                    LAPACK_COL_MAJOR, 'F', 'C', rows, ib,
                    work_ref_dgeqrf(i), ldwork, tau + i, hwork_dgeqrf, ib);
            if (*p_info != 0) {
                LOG4_ERROR("Call to LAPACKE_dlarft failed with " << *p_info);
                return *p_info;
            }

            /* Put 0s in the upper triangular part of a panel (and 1s on the
               diagonal); copy the upper triangular in ut and invert it. */
            magma_event_sync(event[0]);
            dsplit_diag_block(ib, work_ref_dgeqrf(i), ldwork, ut);
            magma_dsetmatrix(rows, ib, work_ref_dgeqrf(i), ldwork, a_ref_dgeqrf(i, i), ldda, queue);

            if (i + ib < n) {
                /* Send the triangular factor T to the GPU */
                magma_dsetmatrix(ib, ib, hwork_dgeqrf, ib, t_ref_dgeqrf(i), nb, queue);

                if (i + nb < k - nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    dlarfb_gpu(
                            MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, ib, ib,
                            a_ref_dgeqrf(i, i), ldda, t_ref_dgeqrf(i), nb,
                            a_ref_dgeqrf(i, i + ib), ldda, dd_ref_dgeqrf(0), lddwork, queue);
                } else {
                    cols = n - i - ib;
                    dlarfb_gpu(
                            MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, cols, ib,
                            a_ref_dgeqrf(i, i), ldda, t_ref_dgeqrf(i), nb,
                            a_ref_dgeqrf(i, i + ib), ldda, dd_ref_dgeqrf(0), lddwork, queue);
                    /* Fix the diagonal block */
                    magma_dsetmatrix(ib, ib, ut, ib, d_ref_dgeqrf(i), ib, queue);
                }
                old_i = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib = n - i;
        rows = m - i;
        magma_dgetmatrix(rows, ib, a_ref_dgeqrf(i, i), ldda, work, rows, queue);
        lhwork = lwork - rows * ib;
        *p_info = LAPACKE_dgeqrf_work(LAPACK_COL_MAJOR, rows, ib, work, rows, tau + i, work + ib * rows,
                                      lhwork); // TODO Verify
        if (*p_info != 0) {
            magma_free_cpu(work);
            LOG4_ERROR("Call to LAPACKE_dgeqrf failed with " << *p_info);
            return *p_info;
        }

        magma_dsetmatrix(rows, ib, work, rows, a_ref_dgeqrf(i, i), ldda, queue);
    }
    magma_free_cpu(work);
    return *p_info;
} /* magma_dgeqrf_gpu */


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
        int *p_info)
{
#define a_ref_dormqr(a_1, a_2) dA, (dA_offset+(a_1)+(a_2)*(ldda))
#define c_ref_dormqr(a_1, a_2) dC, (dC_offset+(a_1)+(a_2)*(lddc))
#define t_ref_dormqr(a_1)     dT, (dT_offset+(a_1)*nb)

    double c_one = MAGMA_D_ONE;

    cl_mem dwork;
    int i, lddwork;

    int i1, i2, i3, ib, ic, jc, mi, ni, nq, nw, ret;
    int left, notran, lquery;
    int lwkopt;

    *p_info = 0;
    left = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);
    lquery = (lwork == -1);

    if (!left || notran)
        printf("dormqr_gpu called with arguments not yet supported\n");

/* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    if (!left && side != MagmaRight) {
        *p_info = -1;
    } else if ((!notran) && (trans != MagmaConjTrans)) {
        *p_info = -2;
    } else if (m < 0) {
        *p_info = -3;
    } else if (n < 0) {
        *p_info = -4;
    } else if (k < 0 || k > nq) {
        *p_info = -5;
    } else if (ldda < std::max(1, nq)) {
        *p_info = -7;
    } else if (lddc < std::max(1, m)) {
        *p_info = -10;
    } else if (lwork < std::max(1, nw) && !lquery) {
        *p_info = -12;
    }

    lwkopt = (m - k + nb) * (n + 2 * nb);
    hwork_[0] = MAGMA_D_MAKE(lwkopt, 0);

    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    } else if (lquery) {
        return *p_info;
    }

/* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        hwork_[0] = c_one;
        return *p_info;
    }

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_command_queue queue = cq_command_queue();

    lddwork = k;
    dwork = dT;
    size_t dwork_offset = 2 * lddwork * nb;

    if ((left && (!notran)) || ((!left) && notran)) {
        i1 = 0;
        i2 = k - nb;
        i3 = nb;
    } else {
        i1 = (k - 1 - nb) / nb * nb;
        i2 = 0;
        i3 = -nb;
    }

    if (left) {
        ni = n;
        jc = 0;
    } else {
        mi = m;
        ic = 0;
    }

    if (nb < k) {
        for (i = i1; i3 < 0 ? i > i2 : i < i2; i += i3) {
            ib = std::min(nb, k - i);
            if (left) {
                mi = m - i;
                ic = i;
            } else {
                ni = n - i;
                jc = i;
            }
            ret = dlarfb_gpu(MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                             mi, ni, ib,
                             a_ref_dgeqrf(i, i), ldda, t_ref_dormqr(i), nb,
                             c_ref_dormqr(ic, jc), lddc, dwork, dwork_offset, nw, queue);
            if (ret != MAGMA_SUCCESS)
                return ret;
        }
    } else {
        i = i1;
    }

/* Use unblocked code to multiply the last or only block. */
    if (i < k) {
        ib = k - i;
        if (left) {
            mi = m - i;
            ic = i;
        } else {
            ni = n - i;
            jc = i;
        }

        magma_dgetmatrix(mi, ib, a_ref_dormqr(i, i), ldda, hwork_, mi, queue);
        magma_dgetmatrix(mi, ni, c_ref_dormqr(ic, jc), lddc, hwork_ + mi * ib, mi, queue);

        int lhwork = lwork - mi * (ib + ni);
        *p_info = LAPACKE_dormqr_work(
                LAPACK_COL_MAJOR, 'L', 'T',
                mi, ni, ib,
                hwork_, mi, tau + i,
                hwork_ + mi * ib, mi,
                hwork_ + mi * (ib + ni), lhwork); // TODO Verify
        if (*p_info != 0) {
            LOG4_ERROR("Call to LAPACKE_dormqr failed with " << *p_info);
            return *p_info;
        }

// send the updated part of c back to the GPU
        magma_dsetmatrix(mi, ni, hwork_ + mi * ib, mi, c_ref_dormqr(ic, jc), lddc, queue);
    }

    return *p_info;
/* End of MAGMA_DORMQR_GPU */
}


int
dgeqrs_gpu(
        const int m, const int n, const int nrhs,
        cl_mem dA, const size_t dA_offset, const int ldda,
        double *tau, cl_mem dT, size_t dT_offset,
        cl_mem dB, const size_t dB_offset, const int lddb,
        double *hwork, const int lwork,
        viennacl::ocl::context &ctx,
        int *p_info)
{
#define a_ref_dgeqrs(a_1, a_2)  dA, (dA_offset + (a_1) + (a_2)*(ldda))
#define d_ref_dgeqrs(a_1)      dT, (dT_offset + (lddwork+(a_1))*nb)

    double c_zero = MAGMA_D_ZERO;
    double c_one = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    cl_mem dwork;
    int i, k, lddwork, rows, ib;
    //const int ione = 1;

    int nb = magma_get_dgeqrf_nb(m);
    int lwkopt = (m - n + nb) * (nrhs + nb) + nrhs * nb;
    int lquery = (lwork == -1);

    hwork[0] = MAGMA_D_MAKE((double) lwkopt, 0.);

    *p_info = 0;
    if (m < 0)
        *p_info = -1;
    else if (n < 0 || m < n)
        *p_info = -2;
    else if (nrhs < 0)
        *p_info = -3;
    else if (ldda < std::max(1, m))
        *p_info = -5;
    else if (lddb < std::max(1, m))
        *p_info = -8;
    else if (lwork < lwkopt && !lquery)
        *p_info = -10;

    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    } else if (lquery)
        return *p_info;

    k = std::min(m, n);
    if (k == 0) {
        hwork[0] = c_one;
        return *p_info;
    }

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl_command_queue queue = cq_command_queue();

/* B := Q' * B */
    dormqr_gpu(
            MagmaLeft, MagmaConjTrans,
            m, nrhs, n,
            a_ref_dgeqrs(0, 0), ldda, tau,
            dB, dB_offset, lddb, hwork, lwork, dT, dT_offset, nb, ctx, p_info);
    if (*p_info != 0) return *p_info;

/* Solve R*X = B(1:n,:) */
    lddwork = k;

    int ldtwork;
    size_t dwork_offset = 0;
    if (nb < k) {
        dwork = dT;
        dwork_offset = dT_offset + 2 * lddwork * nb;
    } else {
        ldtwork = (2 * k + ((n + 31) / 32) * 32) * nb;
        magma_dmalloc(ctx, &dwork, ldtwork);
    }
// To do: Why did we have this line originally; seems to be a bug (Stan)?
//dwork = dT;

    i = (k - 1) / nb * nb;
    ib = n - i;
    rows = m - i;

// TODO: this assumes that, on exit from magma_dormqr_gpu, hwork contains
// the last block of A and B (i.e., C in dormqr). This should be fixed.
// Seems this data should already be on the GPU, so could switch to
// magma_dtrsm and drop the dsetmatrix.
/* TODO Finish porting to MKL calls */
/*
    const MKL_INT mkl_ib = ib;
    const MKL_INT mkl_rows = rows;
    const MKL_INT mkl_ione = ione;
    if ( nrhs == 1 ) dtrsv(
                CblasUpper, CblasNoTrans, CblasNonUnit,
                &mkl_ib, hwork,         &mkl_rows,
                hwork+rows*ib, &mkl_ione);
    else dtrsm(
                CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                &ib, nrhs,
                &c_one, hwork, rows,
                hwork + rows * ib, rows);
*/
// update the solution vector
    magma_dsetmatrix(ib, nrhs, hwork + rows * ib, rows, dwork, dwork_offset + i, lddwork, queue);

// update c
    if (nrhs == 1)
        magma_dgemv(
                MagmaNoTrans, i, ib,
                c_neg_one, a_ref_dgeqrs(0, i), ldda,
                dwork, dwork_offset + i, 1,
                c_one, dB, dB_offset, 1, queue);
    else
        magma_dgemm(
                MagmaNoTrans, MagmaNoTrans,
                i, nrhs, ib,
                c_neg_one, a_ref_dgeqrs(0, i), ldda,
                dwork, dwork_offset + i, lddwork,
                c_one, dB, dB_offset, lddb, queue);

    int start = i - nb;
    if (nb < k) {
        for (i = start; i >= 0; i -= nb) {
            ib = std::min(k - i, nb);
            //rows = m -i;

            if (i + ib < n) {
                if (nrhs == 1) {
                    magma_dgemv(MagmaNoTrans, ib, ib,
                                c_one, d_ref_dgeqrs(i), ib,
                                dB, dB_offset + i, 1,
                                c_zero, dwork, dwork_offset + i, 1, queue);
                    magma_dgemv(MagmaNoTrans, i, ib,
                                c_neg_one, a_ref_dgeqrs(0, i), ldda,
                                dwork, dwork_offset + i, 1,
                                c_one, dB, dB_offset, 1, queue);
                } else {
                    magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                                ib, nrhs, ib,
                                c_one, d_ref_dgeqrs(i), ib,
                                dB, dB_offset + i, lddb,
                                c_zero, dwork, dwork_offset + i, lddwork, queue);
                    magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                                i, nrhs, ib,
                                c_neg_one, a_ref_dgeqrs(0, i), ldda,
                                dwork, dwork_offset + i, lddwork,
                                c_one, dB, dB_offset, lddb, queue);
                }
            }
        }
    }
    magma_dcopymatrix(
            (n), nrhs,
            dwork, dwork_offset, lddwork,
            dB, dB_offset, lddb, queue);
    if (nb >= k) magma_free(dwork);
    magma_queue_sync(queue);
    return *p_info;
}

int
dgels_gpu(
        trans_t trans, int m, int n, int nrhs,
        cl_mem dA, size_t dA_offset, int ldda,
        cl_mem dB, size_t dB_offset, int lddb,
        double *hwork_, int lwork,
        viennacl::ocl::context &ctx,
        int *p_info)
{
    cl_mem dT;
    double *tau;
    int k;

    int nb = magma_get_dgeqrf_nb(m);
    int lwkopt = (m - n + nb) * (nrhs + nb) + nrhs * nb;
    int lquery = (lwork == -1);

    hwork_[0] = MAGMA_D_MAKE((double) lwkopt, 0.);

    *p_info = 0;
/* For now, N is the only case working */
    if (trans != MagmaNoTrans)
        *p_info = -1;
    else if (m < 0)
        *p_info = -2;
    else if (n < 0 || m < n) /* LQ is not handle for now*/
        *p_info = -3;
    else if (nrhs < 0)
        *p_info = -4;
    else if (ldda < std::max(1, m))
        *p_info = -6;
    else if (lddb < std::max(1, m))
        *p_info = -8;
    else if (lwork < lwkopt && !lquery)
        *p_info = -10;

    if (*p_info != 0) {
        LOG4_ERROR("Call failed with " << *p_info);
        return *p_info;
    } else if (lquery) return *p_info;

    k = std::min(m, n);
    if (k == 0) {
        hwork_[0] = MAGMA_D_ONE;
        return *p_info;
    }

/*
 * Allocate temporary buffers
 */
    int ldtwork = (2 * k + ((n + 31) / 32) * 32) * nb;
    if (nb < nrhs)
        ldtwork = (2 * k + ((n + 31) / 32) * 32) * nrhs;
    magma_dmalloc(ctx, &dT, ldtwork);

    const auto err = magma_dmalloc_cpu(&tau, k);
    if (tau == NULL || err != MAGMA_SUCCESS) {
        magma_free(dT);
        *p_info = MAGMA_ERR_HOST_ALLOC;
        return *p_info;
    }

    cl12::command_queue cq_command_queue(ctx.get_queue().handle());

    size_t dT_offset = 0;
    dgeqrf_gpu(m, n, dA, dA_offset, ldda, tau, dT, dT_offset, ctx, p_info);

    if (*p_info == 0)
        dgeqrs_gpu(
                m, n, nrhs,
                dA, dA_offset, ldda, tau, dT, dT_offset,
                dB, dB_offset, lddb, hwork_, lwork, ctx, p_info);

    magma_free(dT);
    magma_free_cpu(tau);
    return *p_info;
}


}
#pragma GCC diagnostic pop
#endif