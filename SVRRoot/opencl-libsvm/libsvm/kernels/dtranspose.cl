/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from ztranspose.cl normal z -> d, Sat Nov 15 00:21:35 2014

       auto-converted from dtranspose.cu

       @author Stan Tomov
       @author Mark Gates
*/

#define PRECISION_d

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 16
#define NY 8


// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
// for each subtile
//     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
//     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
//     A  += NX
//     AT += NX*ldat
//
// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
//     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
//     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
//
// e.g., with NB=32, NX=16, NY=8 (z precision)
//     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
//     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
//                                                       (AT21 AT22)
__kernel void
dtranspose_kernel(
    int m, int n,
    const __global double *A, unsigned long A_offset, int lda,
    __global double *AT, unsigned long AT_offset,      int ldat)
{
    A += A_offset;
    AT += AT_offset;

    __local double sA[NB][NX+1];

    int tx  = get_local_id(0);
    int ty  = get_local_id(1);
    int ibx = get_group_id(0)*NB;
    int iby = get_group_id(1)*NB;
    int i, j;
    
    A  += ibx + tx + (iby + ty)*lda;
    AT += iby + tx + (ibx + ty)*ldat;
    
UNROLL()
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from A into sA
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
UNROLL()
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sA[ty + j2][tx] = A[j2*lda];
                }
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        
        // save NB-by-NX subtile from sA into AT
        i = iby + tx;
        j = ibx + ty + tile*NX;
UNROLL()
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
UNROLL()
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        AT[i2 + j2*ldat] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        
        // move to next subtile
        A  += NX;
        AT += NX*ldat;
    }
}
