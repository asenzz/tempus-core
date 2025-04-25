/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from ztranspose_inplace.cl normal z -> d, Sat Nov 15 00:21:35 2014

       auto-converted from dtranspose_inplace.cu

       @author Stan Tomov
       @author Mark Gates
*/

typedef long magma_int_t;  // MKL uses long long int, not int64_t

#define PRECISION_d

#define NB 16

////////////////////////////////////////////////////////////////////////////////
// grid is (n/nb) x ((n/nb)/2 + 1), where n/nb is odd.
// lower indicates blocks in lower triangle of grid, including diagonal.
// lower blocks cover left side of matrix, including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width (or width-1)
// to cover right side of matrix.
//      [ A00 A01 A02 ]                  [ A00  .   .  |  .   .  ]
//      [ A10 A11 A12 ]                  [ A10 A11  .  |  .   .  ]
// grid [ A20 A21 A22 ] covers matrix as [ A20 A21 A22 |  .   .  ]
//      [ A30 A31 A32 ]                  [ A30 A31 A32 | A01  .  ]
//      [ A40 A41 A42 ]                  [ A40 A41 A42 | A02 A12 ]
//
// See dtranspose_inplace_even for description of threads.

__kernel void dtranspose_inplace_odd( int n, __global double *matrix, uint64_t matrix_offset, int lda )
{
    matrix += matrix_offset;

    __local double sA[ NB ][ NB+1 ];
    __local double sB[ NB ][ NB+1 ];

    int i = get_local_id(0);
    int j = get_local_id(1);

    bool lower = (get_group_id(0) >= get_group_id(1));
    int ii = (lower ? get_group_id(0) : (get_group_id(1) + get_num_groups(1) - 1));
    int jj = (lower ? get_group_id(1) : (get_group_id(0) + get_num_groups(1)    ));

    ii *= NB;
    jj *= NB;

    __global double *A = matrix + ii+i + (jj+j)*lda;
    if( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        __global double *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = *B;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// grid is ((n/nb) + 1) x (n/nb)/2, where n/nb is even.
// lower indicates blocks in strictly lower triangle of grid, excluding diagonal.
// lower blocks shift up by one to cover left side of matrix including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width
// to cover right side of matrix.
//      [ A00  A01 ]                  [ A10  .  |  .   .  ]
//      [ A10  A11 ]                  [ A20 A21 |  .   .  ]
// grid [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
//      [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
//      [ A40  A41 ]
//
// Each block is NB x NB threads.
// For non-diagonal block A, block B is symmetric block.
// Thread (i,j) loads A(i,j) into sA(j,i) and B(i,j) into sB(j,i), i.e., transposed,
// syncs, then saves sA(i,j) to B(i,j) and sB(i,j) to A(i,j).
// Threads outside the matrix do not touch memory.

__kernel void dtranspose_inplace_even( int n, __global double *matrix, uint64_t matrix_offset, int lda )
{
    matrix += matrix_offset;

    __local double sA[ NB ][ NB+1 ];
    __local double sB[ NB ][ NB+1 ];

    int i = get_local_id(0);
    int j = get_local_id(1);

    bool lower = (get_group_id(0) > get_group_id(1));
    int ii = (lower ? (get_group_id(0) - 1) : (get_group_id(1) + get_num_groups(1)));
    int jj = (lower ? (get_group_id(1)    ) : (get_group_id(0) + get_num_groups(1)));

    ii *= NB;
    jj *= NB;

    __global double *A = matrix + ii+i + (jj+j)*lda;
    if( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        __global double *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = *B;
        }
        barrier( CLK_LOCAL_MEM_FENCE );
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}
