//
// Created by sstoyanov on 12/27/18.
//

//To use type_float precision in kernel
//On NVIDIA GPU
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//On AMD GPU
#pragma OPENCL EXTENSION cl_amd_fp64 : enable


#define maximum(a, b) (((a) < (b)) ? (b) : (a))
#define minimum(a, b) (((a) < (b)) ? (a) : (b))



/* Useful constants */
#define LOG0 -10000          /* log(0) */
#define LOGP(x, y) (((x)>(y))?(x)+log1p(exp((y)-(x))):(y)+log1p(exp((x)-(y))))



//It seems it is not possible to allocate a dynamic array in a kernel when array's size is not known before kernel compilation.
//That is why logM is used for scratch space - 2*cl*sizeof(double) for each invocation


__kernel void
logGAK_OpenCL_run(__global double *seq1, __global double *seq2, const int nX, const int dimvect, const double sigma,
                  const int triangular, const double lambda, const unsigned long size1,
                  const unsigned long size2,
                  const unsigned long features_internal_vienna_size2,
                  const unsigned long kernel_matrix_internal_vienna_size2,
                  const unsigned long flagxxxyyy,
                  const unsigned long offset,
                  const unsigned long work_chunks,//work_chunks can be less than the total number of started kernels
                  __global double *results_diagonal_GPU_d,
                  __global double *kernel_matrix_GPU_d,
                  __global double *logM)


/* Implementation of the (Triangular) global alignment kernel.
 *
 * See details about the matlab wrapper mexFunction below for more information on the inputs that need to be called from Matlab
 *
 * seq1 is the first sequence represented as a matrix of real elements. Each line i corresponds to the vector of observations at time i.
 * seq2 is the second sequence formatted in the same way.
 * nX, nY and dimvect provide the number of lines of seq1 and seq2.
 * sigma stands for the bandwidth of the \phi_\sigma distance used kernel
 * triangular is a parameter which parameterizes the triangular kernel
 * lambda is an additional factor that can be used with the Geometrically divisible Gaussian Kernel
 * kerneltype selects either the Gaussian Kernel or its geometrically divisible equivalent

 work_chunks - size of chunk with useful input data, can be less than number of kernels.
 */

{
        __private const long i_kernel = get_global_id(0);

        __private const unsigned long index = i_kernel + offset;
        __private const int local_offset1 = index / size2;
        __private const int local_offset2 = index % size2;
        //size1 and size2 are different only when calling with two matrices
        //__private const bool to_compute_or_not= (flagxxxyyy==1)?((local_offset1<=local_offset2)*(local_offset1<size1)*(local_offset2<size1)): i_kernel<size1;
        __private bool to_compute_or_not = false;
        if (flagxxxyyy == 0)
                to_compute_or_not = i_kernel < size1;
        else if (flagxxxyyy == 1)
                to_compute_or_not = (local_offset1 <= local_offset2) * (local_offset1 < size1) * (local_offset2 < size1);
        else if (flagxxxyyy == 2)
                to_compute_or_not = i_kernel < (size1 - 1);
        else if (flagxxxyyy == 3)
                to_compute_or_not = (local_offset1 < size1) * (local_offset2 < size2);

        //Number of threads in all GPU kernel and shifted according calculated group/Compute Unit.
        int i, j, ii, cur, old, curpos, frompos1, frompos2, frompos3;
        double aux;
        int nY = nX;
        int cl = nY + 1;                /* length of a column for the dynamic programming */

        double sum = 0;
        double gram, Sig;

        int trimax = nX - 1; /* Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY */

        //MAIN ASSUMPTION!!!!!!
        //!!! This array depends on triangular and trimax. When they are constant for all kernels the array doesn't depend on seq1 or seq2 => is constant.
        Sig = -1. / (2. * sigma * sigma);

        //Add by me to remove array logTriangularCoefficients. Tested for triangular==0, that is obvious.
        int logTriangularCoefficients_i_condition_const = ((trimax < triangular) ? trimax + 1 : triangular);
        #define logTriangularCoefficients_condition(i) ((triangular==0)?0:(((i)<logTriangularCoefficients_i_condition_const) ? log(1.-(double)(i)/(double)triangular) : LOG0))
        /****************************************************/
        /* First iteration : initialization of columns to 0 */
        /****************************************************/
        /* The left most column is all zeros... */


        barrier(CLK_LOCAL_MEM_FENCE);
        if (to_compute_or_not) {
                for (j = 1; j < cl; j++) {
                        logM[j * work_chunks + i_kernel] = LOG0;
                }
                /* ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0. */
                logM[0 * work_chunks + i_kernel] = 0.;
        }

        long seq1ptr = (flagxxxyyy == 0 || flagxxxyyy == 2) ? features_internal_vienna_size2 * index :
                       features_internal_vienna_size2 * local_offset1;
        long seq2ptr = 0;
        if (flagxxxyyy == 0)
                seq2ptr = features_internal_vienna_size2 * index;
        else if (flagxxxyyy == 1)
                seq2ptr = features_internal_vienna_size2 * local_offset2;
        else if (flagxxxyyy == 2)
                seq2ptr = features_internal_vienna_size2 * (size1 - 1);
        else if (flagxxxyyy == 3)
                seq2ptr = features_internal_vienna_size2 * local_offset2;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* Cur and Old keep track of which column is the current one and which one is the already computed one.*/
        cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
        old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */

        /************************************************/
        /* Next iterations : processing columns 1 .. nX */
        /************************************************/

        barrier(CLK_LOCAL_MEM_FENCE);
        /* Main loop to vary the position for i=1..nX */
        for (i = 1; i <= nX; i++) {
                if (to_compute_or_not) {
                        /* Special update for positions (i=1..nX,j=0) */
                        curpos = cur * cl;                  /* index of the state (i,0) */
                        logM[curpos * work_chunks + i_kernel] = LOG0;
                        /* Secondary loop to vary the position for j=1..nY */
                        for (j = 1; j <= nY; j++) {
                                curpos = cur * cl + j;            /* index of the state (i,j) */

                                double logTriangularCoefficients_condition_abs_i_j_result = logTriangularCoefficients_condition(
                                        abs(i - j));
                                if (logTriangularCoefficients_condition_abs_i_j_result > LOG0) {
                                        frompos1 = old * cl + j;            /* index of the state (i-1,j) */
                                        frompos2 = cur * cl + j - 1;          /* index of the state (i,j-1) */
                                        frompos3 = old * cl + j - 1;          /* index of the state (i-1,j-1) */


                                        /* We first compute the kernel value */
                                        sum = 0;
                                        if (flagxxxyyy == 1 || flagxxxyyy == 2 || flagxxxyyy == 3) {
                                                for (ii = 0; ii < dimvect; ii++) {
                                                        double sum_seq1_and_seq2 = (seq1[seq1ptr + ((i - 1) + ii * nX)] -
                                                                                    seq2[seq2ptr + ((j - 1) + ii * nX)]);
                                                        sum += sum_seq1_and_seq2 * sum_seq1_and_seq2;
                                                }
                                        }
                                        if (flagxxxyyy == 0) {
                                                for (ii = 0; ii < dimvect; ii++) {
                                                        double sum_seq1_and_seq2 = (seq1[seq1ptr + ((i - 1) + ii * nX)] -
                                                                                    seq1[seq1ptr + ((j - 1) + ii * nX)]);
                                                        sum += sum_seq1_and_seq2 * sum_seq1_and_seq2;
                                                }
                                        }
                                        //use sqrt both here and in the CPU code to get closer to exponential kernel
                                        gram = logTriangularCoefficients_condition_abs_i_j_result + sqrt(sum) * Sig;
                                        gram -= log(2. - exp(gram));
                                        gram = lambda * gram;

                                        /* Doing the updates now, in two steps. */
                                        aux = LOGP(logM[frompos1 * work_chunks + i_kernel], logM[frompos2 * work_chunks + i_kernel]);
                                        logM[curpos * work_chunks + i_kernel] = LOGP(aux, logM[frompos3 * work_chunks + i_kernel]) + gram;
                                } else {
                                        logM[curpos * work_chunks + i_kernel] = LOG0;
                                }
                        }
                        /* Update the column order */
                        cur = 1 - cur;
                        old = 1 - old;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

        }

        if (to_compute_or_not) {
                //Write output data.
                if (flagxxxyyy == 0) {
                        results_diagonal_GPU_d[i_kernel] = logM[curpos * work_chunks + i_kernel];
                } else if (flagxxxyyy == 1) {
                        double current_result = logM[curpos * work_chunks + i_kernel];
                        double result_x = results_diagonal_GPU_d[local_offset1];
                        double result_y = results_diagonal_GPU_d[local_offset2];
                        double full_result = exp(current_result - 0.5 * (result_x + result_y));
                        kernel_matrix_GPU_d[local_offset1 * kernel_matrix_internal_vienna_size2 + local_offset2] = full_result;
                        kernel_matrix_GPU_d[local_offset2 * kernel_matrix_internal_vienna_size2 + local_offset1] = full_result;
                } else if (flagxxxyyy == 2) {
                        double current_result = logM[curpos * work_chunks + i_kernel];
                        double result_x = results_diagonal_GPU_d[index];
                        double result_y = results_diagonal_GPU_d[size1 - 1];
                        double full_result = exp(current_result - 0.5 * (result_x + result_y));
                        kernel_matrix_GPU_d[i_kernel] = full_result;
                } else if (flagxxxyyy == 3) {
                        double current_result = logM[curpos * work_chunks + i_kernel];
                        double result_x = results_diagonal_GPU_d[local_offset1];
                        double result_y = results_diagonal_GPU_d[size1 + local_offset2];
                        //double full_result = current_result ;
                        double full_result = exp(current_result - 0.5 * (result_x + result_y));
                        kernel_matrix_GPU_d[local_offset1 * kernel_matrix_internal_vienna_size2 + local_offset2] = full_result;
                }
        }
}
