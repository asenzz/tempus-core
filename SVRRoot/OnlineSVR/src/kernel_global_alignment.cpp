#if 0 // Implementation according to ChatGPT :)

The Global Alignment Kernel (GAK) is a kernel method for time series analysis introduced by Marco Cuturi. It computes the similarity between two time series using dynamic time warping (DTW). Implementing GAK in CUDA can significantly accelerate the computations, especially for large datasets.

Below is an implementation of the Global Alignment Kernel in CUDA.

CUDA Implementation of the Global Alignment Kernel
1. Include Libraries and Define Constants
cpp
Copy code
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define EPSILON 1e-6
2. Define CUDA Kernel for DTW Calculation
The first step is to define the CUDA kernel that will compute the DTW matrix.

cpp
Copy code
__global__ void dtw_kernel(const float* d_X, const float* d_Y, float* d_D, int m, int n, float gamma) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < m * n) {
        int i = tid / n;
        int j = tid % n;

        float cost = expf(-gamma * fabsf(d_X[i] - d_Y[j]));
        if (i == 0 && j == 0) {
            d_D[tid] = cost;
        } else if (i == 0) {
            d_D[tid] = cost + d_D[i * n + (j - 1)];
        } else if (j == 0) {
            d_D[tid] = cost + d_D[(i - 1) * n + j];
        } else {
            d_D[tid] = cost + fminf(fminf(d_D[(i - 1) * n + j], d_D[i * n + (j - 1)]), d_D[(i - 1) * n + (j - 1)]);
        }
    }
}
3. Host Function to Manage Memory and Kernel Launch
Next, write the host function that will allocate memory, copy data to the GPU, launch the kernels, and retrieve the results.

cpp
Copy code
void compute_gak(const std::vector<float>& X, const std::vector<float>& Y, float& similarity, int m, int n, float gamma) {
    float *d_X, *d_Y, *d_D;
    size_t size_X = m * sizeof(float);
    size_t size_Y = n * sizeof(float);
    size_t size_D = m * n * sizeof(float);

    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_Y, size_Y);
    cudaMalloc(&d_D, size_D);

    cudaMemcpy(d_X, X.data(), size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(), size_Y, cudaMemcpyHostToDevice);

    int num_blocks = (m * n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dtw_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_X, d_Y, d_D, m, n, gamma);
    cudaDeviceSynchronize();

    std::vector<float> D(m * n);
    cudaMemcpy(D.data(), d_D, size_D, cudaMemcpyDeviceToHost);

    similarity = D[m * n - 1];

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_D);
}

int main() {
    std::vector<float> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> Y = {1.0, 3.0, 4.0, 2.0};
    float gamma = 1.0;

    float similarity;
    compute_gak(X, Y, similarity, X.size(), Y.size(), gamma);

    std::cout << "Similarity: " << similarity << std::endl;

    return 0;
}
Explanation
CUDA Kernel dtw_kernel:

This kernel computes the dynamic time warping (DTW) matrix. It uses an exponential similarity measure, exp(-gamma * |x_i - y_j|), to fill in the DTW matrix.
The DTW matrix is computed in a parallel fashion by considering the three possible preceding points.
Host Function compute_gak:

This function manages memory allocation and deallocation, copies data between host and device, and launches the CUDA kernel to compute the DTW matrix.
The final similarity score is extracted from the DTW matrix and returned.
Main Function:

Initializes example time series, calls the compute_gak function to calculate the global alignment kernel similarity, and prints the resulting similarity score.
Adjustments and Optimizations
This implementation provides a basic framework. For practical use, you may need to:

Handle larger time series more efficiently.
Optimize memory access patterns.
Consider edge cases and numerical stability.
This implementation follows the principles outlined in Cuturi's paper and adapts them for CUDA to leverage GPU acceleration for time series similarity computation.






#endif

#if 0 // Old GAK implementation by Emo
#include "../include/kernel_global_alignment.hpp"

using namespace svr::datamodel;

namespace svr {

svr::kernel_global_alignment<double>::kernel_values_cache_t kernel_values_cache__;

template<typename scalar_type>
double
kernel_global_alignment<scalar_type>::logGAK_t(
        const vektor<scalar_type> &va,
        const vektor<scalar_type> &vb) // No need to transpose
{
    if (sizeof(scalar_type) != sizeof(double))
        throw std::invalid_argument(
                svr::common::formatter() << "We don't know if it works - scalar_type is not double!");

    const size_t lag_count = this->parameters.get_lag_count();
    if (lag_count <= 0)
        throw std::invalid_argument(
                svr::common::formatter() << "Lag count <= 0 !" << this->parameters.get_lag_count());

    const int dimvect = va.size() / lag_count;
    if (dimvect <= 0)
        throw std::invalid_argument(svr::common::formatter() << "GA dimvect <= 0 !" << dimvect);
    if (va.size() != dtype(va.size())(lag_count * dimvect))
        throw std::invalid_argument(
                svr::common::formatter() << "We don't know what is happening!" << va.size() << " " << lag_count
                                         << " " << dimvect);

    // length of the first vector equals the lag_count
    const auto nX = (int) lag_count;
    // length of the second vector equals the first one in our case
    // but they can be different in general
    const auto nY = nX;
    if (nX == 0 || nY == 0)
        throw std::invalid_argument(
                svr::common::formatter() << "GA sequence has 0 size " << this->parameters.get_lag_count());

    //const int triangular = this->parameters.get_svr_kernel_param2();
    const auto triangular = static_cast<int>(TRIAN * lag_count);
    if (triangular < 0)
        throw std::invalid_argument(svr::common::formatter() << "Invalid GA triangular parameter "
                                                             << this->parameters.get_lag_count());

    const auto lambda = this->parameters.get_svr_kernel_param2();
    if (lambda < 0)
        throw std::invalid_argument(svr::common::formatter() << "Invalid GA lambda parameter "
                                                             << this->parameters.get_svr_kernel_param2());

    int cur, old, curpos, frompos1, frompos2, frompos3;
    curpos = 0;
    double aux, aux1, aux2;

    const int col_len = nY + 1;                /* length of a column for the dynamic programming */
    double gram;
    const double sig =
            -1. / (2. * this->parameters.get_svr_kernel_param() * this->parameters.get_svr_kernel_param());
    /* logM is the array that will stores two successive columns of the (nX+1) x (nY+1) table used to compute the final kernel value*/
    std::vector<double> logM(2 * col_len);
    std::vector<double> logM1(2 * col_len);
    std::vector<double> logM2(2 * col_len);

    const auto trimax = (nX > nY) ? nX - 1 : nY - 1; /* Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY */

    std::vector<double> log_triangular_coefficients(trimax + 1);
    if (triangular > 0) {
        /* initialize */
        auto limit = trimax < triangular ? trimax + 1 : triangular;
        /* cilk_*/ for (dtype(limit) i = 0; i < limit; ++i)log_triangular_coefficients[i] = log(1. - (double) i / (double) triangular);
        for (dtype(limit) i = limit; i <= trimax; ++i)
            log_triangular_coefficients[i] = LOG0; /* Set all to zero */
    } else
        for (int i = 0; i <= trimax; ++i)
            log_triangular_coefficients[i] = 0; /* 1 for all if triangular==0, that is a log value of 0 */


    /****************************************************/
    /* First iteration : initialization of columns to 0 */
    /****************************************************/
    /* The left most column is all zeros... */
    /*cilk_*/for (int j = 1; j < col_len; ++j) {
        logM[j] = LOG0;
        logM1[j] = LOG0;
        logM2[j] = LOG0;
    }
    /* ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0. */
    logM[0] = 0;
    logM1[0] = 0;
    logM2[0] = 0;

    /* Cur and Old keep track of which column is the current one and which one is the already computed one.*/
    cur = 1;      /* Indexes [0..cl-1] are used to process the next column */
    old = 0;      /* Indexes [cl..2*cl-1] were used for column 0 */

    /************************************************/
    /* Next iterations : processing columns 1 .. nX */
    /************************************************/

    /* Main loop to vary the position for i=1..nX */
    for (int i = 1; i <= nX; i++) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur * col_len;                  /* index of the state (i,0) */
        logM[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (int j = 1; j <= nY; ++j) {
            curpos = cur * col_len + j;            /* index of the state (i,j) */
            if (log_triangular_coefficients[abs(i - j)] > LOG0) {
                frompos1 = old * col_len + j;            /* index of the state (i-1,j) */
                frompos2 = cur * col_len + j - 1;          /* index of the state (i,j-1) */
                frompos3 = old * col_len + j - 1;          /* index of the state (i-1,j-1) */

                /* We first compute the kernel value */
                cilk::reducer<cilk::op_add<double>> red_sum(0);
                /*cilk_*/for (int ii = 0; ii < dimvect; ++ii)*red_sum +=
                                                                 (va[i - 1 + ii * nX] - vb[j - 1 + ii * nY]) *
                                                                 (va[i - 1 + ii * nX] - vb[j - 1 + ii * nY]);

                gram = log_triangular_coefficients[abs(i - j)] + sqrt(red_sum.get_value()) * sig;
                gram = gram - log(2. - exp(gram));
                gram = lambda * gram;
                /* Doing the updates now, in two steps. */
                aux = LOGP(logM[frompos1], logM[frompos2]);
                logM[curpos] = LOGP(aux, logM[frompos3]) + gram;
            } else {
                logM[curpos] = LOG0;
            }
        }
        /* Update the column order */
        cur = 1 - cur;
        old = 1 - old;
    }
    aux = logM[curpos]; // curpos = 1 * col_len + nY+1

    cur = 1;
    old = 0;
    for (int i = 1; i <= nX; ++i) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur * col_len;                  /* index of the state (i,0) */
        logM1[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (int j = 1; j <= nY; ++j) {
            curpos = cur * col_len + j;            /* index of the state (i,j) */
            if (log_triangular_coefficients[abs(i - j)] > LOG0) {
                frompos1 = old * col_len + j;            /* index of the state (i-1,j) */
                frompos2 = cur * col_len + j - 1;          /* index of the state (i,j-1) */
                frompos3 = old * col_len + j - 1;          /* index of the state (i-1,j-1) */

                /* We first compute the kernel value */
                cilk::reducer<cilk::op_add<double>> red_sum(0);
                /* cilk_ */ for (int ii = 0; ii < dimvect; ++ii)*red_sum += (va[i - 1 + ii * nX] - va[j - 1 + ii * nY]) *
                                                                     (va[i - 1 + ii * nX] - va[j - 1 + ii * nY]);
                //sum += (va[dimvect*(i - 1 )+ ii] - va[dimvect*(j - 1) + ii]) *
                //(va[dimvect*(i - 1) + ii] - va[dimvect*(j - 1) + ii]);


                gram = log_triangular_coefficients[abs(i - j)] + sqrt(red_sum.get_value()) * sig;
                gram = gram - log(2. - exp(gram));
                gram = lambda * gram;
                //LOG4_DEBUG("Gram log sum sig "<< gram <<" " << logTriangularCoefficients[abs(i - j)]<< " " << sum << " "<< Sig );

                /* Doing the updates now, in two steps. */
                aux1 = LOGP(logM1[frompos1], logM1[frompos2]);
                logM1[curpos] = LOGP(aux1, logM1[frompos3]) + gram;
            } else {
                logM1[curpos] = LOG0;
            }
        }
        /* Update the column order */
        cur = 1 - cur;
        old = 1 - old;
    }
    aux1 = logM1[curpos];

    cur = 1;
    old = 0;
    for (int i = 1; i <= nX; i++) {
        /* Special update for positions (i=1..nX,j=0) */
        curpos = cur * col_len;                  /* index of the state (i,0) */
        logM2[curpos] = LOG0;
        /* Secondary loop to vary the position for j=1..nY */
        for (int j = 1; j <= nY; j++) {
            curpos = cur * col_len + j;            /* index of the state (i,j) */
            if (log_triangular_coefficients[abs(i - j)] > LOG0) {
                frompos1 = old * col_len + j;            /* index of the state (i-1,j) */
                frompos2 = cur * col_len + j - 1;          /* index of the state (i,j-1) */
                frompos3 = old * col_len + j - 1;          /* index of the state (i-1,j-1) */

                /* We first compute the kernel value */
                cilk::reducer<cilk::op_add<double>> red_sum(0);
                /* cilk_ */ for (int ii = 0; ii < dimvect; ++ii)*red_sum += (vb[i - 1 + ii * nX] - vb[j - 1 + ii * nY]) *
                                                                     (vb[i - 1 + ii * nX] - vb[j - 1 + ii * nY]);

                //Can be switched to sqrt(sum) - it will be like the exponential vs rbf
                gram = log_triangular_coefficients[abs(i - j)] + sqrt(red_sum.get_value()) * sig;
                gram = gram - log(2. - exp(gram));
                gram = lambda * gram;
                //LOG4_DEBUG("Gram log sum sig "<< gram <<" " << logTriangularCoefficients[abs(i - j)]<< " " << sum << " "<< Sig );

                /* Doing the updates now, in two steps. */
                aux2 = LOGP(logM2[frompos1], logM2[frompos2]);
                logM2[curpos] = LOGP(aux2, logM2[frompos3]) + gram;
            } else {
                logM2[curpos] = LOG0;
            }
        }
        /* Update the column order */
        cur = 1 - cur;
        old = 1 - old;
    }
    aux2 = logM2[curpos];

    return exp(aux - 0.5 * (aux1 + aux2));
}


template<typename scalar_type>
scalar_type
kernel_global_alignment<scalar_type>::operator()(
        const vektor<scalar_type> &va,
        const vektor<scalar_type> &vb)
{
    LOG4_BEGIN();

    const auto lag_count = this->parameters.get_lag_count();
    if (lag_count <= 0)
        throw std::invalid_argument(
                svr::common::formatter() << "Lag count <= 0 !" << this->parameters.get_lag_count());
    const int dimvect = va.size() / lag_count;
    if (dimvect < 1 || dtype(va.size())(dimvect * lag_count) != va.size())
        throw std::invalid_argument(
                svr::common::formatter() << "GA dimvect <= 0 or otherwise BAD!   dimvect is " << dimvect
                                         << " va.size() is " << va.size() << " and lag_count is " << lag_count);

    const int nX = va.size() / dimvect;
    const int nY = nX;
    if (nX == 0 || nY == 0)
        throw std::invalid_argument(
                svr::common::formatter() << "GA sequence has 0 size !" << this->parameters.get_lag_count());

    const auto triangular = static_cast<int>(TRIAN * lag_count);
    if (triangular < 0)
        throw std::invalid_argument(svr::common::formatter() << "Invalid GA triangular parameter < 0 !"
                                                             << this->parameters.get_lag_count());
    LOG4_END();

    return logGAK_t(va, vb);
}

template<typename scalar_type>
void
kernel_global_alignment<scalar_type>::operator()(
        const viennacl::matrix<scalar_type> &features,
        viennacl::matrix<scalar_type> &p_kernel_matrices) // not used currently
{
    LOG4_BEGIN();

    vmatrix<scalar_type> features_cpu;
    features_cpu.copy_from(features);

    vmatrix<scalar_type> result(features.size1(), features.size1());
    LOG4_WARN("Kernel matrix computation on CPU is very slow!");
    /* cilk_ */ for (ssize_t row = 0; row < features_cpu.get_length_rows(); ++row) {
        /* cilk_ */ for (ssize_t row2 = 0; row2 <= row; ++row2) {
            scalar_type val = this->operator()(features_cpu[row], features_cpu[row2]);
            result.set_value(row, row2, val);
            result.set_value(row2, row, val);
        }
    }
    if (p_kernel_matrices.size1() != features.size1() || p_kernel_matrices.size2() != features.size1())
        p_kernel_matrices.resize(features.size1(), features.size1());

    LOG4_END();

    result.copy_to(p_kernel_matrices);
}

#ifdef VIENNACL_WITH_OPENCL


template<typename scalar_type>
void
kernel_global_alignment<scalar_type>::operator()(
        viennacl::ocl::context &ctx,
        const vektor<double> &features,
        const vmatrix<double> &learning,
        vektor<double> &kernel_values)
{
    LOG4_BEGIN();
#ifdef CACHED_GA_KERNEL
    // Check cache first
    const kernel_cache_key_t kernel_cache_key = std::make_tuple(
            (void *) &features,
            (void *) &learning,
            this->parameters.get_svr_kernel_param(),
            this->parameters.get_svr_kernel_param2());
    if (features.size() > MIN_KERNEL_CACHED_SIZE) {
        const auto iter_kernel_cache = kernel_values_cache__.find(kernel_cache_key);
        if (iter_kernel_cache != kernel_values_cache__.end())
            return iter_kernel_cache->second->copy_to(kernel_values);
    }
#endif /* CACHED_GA_KERNEL */
    //LOG4_DEBUG("Computing GA for features vector with size " << features.size());
    svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "logGAK_OpenCL");
    viennacl::matrix<double> features_gpu(learning.get_length_rows() + 1, learning.get_length_cols(), ctx);
    if (features_gpu.handle().opencl_handle().get() == 0)
        throw std::invalid_argument("Invalid context. featuresp1 is not created on gpu.");

    //for stability, not critical part. will replace and test it later with vmatrix::vienna_clone().
    std::vector<double> temp_copy_vector(features_gpu.internal_size());
    /* cilk_ */ for (unsigned i = 0; i < learning.get_length_rows(); ++i)/* cilk_ */ for (unsigned j = 0; j <
                                                                                            learning.get_length_cols(); ++j)temp_copy_vector[
                                                                                                                                    i *
                                                                                                                                    features_gpu.internal_size2() +
                                                                                                                                    j] = learning.get_value(
                    i, j);
    for (unsigned int j = 0; j < learning.get_length_cols(); ++j)
        temp_copy_vector[learning.get_length_rows() * features_gpu.internal_size2() + j] = features[j];
    viennacl::fast_copy(&(temp_copy_vector[0]), &(temp_copy_vector[0]) + temp_copy_vector.size(), features_gpu);
    const double copied_dif = fabs(features[1] - features_gpu(learning.get_length_rows(), 1));
    if (copied_dif > std::numeric_limits<float>::epsilon())
        throw std::runtime_error(
                svr::common::formatter() << "Copy to GPU error too big " << copied_dif);

    std::vector<double> result(learning.get_length_rows());

    svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "logGAK_OpenCL");
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl12::ocl_context context = ctx.handle();
    if (result.size() != features_gpu.size1() - 1) result.resize(features_gpu.size1() - 1);

    cl_int err = CL_SUCCESS;
    if (features_gpu.handle().opencl_handle().get() == NULL) {
        LOG4_DEBUG("The kernel expects features matrix on the device!");
        throw std::invalid_argument(
                svr::common::formatter() << "NULL pointer "
                                         << size_t(features_gpu.handle().opencl_handle().get()));
    }
    if (sizeof(scalar_type) != sizeof(double))
        throw std::invalid_argument(
                svr::common::formatter() << "We don't know if it works - scalar_type is not double!");
    if (features_gpu.size1() < 1 || features_gpu.size2() < 1)
        throw std::runtime_error(
                svr::common::formatter() << "Empty features matrix passed for GAK computation on GPUs\n");

    const auto sigma = this->parameters.get_svr_kernel_param();
    const auto lag_count = this->parameters.get_lag_count();
    if (lag_count < 1)
        throw std::invalid_argument(
                svr::common::formatter() << "Lag count <= 0 !" << this->parameters.get_lag_count());
    //nX == nY in our case
    const int nX = lag_count, nY = lag_count;
    const int dimvect = features_gpu.size2() / lag_count;
    if (dimvect <= 0)
        throw std::invalid_argument(
                svr::common::formatter() << "GA dimvect <= 0 !" << dimvect);
    const auto triangular = static_cast<int>(TRIAN * lag_count);
    const double lambda = this->parameters.get_svr_kernel_param2();
    if (lambda < 0)
        throw std::invalid_argument(
                svr::common::formatter() << "Invalid GA triangular lambda parameter < 0 !" << lambda);
    const size_t size1 = features_gpu.size1();
    const int cl = nY + 1;                /* length of a column for the dynamic programming */
    cl::Buffer logM_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (2 * cl) * size1, NULL, &err);
    CL_CHECK(err);
    cl::Buffer results_diagonal_gpu_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * size1, NULL, &err);
    CL_CHECK(err);
    cl::Buffer result_matrix_gpu_d;
    const auto result_gpu_d = cl::Buffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(cl_double) * (features_gpu.size1() - 1),
            NULL,
            &err);
    cq_command_queue.finish();
    cl_mem features_gpu_ptr = features_gpu.handle().opencl_handle().get();
    const size_t features_internal_vienna_size2 = features_gpu.internal_size2();
    size_t flagxxxyyy = 0;
    const size_t zero = 0;
    std::vector<double> XX_temp(features_gpu.size1());
    svr::cl12::ocl_kernel forward1(ctx.get_kernel("logGAK_OpenCL", "logGAK_OpenCL_run").handle());

    forward1.set_args(
            features_gpu_ptr,
            nX,
            dimvect,
            sigma,
            triangular,
            lambda,
            features_gpu.size1(),
            features_gpu.size2(),
            features_internal_vienna_size2,
            zero,
            flagxxxyyy,
            zero,
            size1,   //total amount of useful work - we suppose this part always fit in one kernel invocation
            results_diagonal_gpu_d,   // results from this buffer will be used later in the other buffer
            result_gpu_d, //not used
            logM_d);
    err = forward1.enqueue(cq_command_queue, cl::NullRange, cl::NDRange(size1), cl::NullRange, NULL, NULL);
    CL_CHECK(err);
    cq_command_queue.finish();
    err = cq_command_queue.enqueueReadBuffer(
            results_diagonal_gpu_d, CL_TRUE, 0, sizeof(cl_double) * size1, XX_temp.data());
    CL_CHECK(err);
    err = cq_command_queue.finish();
    CL_CHECK(err);

//        size_t memory_acceptable =
//                svr::common::gpu_handler_hid::get().get_max_gpu_data_chunk_size() / sizeof(double) * 0.4;
    const size_t num_kernels = features_gpu.size1() - 1;

    cl::Buffer logM_XY_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (2 * cl) * num_kernels, NULL, &err);
    CL_CHECK(err);

    flagxxxyyy = 2;
    svr::cl12::ocl_kernel forward2(ctx.get_kernel("logGAK_OpenCL", "logGAK_OpenCL_run").handle());
    //cl_mem kernel_gpu_ptr=NULL;
    err = forward2.set_args(
            features_gpu_ptr,
            nX,
            dimvect,
            sigma,
            triangular,
            lambda,
            features_gpu.size1(),
            features_gpu.size2(),
            features_internal_vienna_size2,
            zero,
            flagxxxyyy,
            zero,
            size1 - 1, //total amount of useful work
            results_diagonal_gpu_d,
            result_gpu_d,
            logM_XY_d).enqueue(
            cq_command_queue, cl::NullRange, cl::NDRange(num_kernels), cl::NullRange, NULL, NULL);
    CL_CHECK(err);
    err = cq_command_queue.finish();
    CL_CHECK(err);
    err = cq_command_queue.enqueueReadBuffer(
            result_gpu_d, CL_TRUE, 0,
            sizeof(cl_double) * (size1 - 1),
            result.data());
    CL_CHECK(err);
    err = cq_command_queue.finish();
    CL_CHECK(err);
    kernel_values.resize(result.size());
    for (size_t i = 0; i < result.size(); ++i) kernel_values[i] = result[i];
#ifdef CACHED_GA_KERNEL
    if (features.size() <= MIN_KERNEL_CACHED_SIZE) return;
    if (kernel_values_cache__.size() > MAX_KERNEL_CACHE_SIZE)
        kernel_values_cache__.erase(kernel_values_cache__.begin());
    kernel_values_cache__.insert(
            kernel_values_cache__.end(), {kernel_cache_key, ptr<vektor<double>>(kernel_values)});
#endif
}


template<typename scalar_type>
void
kernel_global_alignment<scalar_type>::operator()(
        viennacl::ocl::context &ctx,
        const viennacl::matrix<scalar_type> &features,
        viennacl::matrix<scalar_type> &p_kernel_matrices)
{
    LOG4_BEGIN();
    svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "logGAK_OpenCL");
    cl12::command_queue cq_command_queue(ctx.get_queue().handle());
    cl12::ocl_context context = ctx.handle();
    LOG4_DEBUG(
            "Starting CPU side of OpenCL when triangular is " << TRIAN << ", gamma is "
                                                              << this->parameters.get_svr_kernel_param()
                                                              << ", lambda is "
                                                              << this->parameters.get_svr_kernel_param2());

    if (p_kernel_matrices.size1() != features.size1() || p_kernel_matrices.size1() != features.size1())
        p_kernel_matrices.resize(features.size1(), features.size1());

    cl_int err = CL_SUCCESS;
    if (features.handle().opencl_handle().get() == NULL) {
        LOG4_DEBUG("The kernel expects features matrix on the device!");
        throw std::invalid_argument(
                svr::common::formatter() << "NULL pointer " << (size_t) (features.handle().opencl_handle().get()));
    }

    if (sizeof(scalar_type) != sizeof(double))
        throw std::invalid_argument(
                svr::common::formatter() << "We don't know if it works - scalar_type is not double!");

    if (features.size1() < 1)
        throw std::runtime_error(
                svr::common::formatter() << "Empty features matrix passed for GAK computation on GPUs\n");

    const auto sigma = this->parameters.get_svr_kernel_param();
    const size_t lag_count = this->parameters.get_lag_count();
    if (lag_count <= 0)
        throw std::invalid_argument(
                svr::common::formatter() << "Lag count <= 0 !" << this->parameters.get_lag_count());

    //nX == nY in our case
    const int nX = lag_count, nY = lag_count;
    const int dimvect = features.size2() / lag_count;
    if (dimvect <= 0) throw std::invalid_argument(svr::common::formatter() << "GA dimvect <= 0 !" << dimvect);

    const int triangular = static_cast<int>(TRIAN * lag_count);
    const double lambda = this->parameters.get_svr_kernel_param2();
    if (lambda < 0)
        throw std::invalid_argument(
                svr::common::formatter() << "Invalid GA triangular lambda parameter < 0 !" << lambda);

    const auto size1 = features.size1();
    const auto size1_sqr = size1 * size1;
    const int cl = nY + 1;                /* length of a column for the dynamic programming */
    cl::Buffer logM_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (2 * cl) * size1, NULL, &err);
    CL_CHECK(err);

    cl::Buffer results_diagonal_gpu_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * size1, NULL, &err);
    CL_CHECK(err);

    cl_mem kernel_gpu_ptr;
    cl::Buffer kernel_matrix_gpu_d;
    if (p_kernel_matrices.handle().ram_handle().get() != NULL) {
        kernel_matrix_gpu_d = cl::Buffer(
                context,
                CL_MEM_READ_WRITE,
                sizeof(cl_double) * p_kernel_matrices.size1() * p_kernel_matrices.internal_size2(),
                NULL,
                &err);
        kernel_gpu_ptr = kernel_matrix_gpu_d();
    } else
        kernel_gpu_ptr = p_kernel_matrices.handle().opencl_handle().get();
    cq_command_queue.finish();

    cl_mem features_gpu_ptr = features.handle().opencl_handle().get();
    size_t features_internal_vienna_size2 = features.internal_size2();
    size_t kernel_matrix_internal_vienna_size2 = p_kernel_matrices.internal_size2();

    size_t flagxxxyyy = 0;
    std::vector<double> XX_temp(features.size1());
    svr::cl12::ocl_kernel forward1(ctx.get_kernel("logGAK_OpenCL", "logGAK_OpenCL_run").handle());
    size_t zero = 0;
    forward1.set_args(
            features_gpu_ptr,
            nX,
            dimvect,
            sigma,
            triangular,
            lambda,
            features.size1(),
            features.size2(),
            features_internal_vienna_size2,
            kernel_matrix_internal_vienna_size2,
            flagxxxyyy,
            zero,
            size1,   //total amount of useful work - we suppose this part always fit in one kernel invocation
            results_diagonal_gpu_d,   // results from this buffer will be used later in the other buffer
            kernel_gpu_ptr,
            logM_d);

    err = forward1.enqueue(cq_command_queue, cl::NullRange, cl::NDRange(size1), cl::NullRange, NULL, NULL);
    CL_CHECK(err);
    cq_command_queue.finish();
    err = cq_command_queue.enqueueReadBuffer(
            results_diagonal_gpu_d, CL_TRUE, 0, sizeof(cl_double) * size1, XX_temp.data());
    CL_CHECK(err);
    err = cq_command_queue.finish();
    CL_CHECK(err);

    const size_t memory_acceptable =
            svr::common::gpu_handler_hid::get().get_max_gpu_data_chunk_size() / sizeof(double) * 0.4;
    size_t num_kernels = std::min(std::min(
            size_t(0.5 * memory_acceptable / cl),
            svr::common::gpu_handler_hid::get().get_max_gpu_kernels()),
                                  size1_sqr);
    cl::Buffer logM_XY_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * 2 * cl * num_kernels, NULL, &err);
    CL_CHECK(err);

    flagxxxyyy = 1;
    svr::cl12::ocl_kernel forward2(ctx.get_kernel("logGAK_OpenCL", "logGAK_OpenCL_run").handle());
    //cl_mem kernel_gpu_ptr=NULL;
    for (size_t offset = 0; offset < size1_sqr; offset += num_kernels) {
        const size_t useful_work = offset + num_kernels >= size1_sqr ? size1_sqr - offset : num_kernels;
        err = forward2.set_args(
                features_gpu_ptr,
                nX,
                dimvect,
                sigma,
                triangular,
                lambda,
                features.size1(),
                features.size2(),
                features_internal_vienna_size2,
                kernel_matrix_internal_vienna_size2,
                flagxxxyyy,
                offset,
                useful_work, //total amount of useful work
                results_diagonal_gpu_d,
                kernel_gpu_ptr,
                logM_XY_d).enqueue(
                cq_command_queue,
                cl::NullRange,
                cl::NDRange(num_kernels),
                cl::NullRange,
                NULL, NULL);
        CL_CHECK(err);
        err = cq_command_queue.finish();
        CL_CHECK(err);
    }

    if (p_kernel_matrices.handle().ram_handle().get() == NULL) {
        LOG4_DEBUG("Matrix produced on GPU as requested");
    } else {
        err = cq_command_queue.enqueueReadBuffer(
                kernel_matrix_gpu_d,
                CL_TRUE,
                0,
                sizeof(cl_double) * p_kernel_matrices.size1() * p_kernel_matrices.internal_size2(),
                p_kernel_matrices.handle().ram_handle().get());
        CL_CHECK(err);
        err = cq_command_queue.finish();
        CL_CHECK(err);
        LOG4_DEBUG("Matrix produced on CPU as requested");
    }

    LOG4_END();
}


#endif /* #ifdef VIENNACL_WITH_OPENCL */

}
#endif