#include <cmath>
#include <vector>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>


#include <cublasLt.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>


#include "fast_functions.hpp"

#define MAX_LEN 720

#define XBLOCK 256
#define YBLOCK 256

double ctimes[10];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__  void old_gpu_kernel_xx_compute(size_t sizeX, int len, int dim, CPTRd X, double *Z, size_t full_sizeZ, double param1, double param2, double param3)
{
    double gamma = param1;
    double gamma_mult = 2. * gamma * gamma;
    double beta = param2;
    double tau = param3;
    int total_len_features = len * dim;

    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double powers[MAX_LEN];
    if (len > MAX_LEN) return;//abort? 720 is the max of len. Usually len ==720
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        powers[i] = pow(1. / (double) (len - i), 2 * beta);
    }
    __syncthreads();

    for (auto i = tid % XBLOCK; i < sizeX; i += XBLOCK) {
        for (auto j = ((tid / XBLOCK) % YBLOCK); j <= i; j += YBLOCK) {
            double s = 0;
            for (int k = 0; k < dim; k++) {
                double bs_sum = 0.;
                for (int m = 0; m < len; m++) {
                    //bs_sum += (X[m+len*i] - X[m+len*j])*(X[m+len*i] - X[m+len*j])*pow(1./(double)(len-m),2*beta);
                    bs_sum += (X[m + k * len + total_len_features * i] - X[m + k * len + total_len_features * j]) *
                              (X[m + k * len + total_len_features * i] - X[m + k * len + total_len_features * j]) * powers[m];
                }
                double cum_sum1 = 0.;
                double cum_sum2 = 0.;
                for (int m = len - 1; m >= 0; m--) {
                    cum_sum1 += X[m + len * k + total_len_features * i];
                    cum_sum2 += X[m + len * k + total_len_features * j];
                    bs_sum += (cum_sum1 - cum_sum2) * (cum_sum1 - cum_sum2) * powers[m] * tau;
                }
                s += bs_sum;
            }
            //Z[i*sizeX+j]=1-s/gamma_mult;
            //Z[j*sizeX+i]=1-s/gamma_mult;
            Z[i * sizeX + j] = s;
            Z[j * sizeX + i] = s;
        }
    }
}

#define TILE_WIDTH 16

#define blockX(i, j) (X[(startX + (i)) * total_len_features+(j)])
#define blockY(i, j) (X[(startY + (i)) * total_len_features+(j)])
#define blockYY(i, j) (Y[(startY + (i)) * total_len_features+(j)])


__global__  void
gpu_kernel_xx_compute(size_t sizeX, size_t startX, size_t startY, size_t numX, size_t numY, int len, int dim, CPTRd X, double *Z, size_t full_sizeZ,
                      double param1, double param2, double param3, double param4)
{
    //double sigma = param1; - not used when only computing distance
    double lambda = param2;
    double tau = param3;
    double w_sum_sym = param4;
    int total_len_features = len * dim;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH];//for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH];//for index-1
    int kk = threadIdx.x + blockIdx.x * blockDim.x;
    int mm = threadIdx.y + blockIdx.y * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __syncthreads();
    if ((blockIdx.x * blockDim.x < numX) && (blockIdx.y * blockDim.y < numY) &&
        ((startX + blockIdx.x * blockDim.x) <= (startY + blockIdx.y * blockDim.y + blockDim.y - 1))) {
        int kk_internal = 0;
        double matrix_prod_sum = 0.0;
        for (int jA = 0; jA < dim; ++jA) {
            double s_mm = 0;
            for (int kk_internal_big = 0; kk_internal_big < len / TILE_WIDTH + (len % TILE_WIDTH == 0 ? 0 : 1); ++kk_internal_big) {
                if (tx == 0) {
                    if (ty + kk_internal_big * TILE_WIDTH < len) {
                        power_mult[ty] = pow(1. / ((double) (len - (ty + kk_internal_big * TILE_WIDTH))), 2 * lambda) * w_sum_sym;
                    }
                }
                if ((kk < numX) * (TILE_WIDTH * kk_internal_big + ty < len)) {
                    ta[tx][ty] = blockX(kk, TILE_WIDTH * kk_internal_big + ty + jA * len);
                    if (TILE_WIDTH * kk_internal_big + ty > 0) {
                        tam1[tx][ty] = ta[tx][ty] - blockX(kk, TILE_WIDTH * kk_internal_big + ty - 1 + jA * len);
                    }
                }
                if ((mm < numY) * (TILE_WIDTH * kk_internal_big + tx < len)) {
                    tb[ty][tx] = blockY(mm, TILE_WIDTH * kk_internal_big + tx + jA * len);
                    if (TILE_WIDTH * kk_internal_big + tx > 0) {
                        tbm1[ty][tx] = tb[ty][tx] - blockY(mm, TILE_WIDTH * kk_internal_big + tx - 1 + jA * len);
                    }
                }

                __syncthreads();

                if ((kk < numX) && (mm < numY) && (startX + kk <= startY + mm)) {
                    for (int kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
                        kk_internal = kk_internal_small + kk_internal_big * TILE_WIDTH;
                        if (kk_internal < len) {
                            //mm_internal = kk_internal;
                            double x_y = ta[tx][kk_internal_small] - tb[ty][kk_internal_small];

                            double t_left = x_y * x_y;
                            double t_right = 0;
                            if (kk_internal > 0) {
                                double diff_x_y = 0;
                                diff_x_y = tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small];
                                t_right = diff_x_y * diff_x_y;

                            }
/*
                for (mm_internal = max(kk_internal - cut_len, 0);
                     mm_internal < min(len, kk_internal + cut_len + 1); ++mm_internal) {
                    if (mm_internal == kk_internal) continue;//already computed above
                    double x_y = blockX(kk, (kk_internal + jA * len)) - blockY(mm, (mm_internal + jA * len));
                    x_y = x_y * x_y / w_sum_sym + delta_right * delta_right * 4. * w(kk_internal - mm_internal);
                    t_left = fmin(t_left, x_y);
                    x_y = blockX(kk, (mm_internal + jA * len)) - blockY(mm, (kk_internal + jA * len));
                    x_y = x_y * x_y / w_sum_sym + delta_left * delta_left * 4. * w(kk_internal - mm_internal);
                    t_right = fmin(t_right, x_y);
                }
*/
                            //if ((kk==0) && (mm==0)) printf("tau is %lf\n",tau);
                            //s_mm += (t_left+tau*t_right) * power_mult[kk_internal_small];
                            //s_mm += (t_left+tau*t_right) * power_mult[kk_internal_small];
                            s_mm += (t_left + tau * t_right) * power_mult[kk_internal_small];
                        }//end if kk_internal
                    }//end cycle kk_internal_small
                }//end if check kk and mm inside
                __syncthreads();
            }
            matrix_prod_sum += s_mm;
        }
        if ((kk < numX) && (mm < numY) && (startX + kk <= startY + mm)) {
#if 0
            Z[(startX + kk) * sizeX + (startY + mm)] = (1. - matrix_prod_sum / (2. * sigma * sigma)+pow(matrix_prod_sum/(2*sigma*sigma),2)/2.);
            if (startX + kk < startY + mm ) {
                Z[(startY + mm) * sizeX + (startX + kk)] = (1. - matrix_prod_sum / (2. * sigma * sigma)+pow(matrix_prod_sum/(2*sigma*sigma),2)/2.);
        }
#else
            //Z[(startX + kk) * sizeX + (startY + mm)] = (1. - matrix_prod_sum / (2. * sigma * sigma));
            Z[(startX + kk) * sizeX + (startY + mm)] = matrix_prod_sum;
            if (startX + kk < startY + mm) {
                Z[(startY + mm) * sizeX + (startX + kk)] = matrix_prod_sum;
            }

#endif
        }
    }//end if  group_id 
}


__global__  void
gpu_kernel_xy_compute(size_t sizeX, size_t sizeY, size_t startX, size_t startY, size_t numX, size_t numY, int len, int dim, CPTRd X, const double *Y, double *Z,
                      size_t full_sizeZ, double param1, double param2, double param3, double param4)
{

    //double sigma = param1; - not used when only computing distance!
    double lambda = param2;
    double tau = param3;
    double w_sum_sym = param4;
    int total_len_features = len * dim;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH];//for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH];//for index-1

    int kk = threadIdx.x + blockIdx.x * blockDim.x;
    int mm = threadIdx.y + blockIdx.y * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __syncthreads();


    if ((blockIdx.x * blockDim.x < numX) && (blockIdx.y * blockDim.y < numY)) {
        int kk_internal = 0;
        double matrix_prod_sum = 0.0;
        for (int jA = 0; jA < dim; ++jA) {
            double s_mm = 0;
            for (int kk_internal_big = 0; kk_internal_big < len / TILE_WIDTH + (len % TILE_WIDTH == 0 ? 0 : 1); ++kk_internal_big) {
                if (tx == 0) {
                    if (ty + kk_internal_big * TILE_WIDTH < len) {
                        power_mult[ty] = pow(1. / ((double) (len - (ty + kk_internal_big * TILE_WIDTH))), 2 * lambda) * w_sum_sym;
                    }
                }
                if ((kk < numX) * (TILE_WIDTH * kk_internal_big + ty < len)) {
                    ta[tx][ty] = blockX(kk, TILE_WIDTH * kk_internal_big + ty + jA * len);
                    if (TILE_WIDTH * kk_internal_big + ty > 0) {
                        tam1[tx][ty] = ta[tx][ty] - blockX(kk, TILE_WIDTH * kk_internal_big + ty - 1 + jA * len);
                    }
                }
                if ((mm < numY) * (TILE_WIDTH * kk_internal_big + tx < len)) {
                    tb[ty][tx] = blockYY(mm, TILE_WIDTH * kk_internal_big + tx + jA * len);
                    if (TILE_WIDTH * kk_internal_big + tx > 0) {
                        tbm1[ty][tx] = tb[ty][tx] - blockYY(mm, TILE_WIDTH * kk_internal_big + tx - 1 + jA * len);
                    }
                }
                __syncthreads();
                if ((kk < numX) && (mm < numY)) {
                    for (int kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
                        kk_internal = kk_internal_small + kk_internal_big * TILE_WIDTH;
                        if (kk_internal < len) {

                            //mm_internal = kk_internal;
                            double x_y = ta[tx][kk_internal_small] - tb[ty][kk_internal_small];
                            double t_left = x_y * x_y;
                            double t_right = 0;
                            if (kk_internal > 0) {
                                double diff_x_y = 0;
                                diff_x_y = tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small];
                                t_right = diff_x_y * diff_x_y;
                            }
                            s_mm += (t_left + tau * t_right) * power_mult[kk_internal_small];
                        }//end if kk_internal
                    }//end for kk_internal_small
                }//end if
                __syncthreads();//DO NOT REMOVE!
            }//end for kk_internal_big - tiles 
            matrix_prod_sum += s_mm;
        }//end for jA 
        if ((kk < numX) && (mm < numY)) {
            //Z[(startX + kk) * sizeY + (startY + mm)] = 1. - matrix_prod_sum / (2. * sigma * sigma)+pow(matrix_prod_sum/(2*sigma*sigma),2)/2.;
            Z[(startX + kk) * sizeY + (startY + mm)] = matrix_prod_sum;
        }
    }//end if check get_global 0 and 1
}


int check_gpu_malloc(double *&d_X, size_t size_X)
{
    gpuErrchk(cudaMalloc(&d_X, size_X * sizeof(double)));
    return 0;
}

int check_gpu_free(double *d_X)
{
    gpuErrchk(cudaFree(d_X));
    return 0;
}

int check_gpu_copy_to_device(double *d_X, double *h_X, size_t size_X)
{
    gpuErrchk(cudaMemcpy(d_X, h_X, size_X * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

int check_gpu_copy_to_host(double *h_X, double *d_X, size_t size_X)
{
    gpuErrchk(cudaMemcpy(h_X, d_X, size_X * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}


int do_gpu_kernel_compute_mat_xx(size_t sizeX, size_t startX, size_t startY, size_t numX, size_t numY, size_t total_len_features, int dim, const double *X, double *Z,
                                 double *&d_Zptr, /* double param1, */ double param2, double param3, double param4)
{
    size_t len = total_len_features / dim;
    assert(len * dim == total_len_features);

    double time0 = msecs();
    ctimes[0] += msecs() - time0;
    double times1 = msecs();
    size_t full_sizeX = sizeX * total_len_features;
    size_t full_sizeZ = sizeX * sizeX;
    thrust::device_vector<double> d_X(full_sizeX);
    gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(d_X.data()), &X[0], sizeofgpu_kernel_xx_compute(double) * full_sizeX, cudaMemcpyHostToDevice));
    ctimes[1] += msecs() - times1;
    double times2 = msecs();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[2] += msecs() - times2;
    double times3 = msecs();

    double *d_Xptr = thrust::raw_pointer_cast(d_X.data());
    gpuErrchk(cudaMalloc(&d_Zptr, full_sizeZ * sizeof(double)));
    ctimes[3] += msecs() - times3;
    double times4 = msecs();

    int tile_x = TILE_WIDTH;
    int tile_y = TILE_WIDTH;
    assert(tile_x == tile_y);
    const dim3 thread_dim(tile_x, tile_y);

    int block_x = (sizeX / tile_x) + (sizeX % tile_x == 0 ? 0 : 1);
    int block_y = (sizeX / tile_y) + (sizeX % tile_y == 0 ? 0 : 1);

    const dim3 block_dim(block_x, block_y);

    std::cout << full_sizeZ << " " << sizeX << " " << len << " " << sizeX * len << " " << block_x * tile_x << std::endl;
    std::cout << "Params " << param1 << " " << param2 << " " << param3 << " " << param4 << std::endl;
    std::cout << "Len dim  " << len << " " << dim << std::endl;
    assert(tile_x * tile_y * block_x * block_y >= sizeX * sizeX);
    gpu_kernel_xx_compute<<<block_dim, thread_dim>>>(sizeX, startX, startY, numX, numY, len, dim, d_Xptr, d_Zptr, /* full_sizeZ,param1, */param2, param3, param4);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[4] += msecs() - times4;
    double times5 = msecs();
    gpuErrchk(cudaMemcpy(&Z[0], d_Zptr, full_sizeZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[5] += msecs() - times5;
    std::cout << "Cuda times ";
    for (int i = 0; i < 6; i++) std::cout << ctimes[i] << " ";
    std::cout << std::endl;
    gpuErrchk(cudaFree(d_Zptr));
    return 0;
}


int do_gpu_kernel_compute_mat_xy(size_t sizeX, size_t sizeY, size_t startX, size_t startY, size_t numX, size_t numY, size_t total_len_features, int dim, const double *X,
                                 const double *Y, double *Z, double *&d_Zptr, double param1, double param2, double param3, double param4)
{
    size_t len = total_len_features / dim;
    assert(len * dim == total_len_features);

    double time0 = msecs();
    ctimes[0] += msecs() - time0;
    double times1 = msecs();
    size_t full_sizeX = sizeX * total_len_features;
    size_t full_sizeY = sizeY * total_len_features;
    size_t full_sizeZ = sizeX * sizeY;
    thrust::device_vector<double> d_X(full_sizeX);
    thrust::device_vector<double> d_Y(full_sizeY);
    gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(d_X.data()), &X[0], sizeof(double) * full_sizeX, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(d_Y.data()), &Y[0], sizeof(double) * full_sizeY, cudaMemcpyHostToDevice));
    ctimes[1] += msecs() - times1;
    double times2 = msecs();
    //thrust::host_vector<double> h_Z(v_Z.begin(), v_Z.end());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[2] += msecs() - times2;
    double times3 = msecs();

    double *d_Xptr = thrust::raw_pointer_cast(d_X.data());
    double *d_Yptr = thrust::raw_pointer_cast(d_Y.data());
    thrust::device_vector<double> d_Z(full_sizeZ);
    gpuErrchk(cudaMalloc(&d_Zptr, full_sizeZ * sizeof(double)));
    ctimes[3] += msecs() - times3;
    double times4 = msecs();

    int tile_x = 16;
    int tile_y = 16;
    assert(tile_x == tile_y);
    const dim3 thread_dim(tile_x, tile_y);
    int block_x = (sizeX / tile_x) + (sizeX % tile_x == 0 ? 0 : 1);
    int block_y = (sizeX / tile_y) + (sizeX % tile_y == 0 ? 0 : 1);

    const dim3 block_dim(block_x, block_y);

    std::cout << full_sizeZ << " " << sizeX << " " << len << " " << sizeX * len << std::endl;
    gpu_kernel_xy_compute<<<block_dim, thread_dim>>>(sizeX, sizeY, startX, startY, numX, numY, len, dim, d_Xptr, d_Yptr, d_Zptr, full_sizeZ, param1, param2, param3,
                                                     param4);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[4] += msecs() - times4;
    double times5 = msecs();
    gpuErrchk(cudaMemcpy(&Z[0], d_Zptr, full_sizeZ * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    ctimes[5] += msecs() - times5;
    std::cout << "Cuda times ";
    for (int i = 0; i < 6; i++) std::cout << ctimes[i] << " ";
    std::cout << std::endl;
    return 0;
}


int
cuda_kernel_xx(size_t total_len_features, int dim, size_t size_X, size_t startX, size_t startY, size_t numX, size_t numY, const double *X,/*double sigma,*/double lambda,
               double tau, double w_sum_sym, double *Z, double *&d_Z_ptr)
{
//    double param1=sigma;
    double param2 = lambda;
    double param3 = tau;
    double param4 = w_sum_sym;
    do_gpu_kernel_compute_mat_xx(size_X, startX, startY, numX, numY, total_len_features, dim, X, Z, d_Z_ptr,/*param1,*/param2, param3, param4);
    return 0;
}

int
cuda_kernel_xy(size_t total_len_features, int dim, size_t size_X, size_t size_Y, size_t startX, size_t startY, size_t numX, size_t numY, const double *X, const double *Y,
               double sigma, double lambda, double tau, double w_sum_sym, double *Z, double *&d_Z_ptr)
{
    double param1 = sigma;
    double param2 = lambda;
    double param3 = tau;
    double param4 = w_sum_sym;
    do_gpu_kernel_compute_mat_xy(size_X, size_Y, startX, startY, numX, numY, total_len_features, dim, X, Y, Z, d_Z_ptr, param1, param2, param3, param4);
    return 0;
}


double score_distance_kernel(size_t sizeX, double *Z_distances, double *Y)
{

//labels = Y 
//kernel matrix - Z


    /* std::vector<double> Z_distances(sizeX*sizeX);
    for(int i=0;i<sizeX;i++){
        for(int j=0;j<sizeX;j++){
            Z_distances[i*sizeX+j]=2.*(1.-Z[i*sizeX+j]);
        }
    } 	*/
    size_t N1 = 0;
    size_t N2 = 0;
    for (size_t i = 0; i < sizeX; i++) {
        if (Y[i] < 0) N1++;
        if (Y[i] > 0) N2++;
    }
    std::cout << " N1,N2" << N1 << " " << N2 << std::endl;
    size_t N = N1 + N2;
    double E12 = 0.;
    size_t i_ctr = 0;
    for (size_t i = 0; i < N1; i++) {
        while (!(Y[i_ctr] < 0)) i_ctr++;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N2; j++) {
            while (!(Y[j_ctr] > 0)) j_ctr++;
            E12 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            j_ctr++;
        }
        i_ctr++;
    }
    E12 = E12 / (double) N1 / (double) N2;
    double E11 = 0.;
    i_ctr = 0;
    for (size_t i = 0; i < N1; i++) {
        while (!(Y[i_ctr] < 0)) i_ctr++;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N1; j++) {
            while (!(Y[j_ctr] < 0)) j_ctr++;
            E11 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            j_ctr++;
        }
        i_ctr++;
    }
    E11 = E11 / (double) N1 / (double) N1;
    double E22 = 0.;
    i_ctr = 0;
    for (size_t i = 0; i < N2; i++) {
        while (!(Y[i_ctr] > 0)) i_ctr++;
        size_t j_ctr = 0;
        for (size_t j = 0; j < N2; j++) {
            while (!(Y[j_ctr] > 0)) j_ctr++;
            E22 += pow(Z_distances[i_ctr * sizeX + j_ctr], 2);
            j_ctr++;
        }
        i_ctr++;
    }
    E22 = E22 / (double) N2 / (double) N2;
    std::cout << E11 << " " << E12 << " " << E22 << std::endl;
    return E12 / ((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22);
    //alternative?
    return E12 - (((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22)) / 2.;
}
