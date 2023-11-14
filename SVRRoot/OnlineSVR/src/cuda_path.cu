#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>

#include <unistd.h>

#include "cuda_path.hpp"
#include "common/cuda_util.cuh"
#include "common/gpu_handler.hpp"


#define MAX_LEN 1000

#define XBLOCK 256
#define YBLOCK 256


namespace svr::kernel::path {


#define blockX(i, j) (X[(startX + (i)) * total_len_features+(j)])
#define blockY(i, j) (X[(startY + (i)) * total_len_features+(j)])
#define blockYY(i, j) (Y[(startY + (i)) * total_len_features+(j)])


__global__  void
gpu_kernel_xx_compute(
        const size_t sizeX, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const size_t len, const size_t dim, const double *X, double *Z,
        const double param2, const double param3, const double param4)
{
    //double sigma = param1; - not used when only computing distance
    const double lambda = param2;
    const double tau = param3;
    const double w_sum_sym = param4;
    const size_t total_len_features = len * dim;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH];//for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH];//for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;

    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    //__syncthreads();
    if ((blockIdx.x * blockDim.x < numX) && (blockIdx.y * blockDim.y < numY) && ((startX + blockIdx.x * blockDim.x) <= (startY + blockIdx.y * blockDim.y + blockDim.y - 1))) {
        size_t kk_internal = 0;
        double matrix_prod_sum = 0;
        for (size_t jA = 0; jA < dim; ++jA) {
            double s_mm = 0;
            for (size_t kk_internal_big = 0; kk_internal_big < len / TILE_WIDTH + (len % TILE_WIDTH == 0 ? 0 : 1); ++kk_internal_big) {
                if (tx == 0) {
                    if (ty + kk_internal_big * TILE_WIDTH < len) {
                        power_mult[ty] = pow(1. / ((double) (len - (ty + kk_internal_big * TILE_WIDTH))), 2. * lambda) * w_sum_sym;
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
                    for (size_t kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
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
    } //end if  group_id
}


__global__  void
gpu_kernel_xy_compute(
        const size_t sizeX, const size_t sizeY, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const size_t len, const size_t dim, const double *X, const double *Y,
        double *Z, const size_t full_sizeZ, const double param2, const double param3, const double param4)
{
    const double lambda = param2;
    const double tau = param3;
    const double w_sum_sym = param4;
    const auto total_len_features = len * dim;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH];//for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH];//for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;

    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    //__syncthreads();

    if ((blockIdx.x * blockDim.x < numX) && (blockIdx.y * blockDim.y < numY)) {
        size_t kk_internal = 0;
        double matrix_prod_sum = 0;
        for (size_t jA = 0; jA < dim; ++jA) {
            double s_mm = 0;
            for (size_t kk_internal_big = 0; kk_internal_big < len / TILE_WIDTH + (len % TILE_WIDTH == 0 ? 0 : 1); ++kk_internal_big) {
                if (tx == 0) {
                    if (ty + kk_internal_big * TILE_WIDTH < len) {
                        power_mult[ty] = pow(1. / ((double) (len - (ty + kk_internal_big * TILE_WIDTH))), 2. * lambda) * w_sum_sym;
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
                            const double x_y = ta[tx][kk_internal_small] - tb[ty][kk_internal_small];
                            const double t_left = x_y * x_y;
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


void
do_gpu_kernel_compute_mat_xx(const size_t sizeX, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const size_t total_len_features, const size_t dim, const double *X,
                             double *Z, const double param2, const double param3, const double param4)
{
    common::gpu_context ctx;
    cudaSetDevice(ctx.phy_id());

    const size_t len = total_len_features / dim;

    const size_t full_sizeX = sizeX * total_len_features;
    const size_t full_sizeZ = sizeX * sizeX;
    thrust::device_vector<double> d_X(full_sizeX);
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_X.data()), &X[0], sizeof(double) * full_sizeX, cudaMemcpyHostToDevice));
    cu_errchk(cudaPeekAtLastError());

    double *d_Zptr, *d_Xptr = thrust::raw_pointer_cast(d_X.data());
    cu_errchk(cudaMalloc(&d_Zptr, full_sizeZ * sizeof(double)));

    const size_t tile_x = TILE_WIDTH;
    const size_t tile_y = TILE_WIDTH;
    const dim3 thread_dim(tile_x, tile_y);

    const size_t block_x = (sizeX / tile_x) + (sizeX % tile_x == 0 ? 0 : 1);
    const size_t block_y = (sizeX / tile_y) + (sizeX % tile_y == 0 ? 0 : 1);
    const dim3 block_dim(block_x, block_y);

    gpu_kernel_xx_compute<<<block_dim, thread_dim>>>(sizeX, startX, startY, numX, numY, len, dim, d_Xptr, d_Zptr, param2, param3, param4);
    cu_errchk(cudaMemcpy(&Z[0], d_Zptr, full_sizeZ * sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_Zptr));
}


void
do_gpu_kernel_compute_mat_xy(const size_t sizeX, const size_t sizeY, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const size_t total_len_features, const size_t dim,
                             const double *X, const double *Y, double *Z, const double param2, const double param3, const double param4)
{
    common::gpu_context ctx;
    cudaSetDevice(ctx.phy_id());
    const size_t len = total_len_features / dim;

    const auto full_sizeX = sizeX * total_len_features;
    const auto full_sizeY = sizeY * total_len_features;
    const auto full_sizeZ = sizeX * sizeY;
    thrust::device_vector<double> d_X(full_sizeX);
    thrust::device_vector<double> d_Y(full_sizeY);
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_X.data()), &X[0], sizeof(double) * full_sizeX, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_Y.data()), &Y[0], sizeof(double) * full_sizeY, cudaMemcpyHostToDevice));

    double *d_Xptr = thrust::raw_pointer_cast(d_X.data());
    double *d_Yptr = thrust::raw_pointer_cast(d_Y.data());
    double *d_Zptr;
    cu_errchk(cudaMalloc(&d_Zptr, full_sizeZ * sizeof(double)));

    const size_t tile_x = TILE_WIDTH;
    const size_t tile_y = TILE_WIDTH;
    const dim3 thread_dim(tile_x, tile_y);
    const size_t block_x = (sizeX / tile_x) + (sizeX % tile_x == 0 ? 0 : 1);
    const size_t block_y = (sizeX / tile_y) + (sizeX % tile_y == 0 ? 0 : 1);
    const dim3 block_dim(block_x, block_y);

    gpu_kernel_xy_compute<<<block_dim, thread_dim>>>(sizeX, sizeY, startX, startY, numX, numY, len, dim, d_Xptr, d_Yptr, d_Zptr, full_sizeZ, param2, param3, param4);
    cu_errchk(cudaMemcpy(&Z[0], d_Zptr, full_sizeZ * sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_Zptr));
}


void cu_distances_xx(const size_t total_len_features, const size_t dim, const size_t size_X, const size_t startX, const size_t startY, const size_t numX, const size_t numY, const double *X,
                     const double lambda, const double tau, const double w_sum_sym, double *Z)
{
    const double param2 = lambda;
    const double param3 = tau;
    const double param4 = w_sum_sym;
    do_gpu_kernel_compute_mat_xx(size_X, startX, startY, numX, numY, total_len_features, dim, X, Z, param2, param3, param4);
}

void cu_distances_xy(const size_t total_len_features, const size_t dim, const size_t size_X, const size_t size_Y, const size_t startX, const size_t startY, const size_t numX, const size_t numY,
                     const double *X, const double *Y, const double lambda, const double tau, const double w_sum_sym, double *Z)
{
    const double param2 = lambda;
    const double param3 = tau;
    const double param4 = w_sum_sym;
    do_gpu_kernel_compute_mat_xy(size_X, size_Y, startX, startY, numX, numY, total_len_features, dim, X, Y, Z, param2, param3, param4);
}


double
score_distance_kernel(const size_t sizeX, double *Z_distances, double *Y)
{
    // labels = Y
    // kernel matrix - Z
    /* std::vector<double> Z_distances(sizeX*sizeX);
    for(int i=0;i<sizeX;i++){
            for(int j=0;j<sizeX;j++){
                    Z_distances[i*sizeX+j]=2.*(1.-Z[i*sizeX+j]);
            }
    }
    */
    size_t N1 = 0;
    size_t N2 = 0;
    for (size_t i = 0; i < sizeX; i++) {
        if (Y[i] < 0) N1++;
        if (Y[i] > 0) N2++;
    }
    // std::cout << " N1,N2" << N1 << " " << N2 << std::endl;
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
    // std::cout << E11 << " " << E12 << " " << E22 << std::endl;
    return E12 / ((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22);
    //alternative?
    //return E12 - (((double) N1 / (double) N * E11 + (double) N2 / (double) N * E22)) / 2.;
}

}