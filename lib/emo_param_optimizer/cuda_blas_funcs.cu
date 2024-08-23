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
#include <mutex>
#include "fast_functions.hpp"

#define BLOCK_SIZE 32
 
// Example with cublasLt library to execute single precision 
// gemm with cublasLtMatmul. This is almost a drop-in replacement for 
// cublasSgemm, with the addition of the workspace to support 
// split-K algorithms.
//
// Additional notes: Pointer mode is always host. To change it,
// configure the appropriate matmul descriptor attribute.
//
// Matmul here does not use cuBLAS handle's configuration of math
// mode. Also, here tensor ops are implicitly allowed; to change
// this, configure appropriate attribute in the preference handle.




#ifndef cublasSafeCall
#define cublasSafeCall(err)     {if (err) std::cout << " Abort "<<  __FILE__<< __LINE__ << std::endl;}
#endif

#define gpu_errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cufft_errchk(ans) { cufftAssert((ans), __FILE__, __LINE__); }

inline void cufftAssert(cufftResult_t code,const  char *file, int line, bool abort=true){
        if (code != CUFFT_SUCCESS) {
                fprintf(stderr,"cufftAssert: error code %i file %s line %d\n", (int)code, file, line);
                if (abort) exit(code);
        }
}




#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,const  char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
double stimes[20];
int dyn_qrsolve(int Nrows, int lda ,  double*d_Ainput,double*d_rhs, double*d_B){
        double times0=msecs();
        int Nright=1;
        double*d_work=nullptr;
        int lwork=0;
        int info_gpu=0;
        //const double one=1;
        const int ldb=lda;
        int Ncols = Nrows;
        //const int nrhs=Nright;
        const int m = Ncols;
        thrust::device_vector<double> d_tau_vector(m);
        thrust::device_vector<double> d_A_vector(Nrows*lda);
        thrust::device_vector<int> d_Ipiv_vector(m);
        int *d_Ipiv = thrust::raw_pointer_cast(d_Ipiv_vector.data());
        double *d_A = thrust::raw_pointer_cast(d_A_vector.data());
        gpuErrchk(cudaMemcpy(d_A,d_Ainput,sizeof(double)*Nrows*lda,cudaMemcpyDeviceToDevice));
        stimes[11]+=msecs()-times0;
        double times1=msecs();
        double *d_tau=thrust::raw_pointer_cast(d_tau_vector.data());
        thrust::device_vector<int> d_devInfo_vector(1);
        int*devInfo=thrust::raw_pointer_cast(d_devInfo_vector.data());

        cusolverDnHandle_t cusolverH;
        cublasHandle_t cublasH;

        cusolverDnCreate(&cusolverH);
        cublasSafeCall(cublasCreate(&cublasH));
        cublasSafeCall(cusolverDnDgetrf_bufferSize( cusolverH, m, m, d_A, lda, &lwork));
        gpuErrchk(cudaMalloc((void**)&d_work, sizeof(double)*lwork));

        gpuErrchk(cudaMemcpy(d_B,d_rhs,sizeof(double)*Nright*Nrows,cudaMemcpyDeviceToDevice));
        stimes[12]+=msecs()-times1;
        double times2=msecs();
        cublasSafeCall(cusolverDnDgetrf( cusolverH, m, m, d_A, lda, d_work, d_Ipiv,devInfo));
        //cudaDeviceSynchronize();
        stimes[13]+=msecs()-times2;
        double times3=msecs();

        gpuErrchk(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        assert(0==info_gpu);

        stimes[14]+=msecs()-times3;
        double times4=msecs();
        cublasSafeCall(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */ d_A, lda, d_Ipiv, d_B, ldb, devInfo));
        cudaDeviceSynchronize();
       stimes[15]+=msecs()-times4;
        double times5=msecs();
        gpuErrchk(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        stimes[16]+=msecs()-times5;
        double times6=msecs();
        assert(0 == info_gpu);
        if (cublasH ) cublasDestroy(cublasH);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        gpuErrchk(cudaFree(d_work));

        stimes[17]+=msecs()-times6;
        return 0;

}




int do_gpu_solve(size_t m, double*Left, double*Right, double*output){
        thrust::device_vector<double> d_Left(m*m);
        thrust::device_vector<double> d_Right(m);
        thrust::device_vector<double> d_output(m);
        gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(d_Left.data()), Left, sizeof(double)*m*m,cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(thrust::raw_pointer_cast(d_Right.data()), Right, sizeof(double)*m,cudaMemcpyHostToDevice));
        dyn_qrsolve(m, m, thrust::raw_pointer_cast(d_Left.data()) ,thrust::raw_pointer_cast(d_Right.data()),  thrust::raw_pointer_cast(d_output.data()));
        gpuErrchk(cudaMemcpy(output,thrust::raw_pointer_cast(d_output.data()),sizeof(double)*m,cudaMemcpyDeviceToHost));
        return 0;
}


void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const double alf = 1;
     const double bet = 0;
     const double *alpha = &alf;
     const double *beta = &bet;

     // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

     // Do the actual multiplication
     auto result=cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,  n,k  , alpha, A, lda, B, ldb, beta, C, ldc);

     // Destroy the handle
     cublasDestroy(handle);
 }


