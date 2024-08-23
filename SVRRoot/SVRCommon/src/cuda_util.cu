//
// Created by zarko on 7/19/24.
//

#include "common/cuda_util.cuh"

namespace svr {

__global__ void
G_copy_submat(CRPTR(double) in, double *const out, const unsigned in_m, const unsigned out_m, const unsigned out_mn)
{
    CU_STRIDED_FOR_i(out_mn) out[i] = in[LDi(i, out_m, in_m)];
}

void copy_submat(CPTR(double) in, double *const out, const unsigned ldin, const unsigned in_start_m, const unsigned in_start_n, const unsigned in_end_m,
                 const unsigned in_end_n, const unsigned ldout, cudaMemcpyKind kind, const cudaStream_t stm)
{
#if 1
    cu_errchk(cudaMemcpy2DAsync(out, ldout * sizeof(double), in + in_start_m + in_start_n * ldin, ldin * sizeof(double), (in_end_m - in_start_m) * sizeof(double),
                                in_end_n - in_start_n, kind, stm));
#else
    const auto out_m = in_end_m - in_start_m;
    const auto out_n = in_end_n - in_start_n;
    const unsigned out_mn = out_m * out_n;
    const auto start_offset = out_start_n * in_m + out_start_m;
    G_copy_submat<<<CU_BLOCKS_THREADS(out_mn), 0, strm>>>(in + start_offset, out, in_m, out_m, start_offset, out_mn);
#endif
}

NppStreamContext get_npp_context(const unsigned gpuid, const cudaStream_t custream)
{
    NppStreamContext ctx;
    ctx.hStream = custream;
    ctx.nCudaDeviceId = gpuid;

#ifdef HETEROGENOUS_GPU_HW
    cudaDeviceProp prop;
    cu_errchk(cudaGetDeviceProperties(&prop, gpuid));
#else
    static auto prop = [gpuid]() {
        cudaDeviceProp prop;
        cu_errchk(cudaGetDeviceProperties(&prop, gpuid));
        return prop;
    } ();
#endif
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    cu_errchk(cudaStreamGetFlags(custream, &ctx.nStreamFlags));
    return ctx;
}

}