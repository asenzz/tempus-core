//
// Created by zarko on 2/17/22.
//

#ifndef SVR_CUDA_UTIL_CUH
#define SVR_CUDA_UTIL_CUH

#include <cufft.h>
#include <sstream>
#include <cstdint>
#include <thrust/device_vector.h>
#include "common/logging.hpp"
#include "common/defines.h"

constexpr unsigned long long C_sign_mask_dbl = 0x7FFFFFFF;

#define _CUSIGN(X) ((X) > 0. ? 1.: (X) < 0 ? -1. : 0.)
#define _MIN(X, Y) ((X) > (Y) ? (Y) : (Y) > (X) ? (X) : (X))


#define CUDA_THREADS(x_size) ((x_size) > CUDA_BLOCK_SIZE ? CUDA_BLOCK_SIZE : (x_size))
#define CUDA_THREADS_BLOCKS(x_size) (x_size) > CUDA_BLOCK_SIZE ? ((x_size) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE : 1, CUDA_THREADS(x_size)


static inline double msecs()
{
    struct timespec start;
    long seconds, useconds;

    //gettimeofday(&start, NULL);
    clock_gettime(CLOCK_MONOTONIC, &start);
    seconds = start.tv_sec;
    useconds = start.tv_nsec;

    double stime = ((seconds) * 1 + (double) useconds / 1000000000.0);


    return stime;
}

class cuformatter
{
public:
    cuformatter()
    {}

    ~cuformatter()
    {}

    template<typename Type>
    cuformatter &operator<<(const Type &value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const
    { return stream_.str(); }

    operator std::string() const
    { return stream_.str(); }
/*
    operator const char *() const
    { return stream_.str().c_str(); }
*/
    enum ConvertToString
    {
        to_str
    };

    std::string operator>>(ConvertToString)
    { return stream_.str(); }

private:
    std::stringstream stream_;

    cuformatter(const cuformatter &);

    cuformatter &operator=(cuformatter &);
};

#define CU_LOG4_FILE(logfile, msg) { std::ofstream of( ::cuformatter() << logfile, std::ofstream::out | std::ofstream::app); of.precision(std::numeric_limits<double>::max_digits10); of << msg << std::endl; }

template<typename T> static inline std::string
deep_to_nsv(const std::vector<T> &v)
{
    if (v.empty()) return "";

    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < v.size() - 1; ++i) ss << v[i] << '\n';
    ss << v.back();

    return ss.str();
}

#define cufft_errchk(ans) { cufftAssert((ans), __FILE__, __LINE__); }

inline void cufftAssert(cufftResult_t code,const  char *file, int line, bool abort=true){
    if (code != CUFFT_SUCCESS) {
        fprintf(stderr,"cufftAssert: error code %i file %s line %d\n", (int)code, file, line);
        if (abort) exit(code);
    }
}

inline void gpu_assert(const cudaError_t errc, bool abort = false)
{
    if (errc == cudaSuccess) return;
    std::string error_msg = svr::common::formatter() << "Error " << int(errc) << " " << cudaGetErrorName(errc) << ", " << cudaGetErrorString(errc);
    LOG4_ERROR(error_msg);
    if (abort)
        exit(errc);
    else
        THROW_EX_FS(std::runtime_error, error_msg);
}

#define cu_errchk(ans) { gpu_assert((ans)); }

#ifndef cublas_safe_call
#define cublas_safe_call(cmd)                                       \
{                                                                   \
        if (const auto __ec = (cmd))                                \
            LOG4_THROW("Cublas call failed with " << int(__ec));    \
}
#endif

#ifndef cusolver_safe_call
#define cusolver_safe_call(cmd)                                     \
{                                                                   \
        cusolverStatus_t __ec;                                      \
        if ((__ec = (cmd)) != CUSOLVER_STATUS_SUCCESS)              \
            LOG4_THROW("Cublas call failed with " << int(__ec));    \
}
#endif

template<typename T> T *
cuda_malloccopy(const T *source, const size_t size, const cudaMemcpyKind kind)
{
    T *ptr = nullptr;
    switch (kind) {
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            ptr = (T *) malloc(size);
            break;
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDefault:
            cu_errchk(cudaMalloc(&ptr, size));
            break;
    }
    cu_errchk(cudaMemcpy(ptr, source, size, kind));
    return ptr;
}

template<typename T> std::vector<T>
cuda_copy(const T *source, const size_t length)
{
    std::vector<T> res(length);
    cu_errchk(cudaMemcpy(res.data(), source, length * sizeof(T), cudaMemcpyDeviceToHost));
    return res;
}

template<typename T> std::vector<T>
cuda_copy(const thrust::device_vector<T> &source)
{
    std::vector<T> res(source.size());
    cu_errchk(cudaMemcpy(res.data(), thrust::raw_pointer_cast(source.data()), source.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return res;
}

template<typename T> void
cuda_copy(std::vector<T> &res, const T *source, const size_t length)
{
    if (res.size() != length) res.resize(length);
    cu_errchk(cudaMemcpy(res.data(), source, length * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T> T *
cuda_malloccopy(const std::vector<T> &v)
{
    T *ptr = nullptr;
    //std::cout << "Allocating " << v.size() * sizeof(T) << std::endl;
    cu_errchk(cudaMalloc(&ptr, v.size() * sizeof(T)));
    cu_errchk(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
    return ptr;
}

inline void *cuda_calloc(const size_t size, const size_t count)
{
    void *ptr;
    cu_errchk(cudaMalloc(&ptr, count * size));
    cu_errchk(cudaMemset(ptr, 0, count * size));
    return ptr;
}



#endif //SVR_CUDA_UTIL_CUH
