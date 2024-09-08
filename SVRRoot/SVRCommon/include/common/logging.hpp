#pragma once

#include <boost/format.hpp>
#include <boost/date_time.hpp>
#include <boost/stacktrace.hpp>
#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>
#include "util/MemoryManager.hpp"
#include "util/string_utils.hpp"
#include "util/validation_utils.hpp"

namespace bpt = boost::posix_time;

#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define OUTPUT_FORMAT std::setprecision(std::numeric_limits<double>::max_digits10) << boost::format("%|_15|:%|-4| [%|-15|] %s")
enum LOG_LEVEL_T {
    TRACE = 1, DEBUG = 2, INFO = 3, WARN = 4, ERR = 5, FATAL = 6
};
extern int SVR_LOG_LEVEL;

#ifdef SVR_ENABLE_DEBUGGING

#ifdef PRODUCTION_BUILD
#define FLUSH_OUTS
#else
#define FLUSH_OUTS  // (void) ::fflush(stdout); (void) ::fflush(stderr);
#endif

#define CMD_WRAP(cmd) do { cmd; FLUSH_OUTS; } while (0)

#define EXCEPTION_FORMAT FILE_NAME << ":" << __LINE__ << " [" << __FUNCTION__ << "] "
#define PRINT_STACK std::cout << "Executing thread " << std::this_thread::get_id() << ", Backtrace:\n" << boost::stacktrace::stacktrace() << '\n'

#define THROW_EX_F(ex, msg) CMD_WRAP(::svr::common::throwx<ex>(::svr::common::formatter() << EXCEPTION_FORMAT << msg);)
#define THROW_EX_FS(ex, msg) CMD_WRAP(PRINT_STACK; ::svr::common::throwx<ex>(::svr::common::formatter() << EXCEPTION_FORMAT << msg);)

#define BOOST_CODE_LOCATION OUTPUT_FORMAT % FILE_NAME % __LINE__ % __FUNCTION__

#define LOG4_BEGIN() CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE)    BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % "Begin.";)
#define LOG4_END() CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE)      BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % "End.";)
#define LOG4_TRACE(msg) CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE) BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_DEBUG(msg) CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::DEBUG) BOOST_LOG_TRIVIAL(debug)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_INFO(msg)  CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::INFO)  BOOST_LOG_TRIVIAL(info)     << BOOST_CODE_LOCATION % msg;)
#define LOG4_WARN(msg)  CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::WARN)  BOOST_LOG_TRIVIAL(warning)  << BOOST_CODE_LOCATION % msg;)
#define LOG4_ERROR(msg) CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::ERR)   BOOST_LOG_TRIVIAL(error)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_FATAL(msg) CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::FATAL) BOOST_LOG_TRIVIAL(fatal)    << BOOST_CODE_LOCATION % msg;)

#define LOG4_THROW(msg) CMD_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::ERR)   BOOST_LOG_TRIVIAL(error)    << BOOST_CODE_LOCATION % msg; THROW_EX_FS(::std::runtime_error, msg);)
//#define LOG4_ASSERT(cond, failmsg) { if (!(cond)) LOG4_THROW((failmsg)); } // TODO Fix!
#define LOG4_FILE(logfile, msg) { std::ofstream of( ::svr::common::formatter() << logfile, std::ofstream::out | std::ofstream::app); of.precision(std::numeric_limits<double>::max_digits10); of << msg << '\n'; }

#else

#define LOG4_BEGIN()
#define LOG4_END()
#define LOG4_TRACE(msg) {}
#define LOG4_DEBUG(msg) {}
#define LOG4_INFO(msg) {}
#define LOG4_WARN(msg) {}
#define LOG4_ERROR(msg) {}
#define LOG4_FATAL(msg) {}
#define LOG4_THROW(msg) {}
#define LOG4_FILE(logfile, msg) {}

#endif

#define START_TIME__ TOKENPASTE2(__start_time_, __LINE__)
#define PROFILE_EXEC_TIME(X, M_NAME)    \
{                                       \
    const bpt::ptime START_TIME__ = bpt::microsec_clock::local_time(); (X);  \
    LOG4_INFO("Execution time of " << M_NAME << " is " << (bpt::microsec_clock::local_time() - START_TIME__) << ", process memory RSS " << \
    svr::common::memory_manager::get_process_resident_set() << " MB"); \
}
#define PROFILE_(X)    \
{                                       \
    const bpt::ptime START_TIME__ = bpt::microsec_clock::local_time(); (X);  \
    LOG4_INFO("Execution time of " #X " is " << (bpt::microsec_clock::local_time() - START_TIME__) << ", process memory RSS " << \
    svr::common::memory_manager::get_process_resident_set() << " MB"); \
}

#include <cufft.h>
#include <sstream>

std::string cufft_get_error_string(const cufftResult s);

#ifdef PRODUCTION_BUILD
#define cf_errchk(cmd) (cmd)
#define ma_errchk(cmd) (cmd)
#define cu_errchk(cmd) (cmd)
#define cb_errchk(cmd) (cmd)
#define cs_errchk(cmd) (cmd)
#define ip_errchk(cmd) (cmd)
#define vs_errchk(cmd) (cmd)
#define np_errchk(cmd) (cmd)
#else

#ifndef np_errchk
#define np_errchk(cmd) { \
    NppStatus __err;     \
    if ((__err = cmd) != NPP_SUCCESS) \
        LOG4_THROW("NVidia perfomance primitive call " #cmd " failed with error " << int(__err)); }
#endif

#ifndef vs_errchk
#define vs_errchk(cmd) {       \
    int __err;                 \
    if ((__err = cmd) != VSL_STATUS_OK) \
        LOG4_THROW("Intel VSL call " #cmd " failed with error " << __err); }
#endif

#ifndef ip_errchk
#define ip_errchk(cmd) {               \
    IppStatus __err;                 \
    if ((__err = cmd) != ippStsNoErr) \
        LOG4_THROW("Intel Performance Primitives call " #cmd " failed with error " << int(__err) << ", " << ippGetStatusString(__err)); \
    }
#endif

#ifndef cf_errchk
#define cf_errchk(cmd) {               \
    cufftResult __err;                 \
    if ((__err = cmd) != CUFFT_SUCCESS)  \
        LOG4_THROW("CUDA FFT call " #cmd " failed with error " << int(__err) << ", " << cufft_get_error_string(__err)); \
    }
#endif

#ifndef ma_errchk
#define ma_errchk(cmd) {   \
        magma_int_t __err; \
        if ((__err = (cmd)) < MAGMA_SUCCESS) \
            LOG4_THROW("Magma call " #cmd " failed with error " << __err << " " << magma_strerror(__err)); \
    }
#endif

#ifndef cu_errchk

constexpr unsigned C_cu_alloc_retries = 1e2;

#define cu_errchk(cmd) {               \
    cudaError_t __err = (cmd);                 \
    if (__err == cudaErrorMemoryAllocation) {        \
        unsigned __retries = 0;                                             \
        while (__err == cudaErrorMemoryAllocation && __retries < C_cu_alloc_retries) {     \
            LOG4_ERROR("CUDA call " #cmd " failed with error " << int(__err) << " " << cudaGetErrorName(__err) << ", " << cudaGetErrorString(__err) << ", retrying."); \
            std::this_thread::sleep_for(std::chrono::milliseconds(1));                                         \
            __err = (cmd);                               \
        }                                                \
    }                                                \
    if (__err != cudaSuccess && __err != cudaErrorHostMemoryAlreadyRegistered) \
        LOG4_THROW("CUDA call " #cmd " failed with error " << int(__err) << " " << cudaGetErrorName(__err) << ", " << cudaGetErrorString(__err)); \
    }

#endif

#ifndef cb_errchk
#define cb_errchk(cmd) {      \
        cublasStatus_t __err; \
        if ((__err = (cmd)) != CUBLAS_STATUS_SUCCESS) \
            LOG4_THROW("Cublas call " #cmd " failed with " << int(__err) << " " << cublasGetStatusName(__err) << ", " << cublasGetStatusString(__err)); \
        }
#endif

#ifndef cs_errchk
#define cs_errchk(cmd) {                                             \
        cusolverStatus_t __err;                                      \
        if ((__err = (cmd)) != CUSOLVER_STATUS_SUCCESS)              \
            LOG4_THROW("Cusolver call " #cmd " failed with " << int(__err));  \
}
#endif
#endif
