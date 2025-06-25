#pragma once

#include <boost/format.hpp>
#include <boost/date_time.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/stacktrace.hpp>
#include <boost/timer/timer.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/common.hpp>
#include <boost/core/null_deleter.hpp>
#include "util/MemoryManager.hpp"
#include "util/string_utils.hpp"
#include "util/validation_utils.hpp"
#include "util/PropertiesFileReader.hpp"

namespace bpt = boost::posix_time;


#ifdef __FILE_NAME__
#define FILE_NAME __FILE_NAME__
#else
#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define LOG_FORMAT std::setprecision(std::numeric_limits<double>::max_digits10) << boost::format("%|_2x| %|_4|:%|_2| [%|-4|] %s")

class logging {
public:
    logging();

    void flush() const;
};

extern const logging l__;

#ifdef SVR_ENABLE_DEBUGGING

#ifdef PRODUCTION_BUILD
#define FLUSH_OUTS
#else
#define FLUSH_OUTS (void) ::fflush(stdout); (void) ::fflush(stderr);
#endif

#define CMD_WRAP(cmd) do { cmd; FLUSH_OUTS; } while (0)

#define EXCEPTION_FORMAT FILE_NAME << ":" << __LINE__ << " [" << __FUNCTION__ << "] "
#define PRINT_STACK std::cout << "Executing thread " << std::this_thread::get_id() << ", Backtrace:\n" << boost::stacktrace::stacktrace() << '\n'

#define THROW_EX_F(ex, msg) CMD_WRAP( svr::common::throwx<ex>( svr::common::formatter() << EXCEPTION_FORMAT << msg); )
#define THROW_EX_FS(ex, msg) CMD_WRAP( PRINT_STACK; svr::common::throwx<ex>( svr::common::formatter() << EXCEPTION_FORMAT << msg); )

#define BOOST_CODE_LOCATION LOG_FORMAT % std::this_thread::get_id() % FILE_NAME % __LINE__ % __FUNCTION__

#define FUN_START_TIME__ TOKENPASTE2(__start_time_, __FUNCTION__)

#ifdef PROFILE_CALLS

#define LOG4_BEGIN() \
    CMD_WRAP( if (svr::common::PropertiesFileReader::S_log_threshold <= boost::log::trivial::severity_level::trace) \
                BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "Begin."; )                                       \
    const bpt::ptime FUN_START_TIME__ = bpt::microsec_clock::local_time();

#define LOG4_END() \
    CMD_WRAP( if (svr::common::PropertiesFileReader::S_log_threshold <= boost::log::trivial::severity_level::trace) \
        BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "End."; ) \
    LOG4_INFO("Execution time of " << __FUNCTION__ << " is " << (bpt::microsec_clock::local_time() - FUN_START_TIME__) << ", process memory RSS " << \
        svr::common::memory_manager::get_process_resident_set() << " MB");

#else

#define LOG4_BEGIN() \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::trace) \
                BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "Begin."; )

#define LOG4_END() \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::trace) \
        BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % "End."; )

#endif

#define LOG4_TRACE(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::trace) \
        BOOST_LOG_TRIVIAL(trace) << BOOST_CODE_LOCATION % msg; )

#define LOG4_DEBUG(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::debug) \
        BOOST_LOG_TRIVIAL(debug) << BOOST_CODE_LOCATION % msg; )

#define LOG4_INFO(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::info) \
        BOOST_LOG_TRIVIAL(info) << BOOST_CODE_LOCATION % msg; )

#define LOG4_WARN(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::warning) \
        BOOST_LOG_TRIVIAL(warning) << BOOST_CODE_LOCATION % msg; )

#define LOG4_ERROR(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::error) \
        BOOST_LOG_TRIVIAL(error) << BOOST_CODE_LOCATION % msg; )

#define LOG4_FATAL(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::fatal) \
        BOOST_LOG_TRIVIAL(fatal) << BOOST_CODE_LOCATION % msg; )

#define LOG4_THROW(msg) \
    CMD_WRAP( if (svr::common::AppConfig::S_log_threshold <= boost::log::trivial::severity_level::error) \
        BOOST_LOG_TRIVIAL(error) << BOOST_CODE_LOCATION % msg; THROW_EX_FS(std::runtime_error, msg); )

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

#ifdef NDEBUG

#define PROFILE_MSG(X, M_NAME)                                              \
{                                                                           \
    const bpt::ptime START_TIME__ = bpt::microsec_clock::local_time();      \
    try { (X); } catch (const std::exception &ex__) { LOG4_ERROR("Caught exception " << ex__.what() << ", while executing "#X); throw ex__; } \
    LOG4_INFO("Execution time of " << M_NAME << " is " << (bpt::microsec_clock::local_time() - START_TIME__));  \
}

#else

#define PROFILE_MSG(X, M_NAME)                                              \
{                                                                           \
    const bpt::ptime START_TIME__ = bpt::microsec_clock::local_time();      \
    try { (X); } catch (const std::exception &ex__) { LOG4_ERROR("Caught exception " << ex__.what() << ", while executing "#X); throw ex__; } \
    LOG4_INFO("Execution time of " << M_NAME << " is " << (bpt::microsec_clock::local_time() - START_TIME__) << ", process memory RSS " <<    \
        svr::common::memory_manager::get_proc_rss() << " MB"); \
}

#endif

#define PROFILE_BEGIN const bpt::ptime start_time_begin__ = bpt::microsec_clock::local_time();
#define PROFILE_END(MSG) LOG4_INFO("Execution time of " << MSG << " is " << (bpt::microsec_clock::local_time() - start_time_begin__) << ", process memory RSS " << \
        svr::common::memory_manager::get_proc_rss() << " MB");

#define PROFILE_(X) PROFILE_MSG(X, #X)


#include <cufft.h>
#include <sstream>

std::string cufft_get_error_string(const cufftResult s);

#ifdef PRODUCTION_BUILD

#define cf_errchk(cmd) (cmd)
#define ma_errchk(cmd) (cmd)
#define ma_errchki(cmd) (cmd)
#define cu_errchk(cmd) (cmd)
#define cb_errchk(cmd) (cmd)
#define cs_errchk(cmd) (cmd)
#define ip_errchk(cmd) (cmd)
#define vs_errchk(cmd) (cmd)
#define np_errchk(cmd) (cmd)
#define lg_errchk(cmd) (cmd)

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

#ifndef ma_errchki
#define ma_errchki(cmd, INFO) {   \
        magma_int_t __err; \
        if ((__err = (cmd)) < MAGMA_SUCCESS || (INFO) != MAGMA_SUCCESS) \
            LOG4_THROW("Magma call " #cmd " failed with error " << __err << " " << magma_strerror(__err) << ", info " << (INFO)); \
    }
#endif

#ifndef lg_errchk
#define lg_errchk(cmd) {               \
        const auto __err = (cmd);      \
        if (__err != 0) LOG4_THROW("LightGBM call " #cmd " failed with error " << __err << ", " << LGBM_GetLastError()); \
    }
#endif

#ifndef cu_errchk

constexpr unsigned C_cu_alloc_retries = 1e2;

#define cu_errchk(cmd) {               \
    cudaError_t __err = (cmd);                 \
    if (false /* __err == cudaErrorMemoryAllocation */ ) {        \
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
            LOG4_ERROR("Cusolver call " #cmd " failed with " << int(__err));  \
}
#endif
#endif
