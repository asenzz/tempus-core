#pragma once

#include <boost/format.hpp>
#include <boost/date_time.hpp>
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>
#define BOOST_LOG_DYN_LINK 1
#include <boost/log/trivial.hpp>
#include <boost/timer/timer.hpp>

#include "../util/MemoryManager.hpp"
#include "../util/string_utils.hpp"
#include "../util/ValidationUtils.hpp"

namespace bpt = boost::posix_time;

#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define OUTPUT_FORMAT std::setprecision(std::numeric_limits<double>::max_digits10) << boost::format("%|_15|:%|-4| [%|-15|] %s")
enum LOG_LEVEL_T{TRACE=1, DEBUG=2, INFO=3, WARN=4, ERR=5, FATAL=6};
extern int SVR_LOG_LEVEL;

#define FLUSH_OUTS  fflush(stdout); fflush(stderr);

#ifdef ENABLE_DEBUGGING

#define DO_WRAP(cmd) do{ cmd; FLUSH_OUTS; } while(0)

#define EXCEPTION_FORMAT FILE_NAME << ":" << __LINE__ << " [" << __FUNCTION__ << "] "
#define PRINT_STACK std::cout << boost::stacktrace::stacktrace() << std::endl

#define THROW_EX_F(ex, msg) DO_WRAP(::svr::common::throwx<ex>(::svr::common::formatter() << EXCEPTION_FORMAT << msg);)
#define THROW_EX_FS(ex, msg) DO_WRAP(PRINT_STACK; ::svr::common::throwx<ex>(::svr::common::formatter() << EXCEPTION_FORMAT << msg);)

#define BOOST_CODE_LOCATION OUTPUT_FORMAT % FILE_NAME % __LINE__ % __FUNCTION__

#define LOG4_BEGIN() DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE)    BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % "Begin.";)
#define LOG4_END() DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE)      BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % "End.";)
#define LOG4_TRACE(msg) DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::TRACE) BOOST_LOG_TRIVIAL(trace)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_DEBUG(msg) DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::DEBUG) BOOST_LOG_TRIVIAL(debug)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_INFO(msg)  DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::INFO)  BOOST_LOG_TRIVIAL(info)     << BOOST_CODE_LOCATION % msg;)
#define LOG4_WARN(msg)  DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::WARN)  BOOST_LOG_TRIVIAL(warning)  << BOOST_CODE_LOCATION % msg;)
#define LOG4_ERROR(msg) DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::ERR)   BOOST_LOG_TRIVIAL(error)    << BOOST_CODE_LOCATION % msg;)
#define LOG4_FATAL(msg) DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::FATAL) BOOST_LOG_TRIVIAL(fatal)    << BOOST_CODE_LOCATION % msg;)

#define LOG4_THROW(msg) DO_WRAP(if(SVR_LOG_LEVEL <= LOG_LEVEL_T::ERR)   BOOST_LOG_TRIVIAL(error)    << BOOST_CODE_LOCATION % msg; THROW_EX_FS(::std::runtime_error, msg);)
//#define LOG4_ASSERT(cond, failmsg) { if (!(cond)) LOG4_THROW((failmsg)); } // TODO Fix!
#define LOG4_FILE(logfile, msg) { std::ofstream of( ::svr::common::formatter() << logfile, std::ofstream::out | std::ofstream::app); of.precision(std::numeric_limits<double>::max_digits10); of << msg << std::endl; }

#else

#define LOG4_BEGIN()
#define LOG4_END()
#define LOG4_TRACE(msg) {}
#define LOG4_DEBUG(msg) {}
#define LOG4_INFO(msg) {}
#define LOG4_WARN(msg) {}
#define LOG4_ERROR(msg) {}
#define LOG4_FATAL(msg) {}

#endif

#define PROFILE_EXEC_TIME(X, M_NAME) \
    { bpt::ptime __start_time = bpt::microsec_clock::local_time(); X; LOG4_INFO("Execution time of " << M_NAME << " is " << bpt::microsec_clock::local_time() - __start_time << " process memory RSS " << svr::common::memory_manager::get_process_resident_set() << " MB"); }
