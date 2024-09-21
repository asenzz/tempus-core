//
// Created by zarko on 5/26/24.
//

#include <ipp/ippcore.h>
#include <boost/log/expressions/formatters/format.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include "common/logging.hpp"

logging::logging()
{
    // ip_errchk(ippInit());

    auto p_log_core = boost::log::core::get();
    p_log_core->remove_all_sinks();
    p_log_core->set_filter(boost::log::trivial::severity >= boost::log::trivial::severity_level(svr::common::PropertiesFileReader::S_log_threshold));
#ifdef NDEBUG
    auto p_sink = boost::make_shared<boost::log::sinks::asynchronous_sink<boost::log::sinks::text_ostream_backend>>();
#else
    auto p_sink = boost::make_shared<boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>>();
#endif
    // Add a stream to write log to
    p_sink->locked_backend()->add_stream(boost::make_shared<std::ostream>(std::cout.rdbuf()));
    p_sink->set_formatter(boost::log::expressions::stream <<
                                                          boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%b-%d %H:%M:%S.%f") <<
                                                          " [" << boost::log::expressions::attr<boost::log::trivial::severity_level>("Severity") << "] " <<
                                                          boost::log::expressions::smessage);
    // Register the sink in the logging core
    p_log_core->add_sink(p_sink);
    boost::log::add_common_attributes();

#if 0 // TODO Implement logging application configuration options for log file and enable
    boost::log::add_file_log(
             boost::log::keywords::file_name = "tempus_%N.log",                                        // file name pattern
             boost::log::keywords::rotation_size = 10 * 1024 * 1024,                                   // rotate files every 10 MiB...
             boost::log::keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0), // ...or at midnight
             boost::log::keywords::format = "[%TimeStamp%] %Message%"                                  // log record format
         );
#endif
}

void logging::flush() const
{
    boost::log::core::get()->flush();
}

const auto l__ = []{ return logging(); } ();

std::string cufft_get_error_string(const cufftResult s)
{
    switch (s) {
        case CUFFT_SUCCESS:
            return "Any CUFFT operation is successful.";
        case CUFFT_INVALID_PLAN:
            return "CUFFT is passed an invalid plan handle.";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT failed to allocate GPU memory.";
        case CUFFT_INVALID_TYPE:
            return "The user requests an unsupported type.";
        case CUFFT_INVALID_VALUE:
            return "The user specifies a bad memory pointer.";
        case CUFFT_INTERNAL_ERROR:
            return "Used for all internal driver errors.";
        case CUFFT_EXEC_FAILED:
            return "CUFFT failed to execute an FFT on the GPU.";
        case CUFFT_SETUP_FAILED:
            return "The CUFFT library failed to initialize.";
        case CUFFT_INVALID_SIZE:
            return "The user specifies an unsupported FFT size.";
        default:
            return "Unknown error";
    }
}
