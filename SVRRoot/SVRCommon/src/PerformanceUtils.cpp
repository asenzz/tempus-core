#include <util/PerformanceUtils.hpp>


#ifdef PERFORMANCE_TESTING

namespace svr {
namespace common {


performance_timer::performance_timer(std::string name, std::basic_ostream<char> & out)
: name(std::move(name))
, out(out)
, start(std::chrono::steady_clock::now())
{}


performance_timer::~performance_timer()
{
    std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;

    out << "Performance timer: " << name << ", comment: " << comment.str() << ", duration: " << duration.count() << "\n";
}

std::basic_ostream<char> & performance_timer::get_comment_stream()
{
    return comment;
}


}
}
#endif
