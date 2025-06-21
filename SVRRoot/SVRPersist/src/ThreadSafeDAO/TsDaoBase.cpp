#include "TsDaoBase.hpp"

namespace svr {
namespace dao {
    
std::mutex &common_mutex_instance()
{
    static std::mutex mtx;
    return mtx;
}

}}