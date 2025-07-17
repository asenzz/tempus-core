#include "TsDaoBase.hpp"

namespace svr {
namespace dao {
    
std::recursive_mutex &common_mutex_instance()
{
    static std::recursive_mutex mtx;
    return mtx;
}

}}