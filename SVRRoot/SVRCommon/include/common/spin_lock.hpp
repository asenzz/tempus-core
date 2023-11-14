#ifndef SPIN_LOCK_HPP
#define SPIN_LOCK_HPP

#include <atomic>

namespace svr{
namespace common{

class spin_lock
{
    std::atomic<bool> flag = {0};
public:
    spin_lock();
    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;
};

}
}

#endif /* SPIN_LOCK_HPP */

