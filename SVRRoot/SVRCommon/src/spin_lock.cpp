#include <common/spin_lock.hpp>
#include <emmintrin.h>

namespace svr{
namespace common{

spin_lock::spin_lock()
: flag {0}
{
}


void spin_lock::lock() noexcept
{
    for (;;) {
        // Optimistically assume the lock is free on the first try
        if (!flag.exchange(true, std::memory_order_acquire)) {
            return;
        }
        // Wait for lock to be released without generating cache misses
        while (flag.load(std::memory_order_relaxed)) {
            // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
            // hyper-threads
            _mm_pause();
        }
    }
}


void spin_lock::unlock() noexcept
{
    flag.store(false, std::memory_order_release);
}


bool spin_lock::try_lock() noexcept
{
    if (!flag) return false;
    // First do a relaxed load to check if lock is free in order to prevent
    // unnecessary cache misses if someone does while(!try_lock())
    return !flag.load(std::memory_order_relaxed) && !flag.exchange(true, std::memory_order_acquire);
}

}
}