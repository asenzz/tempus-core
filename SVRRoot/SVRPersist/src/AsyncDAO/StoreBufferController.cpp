#include "StoreBufferController.hpp"

#include <deque>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>

#include <pthread.h>

#include <common/ScopeExit.hpp>
#include "StoreBufferInterface.hpp"

namespace svr { namespace dao {

struct StoreBufferController::StoreBufferControllerImpl
{
    using my_cont_t = std::deque<StoreBufferInterface*>;
    using iterator = typename my_cont_t::iterator;

    my_cont_t storeBuffers;
    std::timed_mutex mutex;
    std::thread * poller {nullptr};
    std::atomic<bool> volatile poll {false}, stopped;
    pthread_t tid;

    static void polling(StoreBufferController::StoreBufferControllerImpl & inst_)
    {
        inst_.stopped = false; auto sg {at_scope_exit([&inst_](){ inst_.stopped = true; } ) }; (void) sg;
        inst_.tid = pthread_self();

        while (inst_.poll)
        {
            inst_.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void flush()
    {
        std::scoped_lock<std::timed_mutex> lg (mutex);
        while(!storeBuffers.empty())
        {
            StoreBufferInterface* intf = storeBuffers.front();
            storeBuffers.pop_front();

            intf->storeOne();
        }
    }
};

StoreBufferController * StoreBufferController::inst{nullptr};

void StoreBufferController::initInstance()
{
    inst = new StoreBufferController();
}

void StoreBufferController::destroyInstance()
{
    delete inst;
}

StoreBufferController& StoreBufferController::getInstance()
{
    return *inst;
}

void StoreBufferController::startPolling()
{
    pImpl.poll = true;
    pImpl.poller = new std::thread(StoreBufferController::StoreBufferControllerImpl::polling, std::ref(pImpl));
}

void StoreBufferController::stopPolling()
{
    pImpl.poll = false;

    flush();

    for(int i = 0; i < 1000 && !pImpl.stopped; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if(!pImpl.stopped)
    {
        pImpl.poller->detach();
        pthread_cancel(pImpl.tid);
    } else
        pImpl.poller->join();

    delete pImpl.poller;

}

void StoreBufferController::flush()
{
    pImpl.flush();
}


void StoreBufferController::addStoreBuffer(StoreBufferInterface & inst_)
{
    const std::scoped_lock l(pImpl.mutex);
    pImpl.storeBuffers.push_back(&inst_);
}

StoreBufferController::StoreBufferController()
:pImpl(*new StoreBufferController::StoreBufferControllerImpl)
{}

StoreBufferController::~StoreBufferController()
{
    delete &pImpl;
}

} }
