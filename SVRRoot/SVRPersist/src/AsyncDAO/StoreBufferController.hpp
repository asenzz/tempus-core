#ifndef StoreBufferCONTROLLER_HPP
#define StoreBufferCONTROLLER_HPP

namespace svr { namespace dao {

struct StoreBufferInterface;

class StoreBufferController
{
public:
    static void initInstance();
    static void destroyInstance();

    static StoreBufferController& getInstance();

    void startPolling();
    void stopPolling();

    void flush();

    void addStoreBuffer(StoreBufferInterface & inst_);
private:
    StoreBufferController();
    ~StoreBufferController();
    StoreBufferController(StoreBufferController const &)=delete;
    void operator=(StoreBufferController const &)=delete;

    struct StoreBufferControllerImpl;
    StoreBufferControllerImpl & pImpl;
    static StoreBufferController * inst;

};


} }

#endif /* StoreBufferCONTROLLER_HPP */

