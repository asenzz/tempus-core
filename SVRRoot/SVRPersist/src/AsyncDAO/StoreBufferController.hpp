#ifndef StoreBufferCONTROLLER_HPP
#define StoreBufferCONTROLLER_HPP

namespace svr {
namespace dao {
struct StoreBufferInterface;

class StoreBufferController
{
public:
    static void initInstance();

    static void destroyInstance();

    static StoreBufferController &get_instance();

    void start_polling();

    void stop_polling();

    void flush();

    void addStoreBuffer(StoreBufferInterface &inst_);

private:
    StoreBufferController();

    ~StoreBufferController();

    StoreBufferController(StoreBufferController const &) = delete;

    void operator=(StoreBufferController const &) = delete;

    struct StoreBufferControllerImpl;
    StoreBufferControllerImpl &pImpl;
    static StoreBufferController *inst;
};
}
}

#endif /* StoreBufferCONTROLLER_HPP */
