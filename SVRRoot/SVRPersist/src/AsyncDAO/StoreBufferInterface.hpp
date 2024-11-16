#ifndef StoreBufferINTERFACE_HPP
#define StoreBufferINTERFACE_HPP

namespace svr { namespace dao {

struct StoreBufferInterface
{
    virtual ~StoreBufferInterface() {}
    virtual uint64_t  storeOne() = 0;
};

} }

#endif /* StoreBufferINTERFACE_HPP */

