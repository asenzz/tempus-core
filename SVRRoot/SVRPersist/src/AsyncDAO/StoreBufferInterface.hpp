#ifndef StoreBufferINTERFACE_HPP
#define StoreBufferINTERFACE_HPP

namespace svr { namespace dao {

struct StoreBufferInterface
{
    virtual ~StoreBufferInterface() {}
    virtual unsigned long  storeOne() = 0;
};

} }

#endif /* StoreBufferINTERFACE_HPP */

