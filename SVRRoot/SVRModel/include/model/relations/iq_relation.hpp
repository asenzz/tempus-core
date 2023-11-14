#ifndef IQ_RELATION_HPP
#define IQ_RELATION_HPP

#include "relation.hpp"

namespace svr {
namespace datamodel {
    
class InputQueue;    

struct iq_storage_adapter
{
    static std::shared_ptr<InputQueue> load(std::string const & table_name);
};

typedef relation<InputQueue, iq_storage_adapter, std::string> iq_relation;

}}

#endif /* IQ_RELATION_HPP */
