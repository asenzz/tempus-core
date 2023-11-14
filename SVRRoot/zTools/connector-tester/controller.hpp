#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include "command_text.hpp"

class Controller
{
public:
    bool execute(CommandText const & command, std::string & message);
};


#endif /* CONTROLLER_HPP */

