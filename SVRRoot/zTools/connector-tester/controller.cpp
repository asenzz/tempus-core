#include "controller.hpp"
#include "command.hpp"

bool Controller::execute(CommandText const & command, std::string & message)
{
    try {
        return CommandRegistry::inst().getCommand(command.command).execute(command, message);
    } catch (std::exception & ex)
    {
        message = std::string ("Error: ") + ex.what();
    }
    return false;
}
