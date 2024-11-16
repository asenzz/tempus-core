#include "command.hpp"

#include <map>
#include <boost/date_time.hpp>
#include "common/compatibility.hpp"
#include "command_text.hpp"

namespace {
    struct Exposure
    {
        typedef std::map<DTYPE(CommandText::command), Command *> my_cont_t;
        my_cont_t commands;

        ~Exposure()
        {
            for(my_cont_t::iterator iter = commands.begin(), end = commands.end(); iter != end; ++iter)
                delete iter->second;
        }

        void registerCommand(std::string const & commandName, Command & command)
        {
            commands[commandName] = &command;
        }
    };

    Exposure * exposure = nullptr;
}

Command::Command(std::string const & commandName)
{
    CommandRegistry::inst();
    exposure->registerCommand(commandName, *this);
}

Command::~Command()
{
}

struct CommandRegistry::Impl : Exposure
{};

CommandRegistry::CommandRegistry()
: pImpl(*new Impl)
{
    exposure = & pImpl;
}

CommandRegistry::~CommandRegistry()
{
    exposure = nullptr;
    delete & pImpl;
}

CommandRegistry& CommandRegistry::inst()
{
    static CommandRegistry inst;
    return inst;
}

Command & CommandRegistry::getCommand(std::string const & commandName)
{
    static Impl::my_cont_t::iterator iter;
    iter = pImpl.commands.find(commandName);
    if(iter == pImpl.commands.end())
        throw std::runtime_error(std::string("No such command: ") + commandName);
    return *iter->second;
}

#include "commands.cmm"
