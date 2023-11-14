#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <string>

struct CommandText;

struct Command
{
    Command(std::string const & commandName);
    virtual ~Command();
    virtual bool execute(CommandText const &, std::string & ) = 0;
    virtual char const * getBriefDescription() = 0;
    virtual char const * getDetailedDescription() = 0;
};

class CommandRegistry
{
public:
    static CommandRegistry& inst();

    Command & getCommand(std::string const & commandName);
private:
    CommandRegistry();
    ~CommandRegistry();
    CommandRegistry(CommandRegistry const &) = delete;
    void operator=(CommandRegistry const &) = delete;

    struct Impl;
    Impl & pImpl;
};


#endif /* COMMAND_HPP */

