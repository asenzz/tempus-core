#ifndef COMMANDTEXT_HPP
#define COMMANDTEXT_HPP

#include <string>
#include <vector>

struct CommandText
{
    std::string command;
    std::vector<std::string> parameters;
    void clear();
};

#endif /* COMMANDTEXT_HPP */

