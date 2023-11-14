#include "tui.hpp"

#include <boost/tokenizer.hpp>
#include <string>
#include <iostream>

typedef boost::tokenizer<boost::escaped_list_separator<char> > so_tokenizer;

Tui::Tui(std::istream & input, std::ostream & output, std::ostream & error_output, bool displayWelcomeMessage)
: input(input), output(output), error_output(error_output), fin(false)
{
    if (displayWelcomeMessage) output << "Please type help for the list of the available commands.\n";
}

CommandText & Tui::getCommand()
{
    static CommandText command;
    command.clear();

    output << "> ";

    std::string tmp;
    std::getline(input, tmp);

    so_tokenizer tok(tmp, boost::escaped_list_separator<char>('\\', ' ', '\"'));

    if(tok.begin() == tok.end())
        return command;

    command.command = *tok.begin();

    for(so_tokenizer::iterator iter = ++tok.begin(); iter != tok.end(); ++iter)
        command.parameters.push_back(*iter);

    return command;
}

void Tui::showMessage(std::string const & message, bool error)
{
    (error ? error_output: output) << message << '\n';
}

bool Tui::finished() const
{
    return fin;
}
