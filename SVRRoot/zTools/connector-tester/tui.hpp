#ifndef TUI_HPP
#define TUI_HPP

#include <iosfwd>

#include "command_text.hpp"

class Tui {
    std::istream & input;
    std::ostream & output
        , & error_output;
    bool fin;
public:
    Tui(std::istream & input, std::ostream & output, std::ostream & error_output, bool displayWelcomeMessage);

    CommandText & getCommand();
    void showMessage(std::string const & message, bool error = false);

    bool finished() const;
};

#endif /* TUI_HPP */

