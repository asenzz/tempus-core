#include <iostream>

#include <appcontext.hpp>

#include "tui.hpp"
#include "controller.hpp"

using svr::context::AppContext;

int main(int argc, char** argv)
{
    AppContext::init_instance("app.config");
    svr::context::AppContextDeleter deleter; (void) deleter;

    Tui tui(std::cin, std::cout, std::cerr, argc <= 1);
    Controller controller;
    std::string message;

    if(argc > 1)
    {
        CommandText command;
        command.command = argv[1];
        for(int i = 2; i < argc; ++i)
            command.parameters.push_back(argv[i]);

        bool success = controller.execute(command, message);
        if(!message.empty())
            tui.showMessage(message, success);
        if(!success)
            return 1;
        return 0;
    }

    do{
        CommandText command = tui.getCommand();

        message.clear();
        bool success = controller.execute(command, message);
        if(!message.empty())
            tui.showMessage(message, success);

    } while(!tui.finished());
}
