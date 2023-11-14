#include "test_harness.hpp"

#include <memory>
#include <fstream>
#include <common.hpp>


bool erase_after(std::string & where, char what)
{
    std::string::size_type pos = where.find(what);
    if (pos == std::string::npos)
        return false;

    where.erase(where.begin() + pos, where.end());
    return true;
}

std::string exec(char const * cmd, int & procExitCode)
{
    char buffer[128];
    std::string result = "";
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), [&procExitCode](FILE* what){procExitCode = pclose(what);});
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
    return result;
}


std::string exec(std::string const & cmd)
{
    int pec;
    std::string result =  exec(cmd.c_str(), pec);
    if(pec != 0)
        throw std::runtime_error("Cannot exec the command.");
    return result;
}

std::vector<double> read_test_data(const std::string &file_name_to_search)
{
    auto f = open_data_file(file_name_to_search);

    std::vector<double> result;

    while(f.good())
    {
        double a;
        f >> a;
        result.push_back(a);
    }
    return result;
}

std::ifstream open_data_file(std::string file_name_to_search)
{
    std::string file_path = exec("find ../ -wholename \"*" + file_name_to_search + "*\"");
    erase_after(file_path, '\n');

    std::ifstream f(file_path);
    if (!f) LOG4_THROW("Cannot find the input data file");

    return f;
}

std::vector<double> read_length_of_test_data(std::string file_name_to_search, int length)
{
    auto f = open_data_file(file_name_to_search);

    std::vector<double> result;
    
    int i=0;
    while( f.good() && i++ < length )
    {
        double a;
        f >> a;
        result.push_back(a);
    }
    return result;
}