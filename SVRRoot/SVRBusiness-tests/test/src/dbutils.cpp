#include "../include/dbutils.h"
#include "../include/DaoTestFixture.h"
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <regex>

// TODO Merge with testutils

bool DaoTestFixture::DoPerformanceTests = false;

std::string exec(char const * cmd) {
    int pec;
    return exec(cmd, pec);
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

bool erase_after(std::string & where, char what)
{
    std::string::size_type pos = where.find(what);
    if (pos == std::string::npos)
        return false;

    where.erase(where.begin() + pos, where.end());
    return true;
}

/******************************************************************************/

constexpr char const * TestEnv::ConfigFiles[];

bool TestEnv::init_test_db(char const *dbName)
{
    db_scripts_path.reset();
    char const * ctmp = std::getenv(EnvDbScriptsPathVar);
    if(ctmp != NULL)
        db_scripts_path = boost::make_optional( ctmp ) ;

    if( !db_scripts_path )
    {
        std::string tmp = exec("dirname `find ../../ -name 'init_db.sh'`");
        erase_after(tmp, '\n');
        if(!tmp.empty())
            db_scripts_path.reset(tmp);
    }

    if( !db_scripts_path )
    {
        std::cerr << "Cannot find the db scripts path." << std::endl;
        return false;
    }

    std::string cl = *db_scripts_path + "/init_db.sh " + dbName + " 2>&1";

    std::string expect = exec ("find ~/ -maxdepth 1 -name runwithpass.sh");
    erase_after(expect, '\n');

    if(!expect.empty())
        cl = expect + " " + cl;

    int pec;
    std::string out = exec(cl.c_str(), pec);

    if(pec != 0 || out.find("no tty") != std::string::npos || out.find("3 incorrect password attempts") != std::string::npos)
    {
        std::cerr << "init_db.sh exit code is not 0.\n" << out << std::endl;
        return false;
    }

    std::regex error_regex("error", std::regex::icase);
    std::smatch match;

    if (std::regex_search(out, match, error_regex))
    {
        std::cerr << "Errors found in the init_db.sh output: " << out << std::endl;
        return false;
    }
#if 0
    std::regex copy_regex("COPY [0-9]+", std::regex::icase);
    if (!std::regex_search(out, match, copy_regex))
    {
        std::cerr << "Output indicates no data was copied: " << out << std::endl;
        return false;
    }
#endif
    return true;
}

bool TestEnv::prepareSvrConfig(char const * dbName, std::string const & daoType)
{
    /**************************************************************************/
    // Copying config files
    int pec;
    std::stringstream str_cp; str_cp << "cp ";

    for (char const * cf : ConfigFiles)
        str_cp << "../config/" << cf << " ";

    str_cp << " .";

    exec(str_cp.str().c_str(), pec);
    if( pec )
        return false;


    /**************************************************************************/
    // Setting up db credentials
    {
        std::stringstream str_sed;
        str_sed << "sed -ri 's/(dbname=|user=|password=)([a-zA-Z0-9_]+)/\\1" << dbName << "/g' ";

        for (char const * cf : ConfigFiles)
        {
            std::stringstream str1;
            str1 << str_sed.str() << cf;
            std::string out = exec(str1.str().c_str(), pec);
            if(pec)
                return false;
        }
    }

    /**************************************************************************/
    // Setting up dao type

    {
        std::stringstream str_sed;
        str_sed << "sed -ri 's/(DAO_TYPE[ \\t]*=[ \\t]*)([a-zA-Z0-9_]+)/\\1" << daoType << "/g' app.config";
        std::string out = exec(str_sed.str().c_str(), pec);
        if(pec)
            return false;
    }

    return true;
}
