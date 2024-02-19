#include "../include/testutils.h"
#include "../include/DaoTestFixture.h"
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <regex>

bool DaoTestFixture::DoPerformanceTests = false;
DaoTestFixture::test_completeness  DaoTestFixture::DoTestPart = DaoTestFixture::test_completeness::full;

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

TestEnv& TestEnv::global_test_env()
{
    static TestEnv env;
    return env;
}

constexpr char const * TestEnv::ConfigFiles[];

bool
TestEnv::init_test_db(char const *db_name)
{
    db_scripts_path.reset();
    char const *ctmp = std::getenv(EnvDbScriptsPathVar);
    if (ctmp != NULL) db_scripts_path = boost::make_optional( ctmp ) ;

    if (!db_scripts_path) {
        std::string tmp = exec("dirname `find ../../ -name 'init_db.sh'`");
        erase_after(tmp, '\n');
        if(!tmp.empty()) db_scripts_path.reset(tmp);
    }

    if( !db_scripts_path )
    {
        std::cerr << "Cannot find the db scripts path." << std::endl;
        return false;
    }

    std::string cl = *db_scripts_path + "/init_db.sh " + db_name + " 2>&1";

    std::string expect = exec("find ~/ -maxdepth 1 -name runwithpass.sh");
    erase_after(expect, '\n');
    if(!expect.empty()) cl = expect + " " + cl;

    int pec;
    std::string out = exec(cl.c_str(), pec);
    if(pec != 0)
    {
        std::cerr << "init_db.sh exit code is not 0" << std::endl;
        return false;
    }

    std::regex regex("ERROR");
    std::smatch match;
    if (std::regex_search(out, match, regex))
    {
        std::cerr << "Errors found in the init_db.sh output: " << out << std::endl;
        return false;
    }

    return true;
}


bool
TestEnv::init_test_db_98(char const *db_name)
{
    db_scripts_path.reset();
    char const *ctmp = std::getenv(EnvDbScriptsPathVar);
    if (ctmp != NULL) db_scripts_path = boost::make_optional( ctmp ) ;

    if (!db_scripts_path) {
        std::string tmp = exec("dirname `find ../../ -name 'init_db.sh'`");
        erase_after(tmp, '\n');
        if(!tmp.empty()) db_scripts_path.reset(tmp);
    }

    if( !db_scripts_path )
    {
        std::cerr << "Cannot find the db scripts path." << std::endl;
        return false;
    }

    std::string cl = *db_scripts_path + "/init_db.sh " + db_name + " 0.98 2>&1";

    std::string expect = exec("find ~/ -maxdepth 1 -name runwithpass.sh");
    erase_after(expect, '\n');
    if(!expect.empty()) cl = expect + " " + cl;

    int pec;
    std::string out = exec(cl.c_str(), pec);
    if(pec != 0)
    {
        std::cerr << "init_db.sh exit code is not 0" << std::endl;
        return false;
    }

    std::regex regex("error", std::regex::icase);
    std::smatch match;

    if (std::regex_search(out, match, regex))
    {
        std::cerr << "Errors found in the init_db.sh output: " << out << std::endl;
        return false;
    }

    return true;
}


bool TestEnv::prepareSvrConfig(char const * dbName, std::string const & dao_type, int max_loop_count)
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
            if (pec) return false;
        }
    }

    /**************************************************************************/
    // Setting up dao type

    {
        std::stringstream str_sed;
        str_sed << "sed -ri 's/(DAO_TYPE[ \\t]*=[ \\t]*)([a-zA-Z0-9_]+)/\\1" << dao_type << "/g' ";

        for (char const * cf : ConfigFiles)
        {
            std::stringstream str1;
            str1 << str_sed.str() << cf;
            std::string out = exec(str1.str().c_str(), pec);
            if (pec) return false;
        }
    }

/**************************************************************************/
    // Setting up iteration number

    {
        std::stringstream str_sed;
        str_sed << "sed -ri 's/(MAX_LOOP_COUNT[ \\t]*=[ \\t]*)([a-zA-Z0-9_-]+)/\\1" << max_loop_count << "/g' daemon.config";
        std::string out = exec(str_sed.str().c_str(), pec);
        if (pec) return false;
    }

    if (getenv("SVRTUNE")) {
        exec("echo 'TUNE_PARAMETERS = 1' >> daemon.config", pec);
        if (pec) return false;
        /*exec("echo 'TUNE_RUN_LIMIT = 1' >> daemon.config", pec);
        if (pec) return false;
        exec(std::string{svr::common::formatter() << "echo 'FUTURE_PREDICT_COUNT = " << getenv("TEST_VALIDATION_WINDOW") << "' >> daemon.config"}.c_str(), pec);
        if (pec) return false;
        exec(std::string{svr::common::formatter() << "echo 'SLIDE_COUNT = " << getenv("SLIDE_COUNT") << "' >> daemon.config"}.c_str(), pec);
        if (pec) return false;*/
    }
    return true;
}

#include <unistd.h>

bool TestEnv::run_daemon()
{
    auto cpid = fork();

    if (cpid == 0)
    {
        // Child process;
        auto exit_code = execl("./SVRDaemon", "./SVRDaemon", "--config", "daemon.config", NULL);
        LOG4_DEBUG("exit code (child): " << exit_code);
        if (exit_code != -1)
            exit_code = 0;
        else
            exit_code = 1;
        exit(exit_code);
    }
    else if (cpid > 0)
    {
        int status;
        auto w = waitpid(cpid, &status, WUNTRACED | WCONTINUED);
        if (w == -1)
            throw std::runtime_error("Cannot wait until child process ends.");

        LOG4_DEBUG("exit code received (parent): " << status);
        return status == 0;
    }
    else
        throw std::runtime_error("Cannot fork a child process");
}


// /usr/bin/valgrind --track-origins=yes --error-limit=no --suppressions=minimal.supp --log-file=minimalraw.log --gen-suppressions=all --vgdb-error=1 --leak-check=full --tool=memcheck --show-leak-kinds=all --max-stackframe=115062830400 CMD: ", "./SVRDaemon", "--config", "daemon.config", NULL

void TestEnv::run_daemon_nowait()
{
    auto cpid = fork();

    if (cpid == 0)
    {
        //Child process;
        //const auto exit_code = execl("/usr/bin/valgrind", "/usr/bin/valgrind", "--track-origins=yes", "--error-limit=no", "--suppressions=minimal.supp", "--log-file=minimalraw.log", "--gen-suppressions=all",
        //        /*"--vgdb-error=1",*/ "--leak-check=full", "--tool=memcheck", "--show-leak-kinds=all", "--max-stackframe=115062830400", "./SVRDaemon", "--config", "daemon.config", NULL);
        //const auto exit_code = execl("/usr/bin/valgrind", "--tool=exp-sgcheck", "./SVRDaemon", "--config", "daemon.config", NULL);
        //const auto exit_code = execl("/usr/bin/valgrind", "--tool=massif", "./SVRDaemon", "--config", "daemon.config", NULL);
        //const auto exit_code = execl("/home/zarko/pub/DrMemory-Linux-2.3.0-1/bin/drmemory", "/home/zarko/pub/DrMemory-Linux-2.3.0-1/bin/drmemory", "--", "./SVRDaemon", "--config", "daemon.config", NULL);
        //const auto exit_code = execl("/usr/bin/gdbserver", "/usr/bin/gdbserver", "0.0.0.0:10000", "./SVRDaemon", "--config", "daemon.config", NULL);
        const auto exit_code = execl("./SVRDaemon", "./SVRDaemon", "--config", "daemon.config", NULL); // Vanilla
        exit(exit_code != -1 ? 0 : 1);
    }
}



/******************************************************************************/


struct diagnostic_interface_alpha::diagnostic_interface_impl
{
    static std::string const mine_pipe_name, their_pipe_name;
    
    std::fstream mine_pipe;
    std::fstream their_pipe;
    
    diagnostic_interface_impl()
    {
        std::remove(mine_pipe_name.c_str());
        if(mkfifo(mine_pipe_name.c_str(), 0777) != 0)
            throw std::runtime_error("Cannot create the FIFO: " + mine_pipe_name);
        
        std::remove(their_pipe_name.c_str());
        if(mkfifo(their_pipe_name.c_str(), 0777) != 0)
            throw std::runtime_error("Cannot create the FIFO: " + their_pipe_name);

        mine_pipe.open(mine_pipe_name.c_str());
        
        mine_pipe << "M: start session" << std::endl;
    }
    
    ~diagnostic_interface_impl()
    {
        mine_pipe << "M: close session" << std::endl;
        mine_pipe.close();
        std::remove(mine_pipe_name.c_str());
    }
    
    void finish_construction()
    {
        do {
            their_pipe.clear();
            their_pipe.open(their_pipe_name.c_str());
        }
        while(their_pipe.bad());
    }
    
    void wait_iteration_finished()
    {
        std::string command;
        while(command.empty())
        {
            if(their_pipe.eof())
                their_pipe.clear();
            
            std::getline(their_pipe, command);
            if(command == "S: iteration finished")
                return;
            command.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    
    void trigger_next_iteration()
    {
        mine_pipe << "M: proceed to next iteration" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
};

std::string const diagnostic_interface_alpha::diagnostic_interface_impl::mine_pipe_name =  "/tmp/SVRDaemon_di_alpha";
std::string const diagnostic_interface_alpha::diagnostic_interface_impl::their_pipe_name = "/tmp/SVRDaemon_di_zwei";


diagnostic_interface_alpha::diagnostic_interface_alpha()
: impl (*new diagnostic_interface_impl())
{
}

diagnostic_interface_alpha::~diagnostic_interface_alpha()
{
    delete &impl;
}


void diagnostic_interface_alpha::finish_construction()
{
    impl.finish_construction();
}

void diagnostic_interface_alpha::wait_iteration_finished()
{
    impl.wait_iteration_finished();
}

void diagnostic_interface_alpha::trigger_next_iteration()
{
    impl.trigger_next_iteration();
}

