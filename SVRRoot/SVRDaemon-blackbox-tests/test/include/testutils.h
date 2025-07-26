#ifndef DBUTILS_H
#define DBUTILS_H

#include <string>
#include <boost/optional.hpp>

std::string exec(char const * cmd);
std::string exec(char const * cmd, int & exit_code);
bool erase_after(std::string & where, char what);

struct TestEnv
{
    boost::optional<std::string> db_scripts_path;
    bool dbInitialized;
    std::string dao_type;

    static constexpr char const * EnvDbScriptsPathVar   {"TEMPUS_DB_SCRIPTS_PATH"};
    static constexpr char const * ConfigFiles[]         {"daemon.config"};
    static constexpr char const * TestDbUserName        {"tempustest"};
    static constexpr char const * AppConfigFile         {"daemon.config"};

    bool init_test_db(char const *db_name);
    bool init_test_db_98(char const *db_name);

    bool prepareSvrConfig(char const * dbName, std::string const & dao_type, int max_loop_count);

    static bool run_daemon();

    static void run_daemon_nowait();

    static TestEnv& global_test_env();
};


class diagnostic_interface_alpha
{
public:
    diagnostic_interface_alpha();
    ~diagnostic_interface_alpha();
    void finish_construction() const;
    void wait_iteration_finished() const;
    void trigger_next_iteration() const;
private:
    struct diagnostic_interface_impl;
    diagnostic_interface_impl & impl;
};


#endif /* DBUTILS_H */

