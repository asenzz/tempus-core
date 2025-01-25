#ifndef DBUTILS_H
#define DBUTILS_H

#include <string>
#include <boost/optional.hpp>

using opt_string = boost::optional<std::string>;

std::string exec(char const * cmd);
std::string exec(char const * cmd, int & exit_code);

bool erase_after(std::string & where, char what);

struct TestEnv
{
    opt_string db_scripts_path;
    bool dbInitialized;

    static constexpr char const * EnvDbScriptsPathVar   {"TEMPUS_DB_SCRIPTS_PATH"};
    static constexpr char const * ConfigFiles[]         {"app.config"};
    static constexpr char const * TestDbUserName        {"tempustest"};
    static constexpr char const * AppConfigFile         {"app.config"};
    static TestEnv& global_test_env()
    {
        static TestEnv env;
        return env;
    }

    bool init_test_db(char const *db_name);

    bool prepareSvrConfig(char const * dbName, std::string const & dao_type);
};

#endif /* DBUTILS_H */

