#ifndef STATEMENTPREPARERDBTEMPLATE_HPP
#define STATEMENTPREPARERDBTEMPLATE_HPP

#include <set>
#include "common.hpp"
#include "model/User.hpp"
#include "onlinesvr.hpp"


namespace bpt = boost::posix_time;

namespace svr {
namespace dao {

class StatementPreparerDBTemplate
{
    pqxx::connection *p_pg_con = nullptr;

    std::string escape(std::nullptr_t);

    std::string escape(const std::shared_ptr<datamodel::OnlineSVR> &model);

    std::string escape(const datamodel::Priority &priority);

    std::string escape(const bpt::ptime &);

    std::string escape(const bpt::time_duration &);

    std::string escape(const std::string &);

    std::string escape(const char *s);

    static std::string escape(char c);

    std::string escape(const bool &);

    static std::string escape(const long v);
    static std::string escape(const bigint v);
    static std::string escape(const float v);
    static std::string escape(const double v);
    std::string escape(datamodel::ROLE role);

    template<typename T> std::string escape(T t);

    template<typename InputIterator> std::deque<std::string> escape(InputIterator begin, InputIterator end);

    template<typename T> inline std::string escape(const std::vector<T> &v);
    
    template<typename T> inline std::string escape(const std::deque<T> &v);
    
    template<typename T> inline std::string escape(const std::set<T> &v);
    
    std::string prepare_statement(const char *format);

public:
    explicit StatementPreparerDBTemplate(const bool is_file_db, const std::string &connection_str);

    ~StatementPreparerDBTemplate();

    template<typename T> std::string esc(const T &str);
    
    template<typename T, typename... Targs> std::string prepare_statement(const char *format, T value, Targs... Fargs);
    
    template<typename T, typename... Targs> std::string prepare_statement(const std::string &format, T value, Targs... Fargs);
    
    static std::string prepare_statement(const std::string &format);
};

}
} // dao

#include "StatementPreparerDBTemplate.tpp"

#endif // STATEMENTPREPARERDBTEMPLATE_HPP
