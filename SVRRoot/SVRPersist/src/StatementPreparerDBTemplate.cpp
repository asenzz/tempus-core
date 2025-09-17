#include "DAO/StatementPreparerDBTemplate.hpp"
#include "appcontext.hpp"

namespace svr {
namespace dao {

StatementPreparerDBTemplate::StatementPreparerDBTemplate(const bool is_file_db, const std::string &connection_str)
{
#ifdef USE_DUCKDB
    if (is_file_db) return;
#endif
    p_pg_con = new pqxx::connection(connection_str);
}

StatementPreparerDBTemplate::~StatementPreparerDBTemplate()
{
    if (p_pg_con)
        delete p_pg_con;
}

std::string StatementPreparerDBTemplate::prepare_statement(const char *format)
{
    return std::string(format);
}

std::string StatementPreparerDBTemplate::escape(const std::string &t)
{
    return "'" + esc(t) + "'";
}

std::string StatementPreparerDBTemplate::escape(const char *t)
{
    return "'" + esc(t) + "'";
}

std::string StatementPreparerDBTemplate::escape(datamodel::ROLE role)
{
    return std::string("'") + (role == datamodel::ROLE::ADMIN ? "ADMIN" : "USER") + "'";
}

std::string StatementPreparerDBTemplate::escape(const bpt::ptime &time)
{
    if (time.is_neg_infinity())
        return "-infinity";
    if (time.is_pos_infinity())
        return "infinity";
    if (time.is_not_a_date_time())
        return "NULL";
    if (time.is_special())
        THROW_EX_FS(std::invalid_argument, "Cannot transform special timestamp value to SQL type!");
    return "'" + bpt::to_simple_string(time) + "'::timestamp";
}

std::string StatementPreparerDBTemplate::escape(const bpt::time_duration &interval)
{
    return "'" + bpt::to_simple_string(interval) + "'::interval";
}

std::string StatementPreparerDBTemplate::escape(const bool &flag)
{
    return flag ? "TRUE" : "FALSE";
}

std::string StatementPreparerDBTemplate::escape(const datamodel::Priority &priority)
{
    return std::to_string((int) priority);
}

std::string StatementPreparerDBTemplate::escape(std::nullptr_t)
{
    return "null";
}

std::string StatementPreparerDBTemplate::escape(const std::shared_ptr<datamodel::OnlineSVR> &model)
{
    if (!model)
        return "''";
    std::stringstream s;
    datamodel::OnlineSVR::save(*model, s);
    return esc(s.str());
}

std::string StatementPreparerDBTemplate::escape(char c)
{
    std::string result;
    result += "'";
    result += c;
    result += "'";
    return result;
}

std::string StatementPreparerDBTemplate::escape(const long v)
{
    return std::to_string(v);
}

std::string StatementPreparerDBTemplate::escape(const bigint v)
{
    return std::to_string(v);
}

std::string StatementPreparerDBTemplate::escape(const float v)
{
    return common::to_string_with_precision(v);
}

std::string StatementPreparerDBTemplate::escape(const double v)
{
    return common::to_string_with_precision(v);
}

std::string StatementPreparerDBTemplate::prepare_statement(const std::string &format)
{
    return format;
}

} /* namespace dao */
} /* namespace svr */
