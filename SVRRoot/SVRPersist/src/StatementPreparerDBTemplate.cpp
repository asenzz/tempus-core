#include "DAO/StatementPreparerDBTemplate.hpp"


namespace svr {
namespace dao {


std::string StatementPreparerDBTemplate::prepare_statement(const char *format)
{
    return std::string(format);
}

std::string StatementPreparerDBTemplate::escape(const std::string &t)
{
    return "'" + connection.esc(t) + "'";
}

std::string StatementPreparerDBTemplate::escape(const char *t)
{
    return "'" + connection.esc(std::string(t)) + "'";
}

std::string StatementPreparerDBTemplate::escape(datamodel::ROLE role)
{
    return std::string("'") + (role == datamodel::ROLE::ADMIN ? "ADMIN" : "USER") + "'";
}

std::string StatementPreparerDBTemplate::escape(const bpt::ptime &time)
{
    if (time.is_neg_infinity()) return "-infinity";
    if (time.is_pos_infinity()) return "infinity";
    if (time.is_not_a_date_time()) return "NULL";
    if (time.is_special()) THROW_EX_FS(std::invalid_argument, "Cannot transform special timestamp value to SQL type!");
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
    if (!model) return "''";
    std::stringstream s;
    datamodel::OnlineSVR::save(*model, s);
    return connection.quote(s);
}

} /* namespace dao */
} /* namespace svr */

