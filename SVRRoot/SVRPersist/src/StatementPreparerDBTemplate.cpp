#include "DAO/StatementPreparerDBTemplate.hpp"

using pqxx::to_string;
using std::shared_ptr;
using std::string;
using bpt::ptime;
using bpt::time_duration;

namespace svr {
namespace dao {


string StatementPreparerDBTemplate::prepareStatement(const char *format)
{
    return string(format);
}

string StatementPreparerDBTemplate::escape(const string &t)
{
//    if (!connection.is_open()) connection = pqxx::connection(connection.connection_string());
    return "'" + connection.esc(t) + "'";
}

string StatementPreparerDBTemplate::escape(const char *t)
{
//    if (!connection.is_open()) connection = pqxx::connection(connection.connection_string());
    return "'" + connection.esc(string(t)) + "'";
}

string StatementPreparerDBTemplate::escape(svr::datamodel::ROLE role)
{
    return string("'") + (role == svr::datamodel::ROLE::ADMIN ? "ADMIN" : "USER") + "'";
}

string StatementPreparerDBTemplate::escape(const ptime &time)
{
    if (time.is_neg_infinity()) {
        return "-infinity";
    }
    if (time.is_pos_infinity()) {
        return "infinity";
    }
    if (time.is_not_a_date_time()) {
        return "NULL";
    }
    if (time.is_special()) {
        throw std::invalid_argument("Cannot transform special timestamp value to SQL type!");
    }
    return "'" + to_simple_string(time) + "'::timestamp";
}

string StatementPreparerDBTemplate::escape(const time_duration &interval)
{
    return "'" + to_simple_string(interval) + "'::interval";
}

string StatementPreparerDBTemplate::escape(const bool &flag)
{
    return flag ? "TRUE" : "FALSE";
}

std::string StatementPreparerDBTemplate::escape(const svr::datamodel::Priority &priority)
{
    return std::to_string((int) priority);
}

std::string StatementPreparerDBTemplate::escape(std::nullptr_t)
{
    return "null";
}

std::string StatementPreparerDBTemplate::escape(const std::shared_ptr<svr::OnlineMIMOSVR> &model)
{
    if (!model) return "''";
    std::stringstream s;
    OnlineMIMOSVR::save(*model, s);
    return connection.quote(s);
}

} /* namespace dao */
} /* namespace svr */

