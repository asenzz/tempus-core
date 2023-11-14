#include <cppcms/json.h>
#include <util/string_utils.hpp>

namespace content{

struct Tick{
    std::string symbol;
    bpt::ptime time;
    double high;
    double low;
    double open;
    double close;
    long volume;
    double weight;
    bool isFinal;
    int period;
    bpt::ptime clientTime;

};


}

namespace cppcms {
namespace json {

// We specilize cppcms::json::traits structure to convert
// objects to and from json values

template<>
struct traits<content::Tick> {
    static content::Tick get(value const &v)
    {
        content::Tick tick;
        if(v.type()!=is_object)
            throw bad_value_cast();

        tick.symbol = v.get<std::string>("symbol");
        tick.time = bpt::time_from_string(v.get<std::string>("time"));
        tick.open = std::stod(v.get<std::string>("open"));
        tick.high = std::stod(v.get<std::string>("high"));
        tick.low = std::stod(v.get<std::string>("low"));
        tick.close = std::stod(v.get<std::string>("close"));
        tick.weight = std::stod(v.get<std::string>("tick_volume"));
        tick.isFinal = svr::common::ignoreCaseEquals(v.get<std::string>("isFinal"), "true");
        tick.period = std::stoi(v.get<std::string>("period"));
        tick.clientTime = bpt::time_from_string(v.get<std::string>("clientTime"));

        return tick;
    }
    static void set(value &v,content::Tick const &in)
    {
        throw std::runtime_error("Not implemented!");
    }
};
} // json
} // cppcms