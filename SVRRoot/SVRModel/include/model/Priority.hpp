#pragma once

#include <util/string_utils.hpp>

namespace svr{
namespace datamodel{

enum class Priority{ Low = 0, BelowNormal = 1, Normal = 2, AboveNormal = 3, High = 4, Max = 5};

inline Priority get_priority_from_string(const std::string& priority_str){
    Priority priority = Priority::Normal;

    using svr::common::ignoreCaseEquals;

    if (ignoreCaseEquals(priority_str, "Low")) {
        priority = Priority::Low;
    } else if (ignoreCaseEquals(priority_str, "BelowNormal")) {
        priority = Priority::BelowNormal;
    } else if (ignoreCaseEquals(priority_str, "Normal")) {
        priority = Priority::Normal;
    } else if (ignoreCaseEquals(priority_str, "AboveNormal")) {
        priority = Priority::AboveNormal;
    } else if (ignoreCaseEquals(priority_str, "High")) {
        priority = Priority::High;
    } else if (ignoreCaseEquals(priority_str, "Max")){
        priority = Priority::Max;
    }
    return priority;

}

inline const std::string to_string(const Priority& priority){
	switch(priority){
        case Priority::Max:         return "MAX";
        case Priority::High:        return "HIGH";
        case Priority::AboveNormal: return "ABOVE_NORMAL";
        case Priority::Normal:      return "NORMAL";
        case Priority::BelowNormal: return "BELOW_NORMAL";
        case Priority::Low:         return "LOW";
    }
    return "";
}

}
}
