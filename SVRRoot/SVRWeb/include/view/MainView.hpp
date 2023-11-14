#pragma once

#include <cppcms/view.h>

namespace content{
    struct Main : public cppcms::base_content{
        std::string pageTitle;
        std::string pageError;
        std::string subTitle;
    };
}

