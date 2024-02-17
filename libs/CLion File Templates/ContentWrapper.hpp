#pragma once

#[[#include]]# <view/MainView.hpp>
#[[#include]]# "model/${NAME}.hpp"

namespace content{
    struct ${NAME} : public Main{
        svr::datamodel::${NAME} object;
    };
}