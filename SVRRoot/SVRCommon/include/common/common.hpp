#pragma once

// common external dependencies
#include <iostream>
#include <fstream> // file stream
#include <sstream> // (i/o)stringstream
#include <memory> // for shared and unique pointers
#include <typeinfo>

// boost
#include <boost/lexical_cast.hpp>

// common internal dependencies
#include "common/compatibility.hpp" // defines missing C++ features or replacement for another
#include "common/Logging.hpp"
#include "common/defines.h"
#include "common/constants.hpp"
#include "common/types.hpp"
#include "common/exceptions.hpp"
#include "common/parallelism.hpp"
