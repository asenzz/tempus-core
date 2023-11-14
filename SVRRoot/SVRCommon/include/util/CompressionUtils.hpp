/*
 * CompressionUtils.hpp
 *
 *  Created on: April 8, 2015
 *      Author: skondrat
 */

#pragma once

#include <string>
#include <sstream>
#include <lz4.h>
#include <algorithm>

namespace svr {
namespace common {

std::string base64_encode(unsigned char const* , unsigned int len);
std::string base64_decode(const std::string & encodedString);

std::string compress(const char * input, int size);
std::string decompress(const char * input, int size);

}
}
