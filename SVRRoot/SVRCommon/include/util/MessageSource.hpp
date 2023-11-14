/*
 * MessageSource.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: vg
 */

#pragma once
#include "common/types.hpp"
#include <memory>

namespace svr{
namespace common{
class MessageSource{

public:
	virtual ~MessageSource(){}

	virtual const MessageProperties::mapped_type& readProperties(const std::string& propertyFile) = 0;
	virtual std::string getProperty(const std::string& propertyFile, const std::string& key, std::string defaultValue = "") = 0;
};
}
}

using MessageSource_ptr = std::shared_ptr<svr::common::MessageSource>;
