#include "util/PropertiesFileReader.hpp"
#include "common/Logging.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include "util/string_utils.hpp"
#include <common/constants.hpp>


// this is inherited dependency for logging output level
int SVR_LOG_LEVEL;

namespace svr {
namespace common {

const std::string PropertiesFileReader::SQL_PROPERTIES_DIR_KEY = "SQL_PROPERTIES_DIR";
const std::string PropertiesFileReader::LOG_LEVEL_KEY = "LOG_LEVEL";
const std::string PropertiesFileReader::DAO_TYPE_KEY = "DAO_TYPE";
const std::string PropertiesFileReader::COMMENT_CHARS = "#";
const std::string PropertiesFileReader::SET_THREAD_AFFINITY = "SET_THREAD_AFFINITY";
const std::string PropertiesFileReader::MULTISTEP_LEN = "MULTISTEP_LEN";
const std::string PropertiesFileReader::ONLINE_LEARN_ITER_LIMIT = "ONLINE_LEARN_ITER_LIMIT";
const std::string PropertiesFileReader::STABILIZE_ITERATIONS_COUNT = "STABILIZE_ITERATIONS_COUNT";
const std::string PropertiesFileReader::ERROR_TOLERANCE = "ERROR_TOLERANCE";
const std::string PropertiesFileReader::SCALING_ALPHA = "SCALING_ALPHA";
const std::string PropertiesFileReader::ONLINESVR_LOG_FILE = "ONLINESVR_LOG_FILE";
const std::string PropertiesFileReader::SLIDE_COUNT = "SLIDE_COUNT";
const std::string PropertiesFileReader::TUNE_RUN_LIMIT = "TUNE_RUN_LIMIT";


size_t PropertiesFileReader::read_property_file(std::string property_file_name)
{
    LOG4_TRACE("Reading properties from file " << property_file_name);
	MessageProperties::mapped_type params;

	std::ifstream is_file(property_files_location + property_file_name);

	if(!is_file.is_open()) {
		throw std::invalid_argument("Cannot read properties file: " + property_files_location + property_file_name);
	}
	std::string line;
    std::string multi_line;
	bool is_multi_line = false;

	while (getline(is_file, line)) {
		trim(line);
		if(line.size() == 0) {
			is_multi_line = false;
			multi_line.clear();
			continue;
		}

		if(is_comment(line)){
			continue;
		}

		if(is_multiline(line)){
			is_multi_line = true;
			multi_line += line.substr(0, line.size()-1);
			continue;
		}

		if(is_multi_line){ // final line of multiline property
			line = multi_line + line;
			is_multi_line = false;
			multi_line.clear();
		}
		std::istringstream is_line(line);
		std::string key;
		if (getline(is_line, key, delimiter)) {
			std::string value;
			if (getline(is_line, value)){
				params[trim(key)] = trim(value);
			}
		}
	}

	size_t items = params.size();
	property_files[property_file_name] = params;
	LOG4_DEBUG("Read total of " << items << " from property file " << property_files_location << property_file_name);
	return items;
}

// TODO Move hardcoded values to header file
PropertiesFileReader::PropertiesFileReader(const std::string& app_config_file, char delimiter):
		delimiter(delimiter), dao_type(ConcreteDaoType::PgDao)
{
#ifndef NDEBUG
	//__cilkrts_set_param("nworkers", "1");
#endif
	read_property_file(app_config_file);
	property_files_location = get_property<std::string>(app_config_file, SQL_PROPERTIES_DIR_KEY, DEFAULT_SQL_PROPERTIES_DIR_KEY);
	set_global_log_level(get_property<std::string>(app_config_file, LOG_LEVEL_KEY, DEFAULT_LOG_LEVEL_KEY));
	std::string sdao_type = get_property<std::string>(app_config_file, DAO_TYPE_KEY, DEFAULT_DAO_TYPE_KEY);
	if (sdao_type == "async") dao_type = ConcreteDaoType::AsyncDao;
    set_thread_affinity_ = get_property<bool>(app_config_file, SET_THREAD_AFFINITY, "0");
    multistep_len = get_property<size_t>(app_config_file, MULTISTEP_LEN, DEFAULT_MULTISTEP_LEN);
    online_learn_iter_limit_ = get_property<size_t>(app_config_file, ONLINE_LEARN_ITER_LIMIT, DEFAULT_ONLINE_ITER_LIMIT);
    stabilize_iterations_count_ = get_property<size_t>(app_config_file, STABILIZE_ITERATIONS_COUNT, DEFAULT_STABILIZE_ITERATIONS_COUNT);
    error_tolerance_ = get_property<double>(app_config_file, ERROR_TOLERANCE, DEFAULT_ERROR_TOLERANCE);
    online_svr_log_file_ = get_property<std::string>(app_config_file, ONLINESVR_LOG_FILE, DEFAULT_ONLINESVR_LOG_FILE);
    slide_count_ = get_property<size_t>(app_config_file, SLIDE_COUNT, DEFAULT_SLIDE_COUNT);
	tune_run_limit_ = get_property<size_t>(app_config_file, TUNE_RUN_LIMIT, DEFAULT_TUNE_RUN_LIMIT);
	scaling_alpha_ = get_property<double>(app_config_file, SCALING_ALPHA, DEFAULT_SCALING_ALPHA);
}

ConcreteDaoType PropertiesFileReader::get_dao_type() const
{
    return dao_type;
}

const MessageProperties::mapped_type& PropertiesFileReader::read_properties(const std::string &property_file)
{
    if(property_files.count(property_file) || read_property_file(property_file))
        return property_files[property_file];
    static MessageProperties::mapped_type empty;
    return empty;
}


void PropertiesFileReader::set_global_log_level(const std::string &log_level_value)
{
    if(strcasecmp(log_level_value.c_str(), "ALL") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::TRACE;
    else if(strcasecmp(log_level_value.c_str(), "TRACE") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::TRACE;
    else if(strcasecmp(log_level_value.c_str(), "DEBUG") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::DEBUG;
    else if(strcasecmp(log_level_value.c_str(), "INFO") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::INFO;
    else if(strcasecmp(log_level_value.c_str(), "WARN") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::WARN;
    else if(strcasecmp(log_level_value.c_str(), "ERROR") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::ERR;
    else if(strcasecmp(log_level_value.c_str(), "FATAL") == 0)
        SVR_LOG_LEVEL = LOG_LEVEL_T::FATAL;
}

bool PropertiesFileReader::is_comment(const std::string& line)
{
	return boost::starts_with(line, COMMENT_CHARS);
}


bool PropertiesFileReader::is_multiline(const std::string &line)
{
	if(line.size() == 0){
		return false;
	}

	bool even_slash_count = true;
	auto c = line.rbegin();
	while(c != line.rend() && *c == '\\'){
		even_slash_count = !even_slash_count;
		c++;
	}
	return !even_slash_count;
}

std::string PropertiesFileReader::get_property_value(
		const std::string &property_file, const std::string &key, std::string default_value)
{
	LOG4_BEGIN();

	if (property_files.count(property_file) == 0 && read_properties(property_file).size() == 0) return default_value;
	if (property_files.count(property_file) && property_files[property_file].count(key))
		default_value = property_files[property_file][key];
	LOG4_TRACE("Found property " << property_file << " " << key << " is " << default_value);

	LOG4_END();

	return default_value;
}

} /* namespace common */
} /* namespace svr */
