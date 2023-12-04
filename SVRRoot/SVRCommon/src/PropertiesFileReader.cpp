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
const std::string PropertiesFileReader::DONT_UPDATE_R_MATRIX = "DONT_UPDATE_R_MATRIX";
const std::string PropertiesFileReader::MAIN_COLUMNS_AUX = "MAIN_COLUMNS_AUX";
const std::string PropertiesFileReader::MAX_SMO_ITERATIONS = "MAX_SMO_ITERATIONS";
const std::string PropertiesFileReader::CASCADE_REDUCE_RATIO = "CASCADE_REDUCE_RATIO";
const std::string PropertiesFileReader::CASCADE_MAX_SEGMENT_SIZE = "CASCADE_MAX_SEGMENT_SIZE";
const std::string PropertiesFileReader::DISABLE_CASCADED_SVM = "DISABLE_CASCADED_SVM";
const std::string PropertiesFileReader::CASCADE_BRANCHES_COUNT = "CASCADE_BRANCHES_COUNT";
const std::string PropertiesFileReader::SET_THREAD_AFFINITY = "SET_THREAD_AFFINITY";
const std::string PropertiesFileReader::MULTISTEP_LEN = "MULTISTEP_LEN";
const std::string PropertiesFileReader::SVR_PARAMTUNE_LEVEL = "SVR_PARAMTUNE_LEVEL";
const std::string PropertiesFileReader::SVR_PARAMTUNE_COLUMN = "SVR_PARAMTUNE_COLUMN";
const std::string PropertiesFileReader::ONLINE_ITERS_LIMIT_MULT = "ONLINE_ITERS_LIMIT_MULT";
const std::string PropertiesFileReader::ONLINE_LEARN_ITER_LIMIT = "ONLINE_LEARN_ITER_LIMIT";
const std::string PropertiesFileReader::SMO_EPSILON_DIVISOR = "SMO_EPSILON_DIVISOR";
const std::string PropertiesFileReader::SMO_COST_DIVISOR = "SMO_COST_DIVISOR";
const std::string PropertiesFileReader::STABILIZE_ITERATIONS_COUNT = "STABILIZE_ITERATIONS_COUNT";
const std::string PropertiesFileReader::DEFAULT_NUMBER_VARIATIONS = "DEFAULT_NUMBER_VARIATIONS";
const std::string PropertiesFileReader::ERROR_TOLERANCE = "ERROR_TOLERANCE";
const std::string PropertiesFileReader::SCALING_ALPHA = "SCALING_ALPHA";
const std::string PropertiesFileReader::ONLINESVR_LOG_FILE = "ONLINESVR_LOG_FILE";
const std::string PropertiesFileReader::MAX_VARIATIONS = "MAX_VARIATIONS";
const std::string PropertiesFileReader::COMB_TRAIN_COUNT = "COMB_TRAIN_COUNT";
const std::string PropertiesFileReader::COMB_VALIDATE_COUNT = "COMB_VALIDATE_COUNT";
const std::string PropertiesFileReader::COMB_VALIDATE_LIMIT = "COMB_VALIDATE_LIMIT";
const std::string PropertiesFileReader::ENABLE_COMB_VALIDATE = "ENABLE_COMB_VALIDATE";
const std::string PropertiesFileReader::FUTURE_PREDICT_COUNT = "FUTURE_PREDICT_COUNT";
const std::string PropertiesFileReader::SLIDE_COUNT = "SLIDE_COUNT";
const std::string PropertiesFileReader::TUNE_RUN_LIMIT = "TUNE_RUN_LIMIT";
const std::string PropertiesFileReader::OEMD_FIND_FIR_COEFFICIENTS = "OEMD_FIND_FIR_COEFFICIENTS";
const std::string PropertiesFileReader::TUNE_PARAMETERS = "TUNE_PARAMETERS";
const std::string PropertiesFileReader::ALL_AUX_LEVELS = "ALL_AUX_LEVELS";


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
		if (getline(is_line, key, this->delimiter)) {
			std::string value;
			if (getline(is_line, value)){
				params[trim(key)] = trim(value);
			}
		}
	}

	size_t items = params.size();
	this->property_files[property_file_name] = params;
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
	dont_update_r_matrix_ = get_property<bool>(app_config_file, DONT_UPDATE_R_MATRIX, DEFAULT_DONT_UPDATE_R_MATRIX);
    main_columns_aux_ = get_property<bool>(app_config_file, MAIN_COLUMNS_AUX, DEFAULT_MAIN_COLUMNS_AUX);
	max_smo_iterations_ = get_property<size_t>(app_config_file, MAX_SMO_ITERATIONS, DEFAULT_MAX_SMO_ITERATIONS);
	cascade_reduce_ratio_ = get_property<double>(app_config_file, CASCADE_REDUCE_RATIO, DEFAULT_CASCADE_REDUCE_RATIO);
    cascade_branches_count_ = get_property<size_t>(app_config_file, CASCADE_BRANCHES_COUNT, DEFAULT_CASCADE_BRANCHES_COUNT);
    disable_cascaded_svm_ = get_property<size_t>(app_config_file, DISABLE_CASCADED_SVM, DEFAULT_DISABLE_CASCADED_SVM);
	cascade_max_segment_size_ = get_property<size_t>(app_config_file, CASCADE_MAX_SEGMENT_SIZE, DEFAULT_CASCADE_MAX_SEGMENT_SIZE);
    set_thread_affinity_ = get_property<bool>(app_config_file, SET_THREAD_AFFINITY, "0");
    multistep_len = get_property<size_t>(app_config_file, MULTISTEP_LEN, DEFAULT_MULTISTEP_LEN);
    svr_paramtune_column = get_property<std::string>(app_config_file, SVR_PARAMTUNE_COLUMN, DEFAULT_SVR_PARAMTUNE_COLUMN);
    svr_paramtune_level = get_property<std::string>(app_config_file, SVR_PARAMTUNE_LEVEL, DEFAULT_SVR_PARAMTUNE_LEVEL);
    online_iters_limit_mult_ = get_property<size_t>(app_config_file, ONLINE_ITERS_LIMIT_MULT, DEFAULT_ONLINE_ITERS_LIMIT_MULT);
    online_learn_iter_limit_ = get_property<size_t>(app_config_file, ONLINE_LEARN_ITER_LIMIT, DEFAULT_LEARN_ITER_LIMIT);
    smo_epsilon_divisor_ = get_property<double>(app_config_file, SMO_EPSILON_DIVISOR, DEFAULT_SMO_EPSILON_DIVISOR);
    smo_cost_divisor_ = get_property<double>(app_config_file, SMO_COST_DIVISOR, DEFAULT_SMO_COST_DIVISOR);
    stabilize_iterations_count_ = get_property<size_t>(app_config_file, STABILIZE_ITERATIONS_COUNT, DEFAULT_STABILIZE_ITERATIONS_COUNT);
    default_number_variations_ = get_property<size_t>(app_config_file, DEFAULT_NUMBER_VARIATIONS, DEFAULT_DEFAULT_NUMBER_VARIATIONS);
    error_tolerance_ = get_property<double>(app_config_file, ERROR_TOLERANCE, DEFAULT_ERROR_TOLERANCE);
    online_svr_log_file_ = get_property<std::string>(app_config_file, ONLINESVR_LOG_FILE, DEFAULT_ONLINESVR_LOG_FILE);
    max_variations_ = get_property<size_t>(app_config_file, MAX_VARIATIONS, DEFAULT_MAX_VARIATIONS);
	comb_train_count_ = get_property<size_t>(app_config_file, COMB_TRAIN_COUNT, DEFAULT_COMB_TRAIN_COUNT);
	comb_validate_count_ = get_property<size_t>(app_config_file, COMB_VALIDATE_COUNT, DEFAULT_COMB_VALIDATE_COUNT);
	comb_validate_limit_ = get_property<size_t>(app_config_file, COMB_VALIDATE_LIMIT, DEFAULT_COMB_VALIDATE_LIMIT);
	enable_comb_validate_ = get_property<bool>(app_config_file, ENABLE_COMB_VALIDATE, DEFAULT_ENABLE_COMB_VALIDATE);
    tune_parameters_ = get_property<bool>(app_config_file, TUNE_PARAMETERS, DEFAULT_TUNE_PARAMETERS);
    future_predict_count_ = get_property<size_t>(app_config_file, FUTURE_PREDICT_COUNT, DEFAULT_FUTURE_PREDICT_COUNT);
    slide_count_ = get_property<size_t>(app_config_file, SLIDE_COUNT, DEFAULT_SLIDE_COUNT);
	tune_run_limit_ = get_property<size_t>(app_config_file, TUNE_RUN_LIMIT, DEFAULT_TUNE_RUN_LIMIT);
	scaling_alpha_ = get_property<double>(app_config_file, SCALING_ALPHA, DEFAULT_SCALING_ALPHA);
	all_aux_levels_ = get_property<bool>(app_config_file, ALL_AUX_LEVELS, DEFAULT_ALL_AUX_LEVELS);
    oemd_find_fir_coefficients_ = get_property<bool>(app_config_file, OEMD_FIND_FIR_COEFFICIENTS, DEFAULT_OEMD_FIND_FIR_COEFFICIENTS);
}

ConcreteDaoType PropertiesFileReader::get_dao_type() const
{
    return dao_type;
}

const MessageProperties::mapped_type& PropertiesFileReader::read_properties(const std::string &property_file)
{
    if(this->property_files.count(property_file) || read_property_file(property_file))
        return this->property_files[property_file];
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

	if(this->property_files.count(property_file) == 0 && read_properties(property_file).size() == 0) return default_value;
	if(this->property_files.count(property_file) && this->property_files[property_file].count(key))
		default_value = this->property_files[property_file][key];
	LOG4_TRACE("Found property " << property_file << " " << key << " is " << default_value);

	LOG4_END();

	return default_value;
}

std::vector<size_t> PropertiesFileReader::get_svr_paramtune_levels() const
{
    LOG4_BEGIN();

    std::stringstream iss(svr_paramtune_level);
    std::vector<size_t> result;
    size_t num;
    while (iss >> num) result.push_back(num);
    return result;
}
} /* namespace common */
} /* namespace svr */
