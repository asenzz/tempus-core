#include <boost/algorithm/string/predicate.hpp>
#include "util/PropertiesFileReader.hpp"
#include "common/logging.hpp"
#include "common/constants.hpp"
#include "util/string_utils.hpp"


namespace svr {
namespace common {

tbb::mutex log_level_mx;

uint8_t AppConfig::S_log_threshold = (uint8_t) boost::log::trivial::severity_level::trace;

AppConfig::~AppConfig()
{
}

boost::log::trivial::severity_level AppConfig::set_global_log_level(const std::string &log_level_value)
{
    boost::log::trivial::severity_level log_threshold = boost::log::trivial::severity_level::info;
    if (strcasecmp(log_level_value.c_str(), "ALL") == 0)
        log_threshold = boost::log::trivial::severity_level::trace;
    else if (strcasecmp(log_level_value.c_str(), "TRACE") == 0)
        log_threshold = boost::log::trivial::severity_level::trace;
    else if (strcasecmp(log_level_value.c_str(), "DEBUG") == 0)
        log_threshold = boost::log::trivial::severity_level::debug;
    else if (strcasecmp(log_level_value.c_str(), "INFO") == 0)
        log_threshold = boost::log::trivial::severity_level::info;
    else if (strcasecmp(log_level_value.c_str(), "WARN") == 0)
        log_threshold = boost::log::trivial::severity_level::warning;
    else if (strcasecmp(log_level_value.c_str(), "ERROR") == 0)
        log_threshold = boost::log::trivial::severity_level::error;
    else if (strcasecmp(log_level_value.c_str(), "FATAL") == 0)
        log_threshold = boost::log::trivial::severity_level::fatal;
    const tbb::mutex::scoped_lock lk(log_level_mx);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_threshold);
    AppConfig::S_log_threshold = (uint8_t) log_threshold;
    return log_threshold;
}

boost::log::trivial::severity_level AppConfig::set_global_log_level(const boost::log::trivial::severity_level log_threshold)
{
    const tbb::mutex::scoped_lock lk(log_level_mx);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_threshold);
    AppConfig::S_log_threshold = (uint8_t) log_threshold;
    return log_threshold;
}

boost::log::trivial::severity_level AppConfig::set_global_log_level(const uint8_t log_value)
{
    const tbb::mutex::scoped_lock lk(log_level_mx);
    const auto log_threshold = static_cast<const boost::log::trivial::severity_level>(log_value);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_threshold);
    AppConfig::S_log_threshold = log_value;
    return log_threshold;
}

size_t AppConfig::get_default_feature_quantization() const noexcept
{ return feature_quantization_; }

double AppConfig::get_prediction_horizon() const noexcept
{ return prediction_horizon_; }

const std::string &AppConfig::get_db_connection_string() const noexcept
{ return db_connection_string_; }

bool AppConfig::get_set_thread_affinity() const noexcept
{ return set_thread_affinity_; }

size_t AppConfig::get_multistep_len() const noexcept
{ return multistep_len; }

size_t AppConfig::get_multiout() const noexcept
{ return multiout; }

size_t AppConfig::get_online_learn_iter_limit() const noexcept
{ return online_learn_iter_limit_; }

size_t AppConfig::get_stabilize_iterations_count() const noexcept
{ return stabilize_iterations_count_; }

double AppConfig::get_scaling_alpha() const noexcept
{ return scaling_alpha_; }

bool AppConfig::get_tune_parameters() const noexcept
{ return tune_parameters_; }

bool AppConfig::get_recombine_parameters() const noexcept
{ return recombine_parameters_; }

size_t AppConfig::get_slide_count() const noexcept
{ return slide_count_; }

size_t AppConfig::get_slide_skip() const noexcept
{ return slide_skip_; }

size_t AppConfig::get_tune_run_limit() const noexcept
{ return tune_run_limit_; }

boost::log::trivial::severity_level AppConfig::get_log_level() const noexcept
{ return log_level_; }

bool AppConfig::get_self_request() const noexcept
{ return self_request_; }

std::chrono::milliseconds AppConfig::get_loop_interval() const noexcept
{ return loop_interval_; }

std::chrono::milliseconds AppConfig::get_stream_loop_interval() const noexcept
{ return stream_loop_interval_; }

bool AppConfig::get_daemonize() const noexcept
{ return daemonize_; }

uint16_t AppConfig::get_num_quantisations() const noexcept
{ return num_quantisations_; }

uint16_t AppConfig::get_quantisation_divisor() const noexcept
{ return quantisation_divisor_; }

uint16_t AppConfig::get_oemd_quantisation_skipdiv() const noexcept
{ return oemd_quantisation_skipdiv_; }

float AppConfig::get_solve_iterations_coefficient() const noexcept
{ return solve_iterations_coefficient_; }

uint16_t AppConfig::get_oemd_tune_particles() const noexcept
{ return oemd_tune_particles_; }

uint16_t AppConfig::get_oemd_tune_iterations() const noexcept
{ return oemd_tune_iterations_; }

size_t PropertiesReader::read_property_file(std::string property_file_name)
{
    const tbb::mutex::scoped_lock lk(load_mx);
    if (property_files.find(property_file_name) != property_files.cend()) {
        LOG4_DEBUG("Properties file " << property_file_name << " already loaded");
        return property_files[property_file_name].size();
    }

    LOG4_TRACE("Reading properties from file " << property_file_name);
    MessageProperties::mapped_type params;

    std::ifstream is_file(property_files_location + property_file_name);
    if (!is_file.is_open()) is_file.open(property_file_name);
    if (!is_file.is_open()) THROW_EX_FS(std::invalid_argument, "Cannot read properties file " + property_file_name + " or " + property_files_location + property_file_name);
    std::string line, multi_line;
    bool is_multi_line = false;

    while (getline(is_file, line)) {
        trim(line);
        if (line.size() == 0) {
            is_multi_line = false;
            multi_line.clear();
            continue;
        }

        if (is_comment(line)) continue;

        if (is_multiline(line)) {
            is_multi_line = true;
            multi_line += line.substr(0, line.size() - 1);
            continue;
        }

        if (is_multi_line) { // final line of multiline property
            line = multi_line + line;
            is_multi_line = false;
            multi_line.clear();
        }
        std::istringstream is_line(line);
        std::string key;
        if (getline(is_line, key, delimiter)) {
            std::string value;
            if (getline(is_line, value)) params[trim(key)] = trim(value);
        }
    }

    const auto items = params.size();
    property_files[property_file_name] = params;
    LOG4_DEBUG("Read total of " << items << " from property file " << property_files_location << property_file_name);
    return items;
}

PropertiesReader::PropertiesReader(const char delimiter, const std::string &config_file) :
        delimiter(delimiter), config_file(config_file)
{
    if (config_file.empty()) THROW_EX_FS(std::invalid_argument, "Empty config file name");
    read_property_file(config_file);
    property_files_location = get_property<DTYPE(property_files_location) >(config_file, SQL_PROPERTIES_DIR_KEY, C_default_sql_properties_dir);
}

// TODO Move hardcoded values to header file using the CONFPROP macro
AppConfig::AppConfig(const std::string &app_config_file, const char delimiter) :
        PropertiesReader(delimiter, app_config_file), dao_type(ConcreteDaoType::PgDao)
{
    feature_quantization_ = get_property<DTYPE(feature_quantization_) >(app_config_file, FEATURE_QUANTIZATION, C_default_feature_quantization_str);
    prediction_horizon_ = get_property<DTYPE(prediction_horizon_) >(app_config_file, PREDICTION_HORIZON, C_default_prediction_horizon_str);
    recombine_parameters_ = get_property<DTYPE(recombine_parameters_) >(app_config_file, RECOMBINE_PARAMETERS, C_default_recombine_parameters_str);
    tune_parameters_ = get_property<DTYPE(tune_parameters_) >(app_config_file, TUNE_PARAMETERS, C_default_tune_parameters_str);
    log_level_ = set_global_log_level(get_property<std::string>(app_config_file, LOG_LEVEL_KEY, C_default_log_level));
    dao_type = get_property<std::string>(app_config_file, DAO_TYPE_KEY, C_default_DAO_type) == "async" ? ConcreteDaoType::AsyncDao : ConcreteDaoType::PgDao;
    set_thread_affinity_ = get_property<DTYPE(set_thread_affinity_) >(app_config_file, SET_THREAD_AFFINITY, "0");
    multistep_len = get_property<DTYPE(multistep_len) >(app_config_file, MULTISTEP_LEN, C_default_multistep_len_str);
    multiout = get_property<DTYPE(multiout) >(app_config_file, MULTIOUT, C_default_multiout_str);
    online_learn_iter_limit_ = get_property<DTYPE(online_learn_iter_limit_) >(app_config_file, ONLINE_LEARN_ITER_LIMIT, C_default_online_iter_limit_str);
    stabilize_iterations_count_ = get_property<DTYPE(stabilize_iterations_count_) >(app_config_file, STABILIZE_ITERATIONS_COUNT, C_default_stabilize_iterations_count_str);
    slide_count_ = get_property<DTYPE(slide_count_) >(app_config_file, SLIDE_COUNT, C_default_slide_count_str);
    tune_run_limit_ = get_property<DTYPE(tune_run_limit_) >(app_config_file, TUNE_RUN_LIMIT, C_default_tune_run_limit_str);
    self_request_ = get_property<DTYPE(self_request_) >(app_config_file, SELF_REQUEST, "0");
    scaling_alpha_ = get_property<DTYPE(scaling_alpha_) >(app_config_file, SCALING_ALPHA, C_default_scaling_alpha_str);
    db_connection_string_ = get_property<DTYPE(db_connection_string_) >(app_config_file, CONNECTION_STRING, C_default_connection_str);
    loop_interval_ = std::chrono::milliseconds(get_property<long>(app_config_file, LOOP_INTERVAL, C_default_loop_interval_ms));
    stream_loop_interval_ = std::chrono::milliseconds(get_property<long>(app_config_file, STREAM_LOOP_INTERVAL, C_default_stream_loop_interval_ms));
    daemonize_ = get_property<DTYPE(daemonize_) >(app_config_file, DAEMONIZE, C_default_daemonize);
    num_quantisations_ = get_property<DTYPE(num_quantisations_) >(app_config_file, NUM_QUANTISATIONS, C_default_num_quantisations);
    quantisation_divisor_ = get_property<DTYPE(quantisation_divisor_) >(app_config_file, QUANTISATION_DIVISOR, C_default_quantisation_divisor);
    oemd_quantisation_skipdiv_ = get_property<DTYPE(oemd_quantisation_skipdiv_) >(app_config_file, OEMD_QUANTISATION_SKIPDIV, C_default_oemd_quantisation_skipdiv);
    oemd_tune_particles_ = get_property<DTYPE(oemd_tune_particles_) >(app_config_file, OEMD_TUNE_PARTICLES, C_default_oemd_tune_particles);
    oemd_tune_iterations_ = get_property<DTYPE(oemd_tune_iterations_) >(app_config_file, OEMD_TUNE_ITERATIONS, C_default_oemd_tune_iterations);
    solve_iterations_coefficient_ = get_property<DTYPE(solve_iterations_coefficient_) >(app_config_file, SOLVE_ITERATIONS_COEFFICIENT, C_defaut_solve_iterations_coefficient);
}

ConcreteDaoType AppConfig::get_dao_type() const noexcept
{
    return dao_type;
}

const MessageProperties::mapped_type &PropertiesReader::read_properties(const std::string &property_file)
{
    if (property_files.count(property_file) || read_property_file(property_file)) return property_files[property_file];
    static const MessageProperties::mapped_type empty;
    return empty;
}


bool PropertiesReader::is_comment(const std::string &line)
{
    return boost::starts_with(line, COMMENT_CHARS);
}


bool PropertiesReader::is_multiline(const std::string &line)
{
    if (line.empty()) return false;
    bool even_slash_count = true;
    auto c = line.crbegin();
    while (c != line.crend() && *c == '\\') {
        even_slash_count = !even_slash_count;
        ++c;
    }
    return !even_slash_count;
}

const std::string &PropertiesReader::get_property_value(const std::string &property_file, const std::string &key, const std::string &default_value)
{
    LOG4_TRACE("Getting property value for key " << key << " from property file " << property_file);
    const auto find_property_file = [&property_file](const auto &p) { return p.first == property_file; };
    bool file_loaded = false;
    for (auto property_file_it = std::find_if(C_default_exec_policy, property_files.cbegin(), property_files.cend(), find_property_file);
         property_file_it != property_files.cend();
         property_file_it = std::find_if(C_default_exec_policy, ++property_file_it, property_files.cend(), find_property_file)) {
        file_loaded = true;
        LOG4_TRACE("Checking property file " << property_file_it->first);
        const auto property_it = property_file_it->second.find(key);
        if (property_it != property_file_it->second.cend()) {
            LOG4_TRACE("Found property " << key << " in file " << property_file_it->first << ", with value " << property_it->second);
            return property_it->second;
        }
    }
    LOG4_DEBUG("Loading property file " << property_file);
    if (!file_loaded && read_property_file(property_file)) {
        const auto property_it = property_files[property_file].find(key);
        if (property_it != property_files[property_file].cend()) return property_it->second;
    }
    LOG4_TRACE("Property " << key << " not found, returning default value " << default_value);
    return default_value;
}

} /* namespace common */
} /* namespace svr */