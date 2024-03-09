#pragma once

#include <common/types.hpp>
#include <string>

namespace svr {
namespace common {

enum class ConcreteDaoType { PgDao, AsyncDao };

class PropertiesFileReader
{
    bool is_autotune_running_ = false;

    MessageProperties property_files;
    char delimiter;
    std::string property_files_location;
    ConcreteDaoType dao_type;
    bool set_thread_affinity_;
    size_t multistep_len;
    size_t online_learn_iter_limit_;
    size_t stabilize_iterations_count_;
    double error_tolerance_;
    double scaling_alpha_;
    std::string online_svr_log_file_;
    bool tune_parameters_;
    size_t slide_count_;
    size_t slide_skip_;
    size_t validation_window_;
    size_t tune_run_limit_;

    size_t read_property_file(std::string property_file_name);
    void set_global_log_level(const std::string& log_level_value);
    bool is_comment(const std::string &line);
    bool is_multiline(const std::string &line);

    std::string get_property_value(
            const std::string &property_file, const std::string &key, std::string default_value);

public:
    virtual ~PropertiesFileReader() {}

    explicit PropertiesFileReader(const std::string& app_config_file, char delimiter = '=');
    const MessageProperties::mapped_type& read_properties(const std::string &property_file);

    template<typename T>
    T get_property(const std::string &property_file, const std::string &key, std::string default_value = "") {
        return boost::lexical_cast<T>(get_property_value(property_file, key, default_value));
    }

    ConcreteDaoType get_dao_type() const;

    bool get_set_thread_affinity() const { return set_thread_affinity_; }
    size_t get_multistep_len() const { return multistep_len; }
    size_t get_online_learn_iter_limit() const { return online_learn_iter_limit_; }
    size_t get_stabilize_iterations_count() const { return stabilize_iterations_count_; }
    double get_error_tolerance() const { return error_tolerance_; }
    double get_scaling_alpha() const { return scaling_alpha_; }
    std::string get_online_svr_logfile() const { return online_svr_log_file_;}
    bool get_tune_parameters() const { return tune_parameters_; }
    size_t get_slide_count() const { return slide_count_; }
    size_t get_slide_skip() const { return slide_skip_; }
    size_t get_validation_window() const { return validation_window_; }
    size_t get_tune_run_limit() const { return tune_run_limit_; }

private:
    static const std::string TUNE_PARAMETERS;
    static const std::string SQL_PROPERTIES_DIR_KEY;
    static const std::string LOG_LEVEL_KEY;
    static const std::string COMMENT_CHARS;
    static const std::string DAO_TYPE_KEY;
    static const std::string SET_THREAD_AFFINITY;
    static const std::string MULTISTEP_LEN;
    static const std::string ONLINE_LEARN_ITER_LIMIT;
    static const std::string STABILIZE_ITERATIONS_COUNT;
    static const std::string ERROR_TOLERANCE;
    static const std::string SCALING_ALPHA;
    static const std::string ONLINESVR_LOG_FILE;
    static const std::string SLIDE_COUNT;
    static const std::string TUNE_RUN_LIMIT;
};


} /* namespace common */
} /* namespace svr */

using MessageSource_ptr = std::shared_ptr<svr::common::PropertiesFileReader>;
