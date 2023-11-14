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
    bool dont_update_r_matrix_;
    bool main_columns_aux_;
    size_t max_smo_iterations_;
    double cascade_reduce_ratio_;
    size_t cascade_max_segment_size_;
    size_t cascade_branches_count_;
    bool disable_cascaded_svm_;
    size_t multistep_len;
    std::string svr_paramtune_column;
    std::string svr_paramtune_level;
    size_t online_iters_limit_mult_;
    size_t online_learn_iter_limit_;
    double smo_epsilon_divisor_;
    double smo_cost_divisor_;
    size_t stabilize_iterations_count_;
    size_t default_number_variations_;
    double error_tolerance_;
    double scaling_alpha_;
    std::string online_svr_log_file_;
    size_t max_variations_;
    size_t comb_train_count_;
    size_t comb_validate_count_;
    size_t comb_validate_limit_;
    bool enable_comb_validate_;
    bool tune_parameters_;
    size_t future_predict_count_;
    size_t slide_count_;
    size_t tune_run_limit_;
    bool all_aux_levels_;
    bool oemd_find_fir_coefficients_;

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
    bool get_dont_update_r_matrix() const { return dont_update_r_matrix_; }
    bool get_main_columns_aux() const { return main_columns_aux_; }
    size_t get_max_smo_iterations() const { return max_smo_iterations_; }
    double get_cascade_reduce_ratio() const { return cascade_reduce_ratio_; }
    size_t get_max_segment_size() const { return cascade_max_segment_size_; }
    bool get_disable_cascaded_svm() const { return disable_cascaded_svm_; }
    size_t get_cascade_branches_count() const { return cascade_branches_count_; }

    size_t get_multistep_len() const { return multistep_len; }
    std::string get_svr_paramtune_level() const {return svr_paramtune_level;}
    std::vector<size_t> get_svr_paramtune_levels() const;
    std::string get_svr_paramtune_column() const {return svr_paramtune_column;}
    size_t get_online_iters_limit_mult() const { return online_iters_limit_mult_; }
    size_t get_online_learn_iter_limit() const { return online_learn_iter_limit_; }
    double get_smo_epsilon_divisor() const { return smo_epsilon_divisor_; }
    double get_smo_cost_divisor() const { return smo_cost_divisor_; }
    size_t get_stabilize_iterations_count() const { return stabilize_iterations_count_; }
    size_t get_default_number_variations() const { return default_number_variations_; }
    double get_error_tolerance() const { return error_tolerance_; }
    double get_scaling_alpha() const { return scaling_alpha_; }
    std::string get_online_svr_logfile() const { return online_svr_log_file_;}
    size_t get_max_variations() const { return max_variations_;}
    bool get_autotune_running() const { return is_autotune_running_; } // TODO Remove
    size_t get_comb_train_count() const { return comb_train_count_; }
    size_t get_comb_validate_count() const { return comb_validate_count_; }
    size_t get_comb_validate_limit() const { return comb_validate_limit_; }
    bool get_enable_comb_validate() const { return enable_comb_validate_; }
    bool get_tune_parameters() const { return tune_parameters_; }
    size_t get_future_predict_count() const { return future_predict_count_; }
    size_t get_slide_count() const { return slide_count_; }
    size_t get_tune_run_limit() const { return tune_run_limit_; }
    bool get_all_aux_levels() const { return all_aux_levels_; }
    bool get_oemd_find_fir_coefficients() const { return oemd_find_fir_coefficients_; }

    // TODO Befriend with CLI class and mark private
    void set_autotune_running(const bool running) { is_autotune_running_ = running; }
private:
    static const std::string SQL_PROPERTIES_DIR_KEY;
    static const std::string LOG_LEVEL_KEY;
    static const std::string COMMENT_CHARS;
    static const std::string DAO_TYPE_KEY;
    static const std::string DONT_UPDATE_R_MATRIX;
    static const std::string MAIN_COLUMNS_AUX;
    static const std::string MAX_SMO_ITERATIONS;
    static const std::string CASCADE_REDUCE_RATIO;
    static const std::string CASCADE_MAX_SEGMENT_SIZE;
    static const std::string DISABLE_CASCADED_SVM;
    static const std::string CASCADE_BRANCHES_COUNT;
    static const std::string SET_THREAD_AFFINITY;
    static const std::string MULTISTEP_LEN;
    static const std::string SVR_PARAMTUNE_COLUMN;
    static const std::string SVR_PARAMTUNE_LEVEL;
    static const std::string ONLINE_LEARN_ITER_LIMIT;
    static const std::string ONLINE_ITERS_LIMIT_MULT;
    static const std::string SMO_EPSILON_DIVISOR;
    static const std::string SMO_COST_DIVISOR;
    static const std::string STABILIZE_ITERATIONS_COUNT;
    static const std::string DEFAULT_NUMBER_VARIATIONS;
    static const std::string ERROR_TOLERANCE;
    static const std::string SCALING_ALPHA;
    static const std::string ONLINESVR_LOG_FILE;
    static const std::string MAX_VARIATIONS;
    static const std::string COMB_TRAIN_COUNT;
    static const std::string COMB_VALIDATE_COUNT;
    static const std::string COMB_VALIDATE_LIMIT;
    static const std::string ENABLE_COMB_VALIDATE;
    static const std::string ALL_AUX_LEVELS;
    static const std::string TUNE_PARAMETERS;
    static const std::string FUTURE_PREDICT_COUNT;
    static const std::string SLIDE_COUNT;
    static const std::string TUNE_RUN_LIMIT;
    static const std::string OEMD_FIND_FIR_COEFFICIENTS;
};


} /* namespace common */
} /* namespace svr */

using MessageSource_ptr = std::shared_ptr<svr::common::PropertiesFileReader>;
