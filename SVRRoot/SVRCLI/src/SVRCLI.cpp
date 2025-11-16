//
// Created by zarko on 8/13/25.
//

#include <boost/program_options.hpp>
#include <boost/date_time.hpp>
#include <csignal>
#include "appcontext.hpp"
#include "common/logging.hpp"
#include "onlinesvr.hpp"
#include "model/Priority.hpp"
#include "model/Dataset.hpp"

using namespace svr;

void signal_handler(const int signum)
{
    LOG4_DEBUG("Interrupt signal " << signum << " received.");
    if (signum == SIGINT || signum == SIGABRT || signum == SIGTERM) {
        LOG4_INFO("Daemon process is shutting down gracefully.");
        exit(signum);
    }
    LOG4_WARN("Received unexpected signal " << signum);
}

std::tuple<std::string, std::string> parse(const int argc, const char **argv)
{
    auto gen_desc = boost::program_options::options_description("Daemon options");
    gen_desc.add_options()
            ("help", "produce help message")
            ("config,c", boost::program_options::value<std::string>()->default_value("daemon.config"), "Path to file with configuration for CLI application.")
            ("train,t", boost::program_options::value<std::string>()->default_value("train.csv"), "Path to file with training data.")
            ("validation,v", boost::program_options::value<std::string>()->default_value("validate.csv"), "Path to file with validate data.");

    boost::program_options::variables_map vm;
    // parse command line
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(gen_desc).run(), vm);
    if (vm.count("help") or !vm.count("config") or !vm.count("input")) {
        std::cout << gen_desc << std::endl;
        exit(0);
    }
    const auto config_str = vm["config"].as<std::string>();
    if (config_str.empty()) THROW_EX_FS(std::invalid_argument, "Empty path to config file.");

    context::AppContext::init_instance(config_str);

    const auto training_data_str = vm["data"].as<std::string>();
    if (training_data_str.empty()) THROW_EX_FS(std::invalid_argument, "Empty path to data file.");

    const auto validation_data_str = vm["data"].as<std::string>();
    if (validation_data_str.empty()) THROW_EX_FS(std::invalid_argument, "Empty path to data file.");

    return {training_data_str, validation_data_str};
}

void start_cli(const std::string &training_data_file_path, const std::string &validation_data_file_path)
{
    arma::mat training_data;
    training_data.load(training_data_file_path, arma::csv_ascii);
    const auto n_label_cols = PROPS.get_outputs();
    const auto w = ptr<arma::mat>(training_data.col(0));
    const auto y = ptr<arma::mat>(training_data.cols(1, n_label_cols));
    const auto x = ptr<arma::mat>(training_data.cols(n_label_cols + 1, training_data.n_cols - 1));
    auto parameters = ptr<datamodel::SVRParameters>();
    constexpr uint32_t C_dataset_id = 0xDeadBeef;
    const std::string C_test_input_name = "q_svrcli_";
#define MAIN_QUEUE_RES 3600
#define STR_MAIN_QUEUE_RES TOSTR(MAIN_QUEUE_RES)
    const std::string C_test_input_table_name(C_test_input_name + STR_MAIN_QUEUE_RES);
    const std::string C_test_aux_input_table_name(C_test_input_name + "1");
    constexpr uint16_t C_test_levels = 1;
    constexpr uint16_t C_test_gradient_count = 1;
    auto p_dataset = ptr<datamodel::Dataset>(
            C_dataset_id, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "", common::C_default_residual_coef,
            C_test_gradient_count, PROPS.get_kernel_length(), PROPS.get_steps(), C_test_levels, "cvmd", common::C_default_features_max_time_gap);
    datamodel::OnlineSVR model(0, 0, {parameters}, x, y, w, bpt::second_clock::local_time(),  {}, p_dataset);

    arma::mat validation_data;
    validation_data.load(validation_data_file_path, arma::csv_ascii);
    const arma::mat w_valid = validation_data.col(0);
    const arma::mat y_valid = validation_data.cols(1, n_label_cols);
    const arma::mat x_valid = validation_data.cols(n_label_cols + 1, validation_data.n_cols - 1);
    const auto y_predict = model.predict(x_valid);
    LOG4_INFO("Residuals " << common::present<double>(y_predict - y_valid) << ", validation " << common::present(y_valid) << ", predicted " << common::present(y_predict));
}

int main(const int argc, const char **argv)
{
#ifdef INTEGRATION_TEST
    LOG4_FATAL("Integration test build cannot be run in production.");
    exit(0xfd);
#else
    (void) signal(SIGINT, signal_handler);
    (void) signal(SIGABRT, signal_handler);
    (void) signal(SIGTERM, signal_handler);

    omp_set_nested(true);
    if (common::gpu_handler_1::get().get_gpu_devices_count()) omp_set_default_device(0);

    std::shared_ptr<daemon::DaemonFacade> p_daemon_facade;
    int rc = 0;
    try {
        std::apply(start_cli, parse(argc, argv));
    } catch (const std::invalid_argument &e) {
        LOG4_ERROR(e.what());
        rc = 1;
    } catch (const std::exception &e) {
        LOG4_ERROR(e.what());
        rc = 0xff;
    } catch (...) {
        LOG4_ERROR("Unknown exception thrown. ");
        rc = 0xfe;
    }
    LOG4_INFO("CLI process finishing");

    return rc;
#endif
}
