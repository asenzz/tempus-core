#pragma once

#include <string>
#include "common.hpp"
#include "business.hpp"
#include "WScalingFactorService.hpp"

#define APP svr::context::AppContext::get_instance()
#define PROPS APP.app_properties

// To pass application properties initialize Service
// classes with values as constructor arguments.

namespace svr {
namespace dao { class scoped_transaction_guard; }
namespace common { class AppConfig; }

namespace context {

class AppContext {
    static AppContext *p_instance;

    AppContext(const std::string &config_path, const bool use_threadsafe_dao);

    ~AppContext();

    AppContext(AppContext &) = delete;

    AppContext(AppContext &&) = delete;

    void operator=(AppContext &) = delete;

    void operator=(AppContext &&) = delete;

    struct AppContextImpl;
    AppContextImpl &p_impl;

public:
    static inline AppContext &get_instance()
    { return *p_instance; }

    static void init_instance(const std::string &config_path, const bool use_threadsafe_dao = true);

    static void destroy_instance();

    common::AppConfig &app_properties;

    business::UserService &user_service;
    business::InputQueueService &input_queue_service;
    business::SVRParametersService &svr_parameters_service;
    business::ModelService &model_service;
    business::DeconQueueService &decon_queue_service;
    business::EnsembleService &ensemble_service;
    business::DatasetService &dataset_service;
    business::RequestService &request_service;
    business::AuthenticationProvider &authentication_provider;
    business::PredictionTaskService &prediction_task_service;
    business::ScalingFactorsTaskService &scaling_factors_task_service;
    business::AutotuneTaskService &autotune_task_service;
    business::DecrementTaskService &decrement_task_service;
    business::IQScalingFactorService &iq_scaling_factor_service;
    business::WScalingFactorService &w_scaling_factor_service;
    business::DQScalingFactorService &dq_scaling_factor_service;

    void flush_dao_buffers();

    bool is_threadsafe_dao() const;
};

struct AppContextDeleter {
    ~AppContextDeleter();
};

}
}
