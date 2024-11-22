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
namespace common { class PropertiesFileReader; }

namespace context {

class AppContext{
private:
    struct AppContextImpl;
    AppContextImpl &p_impl;

public:
    static inline AppContext& get_instance() { return *p_instance; }
    static void init_instance(const std::string &config_path, bool use_threadsafe_dao = true);
    static void destroy_instance();

    svr::common::PropertiesFileReader &app_properties;

    svr::business::UserService &user_service;
    svr::business::InputQueueService &input_queue_service;
public:
    svr::business::SVRParametersService &svr_parameters_service;
    svr::business::ModelService &model_service;
    svr::business::DeconQueueService &decon_queue_service;
    svr::business::EnsembleService &ensemble_service;
    svr::business::DatasetService &dataset_service;
    svr::business::RequestService &request_service;
    svr::business::AuthenticationProvider &authentication_provider;
    svr::business::PredictionTaskService &prediction_task_service;
    svr::business::ScalingFactorsTaskService &scaling_factors_task_service;
    svr::business::AutotuneTaskService &autotune_task_service;
    svr::business::DecrementTaskService &decrement_task_service;
    svr::business::IQScalingFactorService &iq_scaling_factor_service;
    svr::business::WScalingFactorService &w_scaling_factor_service;
    svr::business::DQScalingFactorService &dq_scaling_factor_service;

    void flush_dao_buffers();

    bool is_threadsafe_dao() const;

private:
    static AppContext *p_instance;
    AppContext(const std::string& config_path, const bool use_threadsafe_dao);
    ~AppContext();
    AppContext(AppContext&)=delete;
    AppContext(AppContext&&)=delete;
    void operator=(AppContext&)=delete;
    void operator=(AppContext&&)=delete;
};

struct AppContextDeleter
{
    ~AppContextDeleter();
};


}
}
