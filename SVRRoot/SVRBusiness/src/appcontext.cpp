#include "appcontext.hpp"

#include "DAO/UserDAO.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/SVRParametersDAO.hpp"
#include "DAO/InputQueueDAO.hpp"
#include "DAO/DatasetDAO.hpp"
#include "DAO/DeconQueueDAO.hpp"
#include "DAO/EnsembleDAO.hpp"
#include "DAO/ModelDAO.hpp"
#include "DAO/RequestDAO.hpp"
#include "DAO/PredictionTaskDAO.hpp"
#include "DAO/ScalingFactorsTaskDAO.hpp"
#include "DAO/AutotuneTaskDAO.hpp"
#include "DAO/DecrementTaskDAO.hpp"
#include "DAO/IQScalingFactorDAO.hpp"
#include "DAO/DQScalingFactorDAO.hpp"


#include "../../SVRPersist/src/AsyncDAO/StoreBufferController.hpp"
#include "DAO/WScalingFactorDAO.hpp"


namespace svr {
namespace context {

AppContext *AppContext::p_instance = nullptr;

struct StoreBufferInitializer {
    StoreBufferInitializer()
    { dao::StoreBufferController::initInstance(); }

    ~StoreBufferInitializer()
    { dao::StoreBufferController::destroyInstance(); }
};

struct AppContext::AppContextImpl : StoreBufferInitializer {
    common::AppConfig &app_properties;

    dao::DataSource &data_source;
    dao::UserDAO &user_dao;
    dao::InputQueueDAO &input_queue_dao;
    dao::SVRParametersDAO &svr_parameters_dao;
    dao::DatasetDAO &dataset_dao;
    dao::DeconQueueDAO &decon_queue_dao;
    dao::EnsembleDAO &ensemble_dao;
    dao::ModelDAO &model_dao;
    dao::RequestDAO &request_dao;
    dao::PredictionTaskDAO &prediction_task_dao;
    dao::ScalingFactorsTaskDAO &scaling_factors_task_dao;
    dao::AutotuneTaskDAO &autotune_task_dao;
    dao::DecrementTaskDAO &decrement_task_dao;
    dao::IQScalingFactorDAO &iq_scaling_factor_dao;
    dao::WScalingFactorDAO &w_scaling_factor_dao;
    dao::DQScalingFactorDAO &dq_scaling_factor_dao;

    bool threadsafe_dao;

    AppContextImpl(const std::string &config_path, const bool use_threadsafe_dao)
            : app_properties(*new common::AppConfig(config_path)),
              data_source(*new dao::DataSource(app_properties.get_db_connection_string(), true)),
              user_dao(*dao::UserDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              input_queue_dao(*dao::InputQueueDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              svr_parameters_dao(*dao::SVRParametersDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              dataset_dao(*dao::DatasetDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              decon_queue_dao(*dao::DeconQueueDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              ensemble_dao(*dao::EnsembleDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              model_dao(*dao::ModelDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              request_dao(*dao::RequestDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              prediction_task_dao(*dao::PredictionTaskDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              scaling_factors_task_dao(*dao::ScalingFactorsTaskDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              autotune_task_dao(*dao::AutotuneTaskDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              decrement_task_dao(*dao::DecrementTaskDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              iq_scaling_factor_dao(*dao::IQScalingFactorDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              w_scaling_factor_dao(*dao::WScalingFactorDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              dq_scaling_factor_dao(*dao::DQScalingFactorDAO::build(app_properties, data_source, app_properties.get_dao_type(), use_threadsafe_dao)),
              threadsafe_dao(use_threadsafe_dao)
    {
        common::memory_manager::get();
        // common::ThreadPoolAsio::instance();

        if (app_properties.get_dao_type() == common::ConcreteDaoType::AsyncDao)
            dao::StoreBufferController::get_instance().start_polling();
    }

    ~AppContextImpl()
    {
        if (app_properties.get_dao_type() == common::ConcreteDaoType::AsyncDao)
            dao::StoreBufferController::get_instance().stop_polling();

        delete &dq_scaling_factor_dao;
        delete &iq_scaling_factor_dao;
        delete &w_scaling_factor_dao;
        delete &decrement_task_dao;
        delete &autotune_task_dao;
        delete &prediction_task_dao;
        delete &scaling_factors_task_dao;
        delete &request_dao;
        delete &model_dao;
        delete &ensemble_dao;
        delete &decon_queue_dao;
        delete &dataset_dao;
        delete &svr_parameters_dao;
        delete &input_queue_dao;
        delete &user_dao;
        delete &data_source;
        delete &app_properties;
    }

    void flush_dao_buffers()
    {
        dao::StoreBufferController::get_instance().flush();
    }

    bool is_threadsafe_dao() const
    {
        return threadsafe_dao;
    }
};


AppContext::AppContext(const std::string &config_path, const bool use_threadsafe_dao)
        : p_impl(*new AppContextImpl(config_path, use_threadsafe_dao)), app_properties(p_impl.app_properties),
          user_service(*new business::UserService(p_impl.user_dao)),
          input_queue_service(*new business::InputQueueService(p_impl.input_queue_dao)),
          svr_parameters_service(*new business::SVRParametersService(p_impl.svr_parameters_dao)),
          model_service(*new business::ModelService(p_impl.model_dao)),
          decon_queue_service(*new business::DeconQueueService(p_impl.decon_queue_dao)),
          ensemble_service(*new business::EnsembleService(p_impl.ensemble_dao, model_service, decon_queue_service)),
          dataset_service(*new business::DatasetService(p_impl.dataset_dao, ensemble_service, svr_parameters_service)),
          request_service(*new business::RequestService(p_impl.request_dao)),
          authentication_provider(*new business::LocalAuthenticationProvider(user_service)),
          prediction_task_service(*new business::PredictionTaskService(p_impl.prediction_task_dao)),
          scaling_factors_task_service(*new business::ScalingFactorsTaskService(p_impl.scaling_factors_task_dao)),
          autotune_task_service(*new business::AutotuneTaskService(p_impl.autotune_task_dao)),
          decrement_task_service(*new business::DecrementTaskService(p_impl.decrement_task_dao)),
          iq_scaling_factor_service(*new business::IQScalingFactorService(p_impl.iq_scaling_factor_dao)),
          w_scaling_factor_service(*new business::WScalingFactorService(p_impl.w_scaling_factor_dao)),
          dq_scaling_factor_service(*new business::DQScalingFactorService(p_impl.dq_scaling_factor_dao))
{}

AppContext::~AppContext()
{
    delete &dq_scaling_factor_service;
    delete &iq_scaling_factor_service;
    delete &decrement_task_service;
    delete &autotune_task_service;
    delete &prediction_task_service;
    delete &scaling_factors_task_service;
    delete &authentication_provider;
    delete &request_service;
    delete &model_service;
    delete &ensemble_service;
    delete &decon_queue_service;
    delete &dataset_service;
    delete &svr_parameters_service;
    delete &input_queue_service;
    delete &user_service;
    delete &p_impl;
}

void AppContext::init_instance(const std::string &config_path, bool use_threadsafe_dao)
{
    if (AppContext::p_instance)
        LOG4_THROW("AppContext instance has already been initialized");

    AppContext::p_instance = new AppContext(config_path, use_threadsafe_dao);
}

void AppContext::destroy_instance()
{
    delete AppContext::p_instance;
    AppContext::p_instance = nullptr;
}

void AppContext::flush_dao_buffers()
{
    p_impl.flush_dao_buffers();
}


bool AppContext::is_threadsafe_dao() const
{
    return p_impl.is_threadsafe_dao();
}


AppContextDeleter::~AppContextDeleter()
{
    AppContext::destroy_instance();
}
}
}
