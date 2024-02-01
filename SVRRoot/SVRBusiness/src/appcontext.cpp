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




namespace svr {
namespace context {

AppContext *AppContext::p_instance = nullptr;

struct StoreBufferInitializer
{
    StoreBufferInitializer()
    { svr::dao::StoreBufferController::initInstance(); }

    ~StoreBufferInitializer()
    { svr::dao::StoreBufferController::destroyInstance(); }
};

struct AppContext::AppContextImpl : StoreBufferInitializer
{
    svr::common::PropertiesFileReader &appProperties;

    svr::dao::DataSource &dataSource;
    svr::dao::UserDAO &userDao;
    svr::dao::InputQueueDAO &inputQueueDao;
    svr::dao::SVRParametersDAO &svrParametersDao;
    svr::dao::DatasetDAO &datasetDao;
    svr::dao::DeconQueueDAO &deconQueueDao;
    svr::dao::EnsembleDAO &ensembleDao;
    svr::dao::ModelDAO &modelDao;
    svr::dao::RequestDAO &requestDao;
    svr::dao::PredictionTaskDAO &predictionTaskDao;
    svr::dao::ScalingFactorsTaskDAO &scalingFactorsTaskDao;
    svr::dao::AutotuneTaskDAO &autotuneTaskDao;
    svr::dao::DecrementTaskDAO &decrementTaskDao;
    svr::dao::IQScalingFactorDAO &iQScalingFactorDao;
    svr::dao::DQScalingFactorDAO &dQScalingFactorDao;

    bool threadsafe_dao;

    AppContextImpl(const std::string &config_path, bool use_threadsafe_dao)
            : appProperties(*new svr::common::PropertiesFileReader(config_path)), dataSource(
            *new svr::dao::DataSource(appProperties.get_property<std::string>(config_path, "CONNECTION_STRING"), true)),
              userDao(*svr::dao::UserDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                use_threadsafe_dao)), inputQueueDao(
                    *svr::dao::InputQueueDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                    use_threadsafe_dao)), svrParametersDao(
                    *svr::dao::SVRParametersDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                       use_threadsafe_dao)), datasetDao(
                    *svr::dao::DatasetDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                 use_threadsafe_dao)), deconQueueDao(
                    *svr::dao::DeconQueueDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                    use_threadsafe_dao)), ensembleDao(
                    *svr::dao::EnsembleDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                  use_threadsafe_dao)), modelDao(
                    *svr::dao::ModelDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                               use_threadsafe_dao)), requestDao(
                    *svr::dao::RequestDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                 use_threadsafe_dao)), predictionTaskDao(
                    *svr::dao::PredictionTaskDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                        use_threadsafe_dao)), scalingFactorsTaskDao(
                    *svr::dao::ScalingFactorsTaskDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                        use_threadsafe_dao)), autotuneTaskDao(
                    *svr::dao::AutotuneTaskDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                      use_threadsafe_dao)), decrementTaskDao(
                    *svr::dao::DecrementTaskDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                       use_threadsafe_dao)), iQScalingFactorDao(
                    *svr::dao::IQScalingFactorDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                         use_threadsafe_dao)), dQScalingFactorDao(
                    *svr::dao::DQScalingFactorDAO::build(appProperties, dataSource, appProperties.get_dao_type(),
                                                         use_threadsafe_dao)), threadsafe_dao(use_threadsafe_dao)
    {
        svr::common::memory_manager::get();
        //svr::common::ThreadPoolAsio::instance();

        if (appProperties.get_dao_type() == svr::common::ConcreteDaoType::AsyncDao)
            svr::dao::StoreBufferController::getInstance().startPolling();
    }

    ~AppContextImpl()
    {
        if (appProperties.get_dao_type() == svr::common::ConcreteDaoType::AsyncDao)
            svr::dao::StoreBufferController::getInstance().stopPolling();

        delete &dQScalingFactorDao;
        delete &iQScalingFactorDao;
        delete &decrementTaskDao;
        delete &autotuneTaskDao;
        delete &predictionTaskDao;
        delete &scalingFactorsTaskDao;
        delete &requestDao;
        delete &modelDao;
        delete &ensembleDao;
        delete &deconQueueDao;
        delete &datasetDao;
        delete &svrParametersDao;
        delete &inputQueueDao;
        delete &userDao;
        delete &dataSource;
        delete &appProperties;
    }

    void flush_dao_buffers()
    {
        svr::dao::StoreBufferController::getInstance().flush();
    }

    bool is_threadsafe_dao() const
    {
        return threadsafe_dao;
    }
};


AppContext::AppContext(const std::string &config_path, bool use_threadsafe_dao)
        : p_impl(*new AppContextImpl(config_path, use_threadsafe_dao)), app_properties(p_impl.appProperties),
          user_service(
                  *new svr::business::UserService(p_impl.userDao)),
          input_queue_service(
                  *new svr::business::InputQueueService(p_impl.inputQueueDao)),
          svr_parameters_service(
                  *new svr::business::SVRParametersService(p_impl.svrParametersDao)),
          model_service(
                  *new svr::business::ModelService(p_impl.modelDao)),
          decon_queue_service(
                  *new svr::business::DeconQueueService(p_impl.deconQueueDao, input_queue_service)),
          ensemble_service(
                  *new svr::business::EnsembleService(
                          p_impl.ensembleDao, model_service, decon_queue_service)),
          dataset_service(
                  *new svr::business::DatasetService(
                          p_impl.datasetDao, ensemble_service, svr_parameters_service)),
          request_service(*new svr::business::RequestService(p_impl.requestDao)),
          authentication_provider(*new svr::business::LocalAuthenticationProvider(user_service)),
          prediction_task_service(*new svr::business::PredictionTaskService(p_impl.predictionTaskDao)),
          scaling_factors_task_service(*new svr::business::ScalingFactorsTaskService(p_impl.scalingFactorsTaskDao)),
          autotune_task_service(*new svr::business::AutotuneTaskService(p_impl.autotuneTaskDao)),
          decrement_task_service(*new svr::business::DecrementTaskService(p_impl.decrementTaskDao)),
          iq_scaling_factor_service(*new svr::business::IQScalingFactorService(p_impl.iQScalingFactorDao)),
          dq_scaling_factor_service(*new svr::business::DQScalingFactorService(p_impl.dQScalingFactorDao))
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
        throw std::runtime_error("AppContext instance has already been initialized");

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
