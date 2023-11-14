#pragma once

#include <cppcms/application.h>
#include <cppcms/applications_pool.h>
#include <cppcms/service.h>
#include <cppcms/http_response.h>
#include <cppcms/url_dispatcher.h>
#include <cppcms/url_mapper.h>
#include <cppcms/rpc_json.h>

#include "model/InputQueue.hpp"
#include "view/TickView.hpp"


namespace svr{
namespace web{

class InputQueueView : public cppcms::rpc::json_rpc_server{

public:
    InputQueueView(cppcms::service &srv);

    virtual void main(std::string) override;
    void getAllInputQueues();
    void showInputQueue(std::string queueTableName);
    void getValueColumnsModel(std::string queueTableName);
    void sendTick(cppcms::json::object obj);
    void getNextTimeRangeToBeSent(cppcms::json::object obj);
    void reconcileHistoricalData(cppcms::json::object obj);
    void historyData(cppcms::json::object  data);
};

class InputQueueController : public cppcms::application{

    void handle_create_get();
    void handle_create_post();

public:

    InputQueueController(cppcms::service &svc);
    void show(std::string queueName);
    void showAll();
    void create();

};
}
}
