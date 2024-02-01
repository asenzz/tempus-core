#pragma once

#include <model/DataRow.hpp>
#include <model/InputQueue.hpp>
#include <DAO/InputQueueDAO.hpp>

class InputQueueRowDataGenerator{

    svr::datamodel::InputQueue_ptr _queue;
    svr::business::InputQueueService & _input_queue_service;
    int _numberOfValueColumns;
    long _rowsToGenerate;
    long _rowsGeneratedByFar;
    bpt::ptime _startTime;
    bpt::ptime _currentValueTime;
public:

    InputQueueRowDataGenerator(svr::business::InputQueueService & input_queue_service, svr::datamodel::InputQueue_ptr queue, int numberOfValueColumns, long rowsToGenerate) :
            _queue(queue),
            _input_queue_service(input_queue_service),
            _numberOfValueColumns(numberOfValueColumns),
            _rowsToGenerate(rowsToGenerate),
            _rowsGeneratedByFar(0),
            _startTime(bpt::second_clock::local_time()),
            _currentValueTime(_startTime){}

    svr::datamodel::DataRow_ptr operator()(){

        if(isDone()){
            return svr::datamodel::DataRow_ptr(nullptr);
        }

        _currentValueTime += _queue->get_resolution();

        while(_input_queue_service.adjust_time_on_grid(_queue, _currentValueTime) == bpt::not_a_date_time){
            _currentValueTime += bpt::seconds(1);
        }

        std::vector<double> values(_numberOfValueColumns);

        for(int i = 0; i < _numberOfValueColumns; i++){
            values[i] = rand() % 10000 / 7.7;
        }

        ++_rowsGeneratedByFar;

        return std::make_shared<svr::datamodel::DataRow>(
                _currentValueTime,
                bpt::second_clock::local_time(),
                1.2,
                values);
    }

    bool isDone(){
        return _rowsGeneratedByFar >= _rowsToGenerate;
    }

};

