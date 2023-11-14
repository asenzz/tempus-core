#pragma once

#include "model/Dataset.hpp"

namespace svr{
namespace datamodel{

class PredictionTask : public Entity{

private:
    bigint      dataset_id;
    bpt::ptime  start_train_time;
    bpt::ptime  end_train_time;
    bpt::ptime  start_prediction_time;
    bpt::ptime  end_prediction_time;
    int         status                  = 0; // 0 - new, 1 - in process, 2 - done, 3 - error
    double      mse                     = -1.0;

public:

    PredictionTask() : Entity() {}

    PredictionTask(bigint id, bigint dataset_id,
                   const bpt::ptime &start_train_time, const bpt::ptime &end_train_time,
                   const bpt::ptime &start_prediction_time, const bpt::ptime &end_prediction_time,
                   int status = 0, double mse = -1.0)
            : Entity(id),
              dataset_id(dataset_id),
              start_train_time(start_train_time),
              end_train_time(end_train_time),
              start_prediction_time(start_prediction_time),
              end_prediction_time(end_prediction_time),
              status(status),
              mse(mse) {
    }

    bool operator == (const PredictionTask& other) const {
        return other.get_id()                       == get_id()
                && other.get_dataset_id()           == get_dataset_id()
                && other.get_start_train_time()     == get_start_train_time()
                && other.get_end_train_time()       == get_end_train_time()
                && other.get_start_prediction_time() == get_start_prediction_time()
                && other.get_end_prediction_time()  == get_end_prediction_time()
                && other.get_status()               == get_status()
                && other.get_mse()                  == get_mse();
    }

    bigint get_dataset_id() const
    {
        return dataset_id;
    }

    void set_dataset_id(const bigint dataset_id)
    {
        this->dataset_id = dataset_id;
    }

    bpt::ptime get_start_train_time() const {
        return this->start_train_time;
    }

    void set_start_train_time(const bpt::ptime &start_train_time) {
        this->start_train_time = start_train_time;
    }

    bpt::ptime get_end_train_time() const {
        return this->end_train_time;
    }

    void set_end_train_time(const bpt::ptime &end_train_time) {
        this->end_train_time = end_train_time;
    }

    bpt::ptime get_start_prediction_time() const {
        return start_prediction_time;
    }

    void set_start_prediction_time(const bpt::ptime &start_prediction_time) {
        this->start_prediction_time = start_prediction_time;
    }

    bpt::ptime get_end_prediction_time() const {
        return this->end_prediction_time;
    }

    void set_end_prediction_time(const bpt::ptime &end_prediction_time) {
        this->end_prediction_time = end_prediction_time;
    }

    int get_status() const {
        return status;
    }

    void set_status(int status) {
        this->status = status;
    }

    double get_mse() const {
        return mse;
    }

    void set_mse(double value) {
        this->mse = value;
    }

    virtual std::string to_string() const override {
        std::stringstream ss;

        ss << "Prediction task id: "        << get_id()
            << ", Dataset id: "             << get_dataset_id()
            << ", Start train time: "       << bpt::to_simple_string(get_start_train_time())
            << ", End train time: "         << bpt::to_simple_string(get_end_train_time())
            << ", Start prediction time: "  << bpt::to_simple_string(get_start_prediction_time())
            << ", End prediction time: "    << bpt::to_simple_string(get_end_prediction_time())
            << ", Status: "                 << get_status()
            << ", MSE: "                    << get_mse();

        return ss.str();
    }
};

} /* namespace datamodel */
} /* namespace svr */

using PredictionTask_ptr = std::shared_ptr<svr::datamodel::PredictionTask>;
