#pragma once

#include "model/Dataset.hpp"

namespace svr {
namespace datamodel {

class AutotuneTask : public Entity
{
private:
    bigint dataset_id;
    bigint result_dataset_id;
    bpt::ptime creation_time;
    bpt::ptime done_time;
    std::map<std::string, std::string> parameters;
    bpt::ptime start_train_time;
    bpt::ptime end_train_time;
    bpt::ptime start_validation_time;
    bpt::ptime end_validation_time;

    size_t vp_sliding_direction = 0;
    size_t vp_slide_count = 0;
    bpt::seconds vp_slide_period_sec;

    size_t pso_best_points_counter = 0;
    size_t pso_iteration_number = 0;
    size_t pso_particles_number = 0;
    size_t pso_topology = 0;
    std::string pso_state_file;

    size_t nm_max_iteration_number = 0;
    double nm_tolerance = 0.;

    int status = 0;         // 0 - new, 1 - in process, 2 - done, 3 - error
    double mse = -1.0;

public:
    AutotuneTask() : Entity(), vp_slide_period_sec(bpt::seconds(0))
    {}

    AutotuneTask(bigint id, bigint dataset_id, bigint result_dataset_id,
                 const bpt::ptime &creation_time, const bpt::ptime &done_time,
                 const std::map<std::string, std::string> &parameters,
                 const bpt::ptime &start_train_time, const bpt::ptime &end_train_time,
                 const bpt::ptime &start_validation_time, const bpt::ptime &end_validation_time,
                 const size_t vp_sliding_direction, const size_t vp_slide_count, bpt::seconds vp_slide_period_sec,
                 const size_t pso_best_points_counter, const size_t pso_iteration_number,
                 const size_t pso_particles_number, const size_t pso_topology,
                 const size_t nm_max_iteration_number, double nm_tolerance,
                 const int status = 0, const double mse = -1.)
            : Entity(id),
              dataset_id(dataset_id),
              result_dataset_id(result_dataset_id),
              creation_time(creation_time),
              done_time(done_time),
              parameters(parameters),
              start_train_time(start_train_time),
              end_train_time(end_train_time),
              start_validation_time(start_validation_time),
              end_validation_time(end_validation_time),
              vp_sliding_direction(vp_sliding_direction),
              vp_slide_count(vp_slide_count),
              vp_slide_period_sec(vp_slide_period_sec),
              pso_best_points_counter(pso_best_points_counter),
              pso_iteration_number(pso_iteration_number),
              pso_particles_number(pso_particles_number),
              pso_topology(pso_topology),
              nm_max_iteration_number(nm_max_iteration_number),
              nm_tolerance(nm_tolerance),
              status(status),
              mse(mse)
    {}

    AutotuneTask(bigint id, bigint dataset_id, bigint result_dataset_id,
                 const bpt::ptime &creation_time, const bpt::ptime &done_time,
                 const std::string &parameters,
                 const bpt::ptime &start_train_time, const bpt::ptime &end_train_time,
                 const bpt::ptime &start_validation_time, const bpt::ptime &end_validation_time,
                 const size_t vp_sliding_direction, const size_t vp_slide_count, bpt::seconds vp_slide_period_sec,
                 const size_t pso_best_points_counter, const size_t pso_iteration_number,
                 const size_t pso_particles_number, const size_t pso_topology,
                 const size_t nm_max_iteration_number, double nm_tolerance,
                 const int status = 0, const double mse = -1.0, std::string pso_state_file = "")
            : Entity(id),
              dataset_id(dataset_id),
              result_dataset_id(result_dataset_id),
              creation_time(creation_time),
              done_time(done_time),
              start_train_time(start_train_time),
              end_train_time(end_train_time),
              start_validation_time(start_validation_time),
              end_validation_time(end_validation_time),
              vp_sliding_direction(vp_sliding_direction),
              vp_slide_count(vp_slide_count),
              vp_slide_period_sec(vp_slide_period_sec),
              pso_best_points_counter(pso_best_points_counter),
              pso_iteration_number(pso_iteration_number),
              pso_particles_number(pso_particles_number),
              pso_topology(pso_topology),
              nm_max_iteration_number(nm_max_iteration_number),
              nm_tolerance(nm_tolerance),
              status(status), mse(mse)
    {
        set_parameters_from_string(parameters);
    }

    bool operator==(const AutotuneTask &other) const
    {
        return other.get_id() == get_id()
               && other.get_dataset_id() == get_dataset_id()
               && other.get_result_dataset_id() == get_result_dataset_id()
               && other.get_creation_time() == get_creation_time()
               && other.get_done_time() == get_done_time()
               && other.get_parameters() == get_parameters()
               && other.get_start_train_time() == get_start_train_time()
               && other.get_end_train_time() == get_end_train_time()
               && other.get_start_validation_time() == get_start_validation_time()
               && other.get_end_validation_time() == get_end_validation_time()
               && other.get_status() == get_status()
               && other.get_mse() == get_mse();
    }

    bigint get_dataset_id() const
    {
        return dataset_id;
    }

    void set_dataset_id(bigint value)
    {
        dataset_id = value;
    }

    bigint get_result_dataset_id() const
    {
        return result_dataset_id;
    }

    void set_result_dataset_id(bigint value)
    {
        result_dataset_id = value;
    }

    bpt::ptime get_creation_time() const
    {
        return creation_time;
    }

    void set_creation_time(const bpt::ptime &value)
    {
        creation_time = value;
    }

    bpt::ptime get_done_time() const
    {
        return done_time;
    }

    void set_done_time(const bpt::ptime &value)
    {
        done_time = value;
    }

    std::map<std::string, std::string> get_parameters() const
    {
        return parameters;
    }

    std::string get_parameters_in_string() const
    {
        return svr::common::map_to_json(parameters);
    }

//    std::vector<uint> get_tweaking_levels() const {

//        std::vector<uint> tweaking_levels;
//        uint level = 0;
//        std::string suffix;
//        while (parameters.count("svr_c" + (suffix = "_" + std::to_string(level)))) {
//            std::string suffix = "_" + std::to_string(level);
//            if (parameters.at("svr_c" + suffix) == "tune"
//                || parameters.at("svr_epsilon" + suffix) == "tune"
//                || parameters.at("svr_kernel_param" + suffix) == "tune"
//                || parameters.at("svr_kernel_param2" + suffix)  == "tune"
//                || parameters.at("svr_adjacent_levels_ratio" + suffix) == "tune"
//                || parameters.at("svr_error_tolerance" + suffix) == "tune"

//                tweaking_levels.push_back(level);
//            }
//            ++level;
//        }
//        return tweaking_levels;
//    }

    void set_parameters(const std::map<std::string, std::string> &value)
    {
        LOG4_DEBUG("Setting parameters " << svr::common::map_to_json(value));
        // checking for correct json parameters string and non tune keys checking
        std::vector<std::string> check_keys = {"transformation_levels", "transformation_name"};
        for (const std::string &key : check_keys) {
            if (value.count(key) == 0) {
                throw std::runtime_error("Key " + key + " doesn't exist in params!!!");
            }
        }
        check_keys = {"svr_c", "svr_epsilon", "svr_kernel_param", "svr_kernel_param2",
                      "svr_decremental_distance", "svr_adjacent_levels_ratio", "svr_kernel_type",
                      "lag_count"};
        std::vector<std::string> non_tune_keys = {"svr_decremental_distance"};
        // find get all levels which contain key_levelnumber
        std::vector<size_t> transformation_levels;
        bool find_next = true;
        for (size_t level = 0; find_next; ++level) {
            std::string suffix = "_" + std::to_string(level);
            find_next = false;
            for (std::string key : check_keys) {
                if (value.count(key + suffix)) {
                    transformation_levels.push_back(level);
                    find_next = true;
                    break;
                }
            }
        }
        // for levels which was found check all keys
        for (const auto level: transformation_levels) {
            for (const std::string &key: check_keys)
                if (value.count(key + "_" + std::to_string(level)) == 0)
                    throw std::runtime_error(
                            "Key " + key + "_" + std::to_string(level) + " doesn't exist in params!");
            for (const std::string &key: non_tune_keys)
                if (value.at(key + "_" + std::to_string(level)) == "tune")
                    throw std::runtime_error("Key " + key + "_" + std::to_string(level) + " can't be tuned!");

        }
        parameters = value;
    }

    void set_parameters_from_string(const std::string &json_str)
    {
        set_parameters(svr::common::json_to_map(json_str));
    }

    bpt::ptime get_start_train_time() const
    {
        return start_train_time;
    }

    void set_start_train_time(const bpt::ptime &value)
    {
        start_train_time = value;
    }

    bpt::ptime get_end_train_time() const
    {
        return end_train_time;
    }

    void set_end_train_time(const bpt::ptime &value)
    {
        end_train_time = value;
    }

    bpt::ptime get_start_validation_time() const
    {
        return start_validation_time;
    }

    void set_start_validation_time(const bpt::ptime &value)
    {
        start_validation_time = value;
    }

    bpt::ptime get_end_validation_time() const
    {
        return end_validation_time;
    }

    void set_end_validation_time(const bpt::ptime &value)
    {
        end_validation_time = value;
    }


    size_t get_vp_slide_count() const
    {
        return vp_slide_count;
    }

    void set_vp_slide_count(size_t value)
    {
        vp_slide_count = value;
    }

    bpt::seconds get_vp_slide_period_sec() const
    {
        return vp_slide_period_sec;
    }

    void set_vp_slide_period_sec(const bpt::seconds &value)
    {
        vp_slide_period_sec = value;
    }

    size_t get_pso_best_points_counter() const
    {
        return pso_best_points_counter;
    }

    void set_pso_best_points_counter(size_t value)
    {
        pso_best_points_counter = value;
    }

    size_t get_pso_iteration_number() const
    {
        return pso_iteration_number;
    }

    void set_pso_iteration_number(size_t value)
    {
        pso_iteration_number = value;
    }

    std::string get_pso_state_file() const
    {
        return pso_state_file;
    }

    void set_pso_state_file(std::string filename)
    {
        pso_state_file = filename;
    }

    size_t get_pso_particles_number() const
    {
        return pso_particles_number;
    }

    void set_pso_particles_number(size_t value)
    {
        pso_particles_number = value;
    }

    size_t get_pso_topology() const
    {
        return pso_topology;
    }

    void set_pso_topology(size_t value)
    {
        pso_topology = value;
    }

    size_t get_nm_max_iteration_number() const
    {
        return nm_max_iteration_number;
    }

    void set_nm_max_iteration_number(size_t value)
    {
        nm_max_iteration_number = value;
    }

    double get_nm_tolerance() const
    {
        return nm_tolerance;
    }

    void set_nm_tolerance(double value)
    {
        nm_tolerance = value;
    }

    size_t get_vp_sliding_direction() const
    {
        return vp_sliding_direction;
    }

    void set_vp_sliding_direction(size_t value)
    {
        vp_sliding_direction = value;
    }

    int get_status() const
    {
        return status;
    }

    void set_status(int value)
    {
        status = value;
    }

    double get_mse() const
    {
        return mse;
    }

    void set_mse(double mse)
    {
        this->mse = mse;
    }

    virtual std::string to_string() const override
    {
        std::stringstream ss;

        ss << "Autotune id: " << get_id()
           << ", dataset id: " << get_dataset_id()
           << ", result dataset id: " << get_result_dataset_id()
           << ", creation time: " << get_creation_time()
           << ", done time: " << get_done_time()
           << ", parameters:" << get_parameters_in_string()
           << ", status: " << get_status()
           << ", mse: " << get_mse();

        return ss.str();
    }
};


} /* namespace datamodel */
} /* namespace svr */

using AutotuneTask_ptr = std::shared_ptr<svr::datamodel::AutotuneTask>;
