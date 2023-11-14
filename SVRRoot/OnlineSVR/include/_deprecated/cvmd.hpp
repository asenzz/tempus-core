//
// Created by zarko on 19.12.20 ?..
//

#ifndef SVR_CVMD_HPP
#define SVR_CVMD_HPP

#include "spectral_transform.hpp"
#include <mutex>
#include <unordered_map>

// TODO Make config options
#define CVMD_INPUT_MULTIPLIER 10. // Leave at 1000 for best precision in EURUSD or 10 for NZDJPY

// Best precision TOL
#define EPSI_TOL_VMD 1e-8
#define FAST_CVMD
//#define DEBUG_CVMD

#define TAU_FIDELITY 0

#define ALPHA_BINS 2000

#include <armadillo>
#include <boost/date_time/posix_time/ptime.hpp>

namespace svr {

struct cvmd_frequency_outputs;

typedef std::tuple<
        std::string /* decon queue table_name */,
        size_t /* levels */> freq_key_t;

class cvmd : public spectral_transform {
    size_t levels = 0;
    std::map<freq_key_t, cvmd_frequency_outputs> vmd_frequencies;

public:
    explicit cvmd(const size_t levels);

    virtual ~cvmd() final;

    virtual void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const size_t padding /* = 0 */) override;

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const std::string &table_name,
            const std::vector<double> &prev_decon,
            std::vector<std::vector<double>> &phase_series,
            const std::vector<double> &start_phase);

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const std::string &table_name,
            const size_t count_from_start);

    virtual void inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding /* = 0 */) const override;

    size_t get_residuals_length(const std::string &decon_queue_table_name);

    std::vector<double> calculate_phases(const size_t count_from_start, const std::string &decon_queue_table_name) const;

    bool initialized(const std::string &decon_queue_table_name);
    void initialize(const std::vector<double> &input, const std::string &decon_queue_table_name, const bool load_solvers = false);
    void uninitialize(const bool save);
    void load_solvers_file();
};

}
#endif //SVR_CVMD_HPP
