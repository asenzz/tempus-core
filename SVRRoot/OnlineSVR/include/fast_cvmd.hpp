//
// Created by zarko on 7/5/22.
//

#ifndef SVR_FAST_CVMD_HPP
#define SVR_FAST_CVMD_HPP

#include <vector>
#include <armadillo>

#include "spectral_transform.hpp"

#define OMEGA_DIVISOR 1. // 1e2 // Divides frequency by N, 1. for XAUUSD, 14 for EURUSD
#define TAU_FIDELITY 0
#define ALPHA_BINS 100 // 2000 for EURUSD, 1600 for XAUUSD
#define MAX_VMD_ITERATIONS 500
#define DEFAULT_PHASE_STEP 1
#define EPS std::numeric_limits<double>::epsilon()
#define CVMD_TOL EPS
#define EMOS_OMEGAS
#define FASTER_CVMD
//#define ORTHO_CVMD

namespace svr {

typedef std::tuple<std::string /* decon queue table_name */, size_t /* levels */> freq_key_t;
struct fcvmd_frequency_outputs {
    std::vector<double> phase_cos, phase_sin;
    arma::mat H;
};

class fast_cvmd final : public spectral_transform
{
    size_t levels = 0;
    std::map<freq_key_t, fcvmd_frequency_outputs> vmd_frequencies;

    void step_decompose_matrix(
            const std::vector<double> &phase_cos,
            const std::vector<double> &phase_sin,
            const std::vector<double> &values,
            const std::vector<double> &previous,
            std::vector<std::vector<double>> &decomposition,
            arma::mat &H) const;

public:
    std::map<freq_key_t, fcvmd_frequency_outputs> get_vmd_frequencies() const { return vmd_frequencies; }
    void set_vmd_frequencies(const std::map<freq_key_t, fcvmd_frequency_outputs> &_vmd_frequencies) { vmd_frequencies = _vmd_frequencies; }

    explicit fast_cvmd(const size_t levels);

    ~fast_cvmd() final = default;

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const size_t padding /* = 0 */) override;

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const std::string &table_name,
            const std::vector<double> &prev_decon);

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const std::string &table_name);

    void inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding /* = 0 */) const override;

    size_t get_residuals_length(const std::string &decon_queue_table_name);

    bool initialized(const std::string &decon_queue_table_name);
    void initialize(const std::vector<double> &input, const std::string &decon_queue_table_name);
};

}

#endif //SVR_FAST_CVMD_HPP
