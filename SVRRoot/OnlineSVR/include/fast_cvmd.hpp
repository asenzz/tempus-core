//
// Created by Zarko on 7/5/22.
//

#ifndef SVR_FAST_CVMD_HPP
#define SVR_FAST_CVMD_HPP

#include <deque>
#include <tuple>
#include <vector>
#include <armadillo>
#include <oneapi/tbb/concurrent_map.h>
#include "model/DataRow.hpp"
#include "model/DeconQueue.hpp"
#include "spectral_transform.hpp"
#include "IQScalingFactorService.hpp"

constexpr unsigned CVMD_INIT_LEN = 100'000; // Last N samples of the input queue used for calculating frequencies
constexpr auto TAU_FIDELITY = 1e3; // 1000 seems to yield best results
constexpr auto HAS_DC = false; // Has DC component is always false, it's removed during scaling
constexpr double C_default_alpha_bins = .016 * CVMD_INIT_LEN; // 2000 for EURUSD, 1600 for XAUUSD, Matlab examples use 50 on input length 1200 (CVMD_INIT_LEN * 50 / 1200 = 4166)
constexpr unsigned MAX_VMD_ITERATIONS = 500;
constexpr double DEFAULT_PHASE_STEP = 1; // Multiplies frequency by N, best so far 1e-1 for XAUUSD, 1/14 for EURUSD
constexpr size_t C_freq_init_type = 1; // Type of initialization, let's use 1 for now.
constexpr auto CVMD_TOL = 1e-7;
constexpr auto CVMD_EPS = 2 * std::numeric_limits<double>::epsilon();
const auto C_arma_solver_opts = arma::solve_opts::refine + arma::solve_opts::equilibrate;
// #define EMOS_OMEGAS

namespace svr {
namespace vmd {


typedef std::tuple<std::string /* decon queue table_name */, size_t /* levels */> freq_key_t;

struct fcvmd_frequency_outputs
{
    arma::vec phase_cos, phase_sin;
};

class fast_cvmd final : public spectral_transform
{
    const size_t levels = 0, K = 0; // Number of modes/frequencies. because of DC=1 we use 1 more than the natural number of frequencies (3 in the signal above). Half of VMD levels, quarter of total levels.
    arma::rowvec A;
    const arma::mat H;
    const arma::uvec even_ixs, odd_ixs;
    arma::vec f, row_values, soln;
    const bpt::ptime timenow;
    tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> vmd_frequencies;
    static const business::t_iqscaler C_no_scaler;

    static fcvmd_frequency_outputs compute_cos_sin(const arma::vec &omega, const double step = DEFAULT_PHASE_STEP);

public:
    tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> get_vmd_frequencies() const;

    void set_vmd_frequencies(const tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> &_vmd_frequencies);

    explicit fast_cvmd(const size_t levels_);

    ~fast_cvmd() final = default;

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const size_t padding /* = 0 */) override
    { THROW_EX_FS(std::logic_error, "Not implemented!"); };

    void
    transform(
            const data_row_container &input,
            datamodel::DeconQueue &decon,
            const size_t input_colix = 0,
            const size_t test_offset = 0,
            const business::t_iqscaler &scaler = C_no_scaler);

    void inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding /* = 0 */) const override;

    size_t get_residuals_length(const std::string &decon_queue_table_name);

    static size_t get_residuals_length(const unsigned levels);

    bool initialized(const std::string &decon_queue_table_name);

    void initialize(const datamodel::datarow_crange &input, const size_t input_column_index, const std::string &decon_queue_table_name,
                    const business::t_iqscaler &scaler = C_no_scaler);
};

}
}

#endif //SVR_FAST_CVMD_HPP
