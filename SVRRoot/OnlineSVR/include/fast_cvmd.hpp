//
// Created by zarko on 7/5/22.
//

#ifndef SVR_FAST_CVMD_HPP
#define SVR_FAST_CVMD_HPP

#include <deque>
#include <vector>
#include <armadillo>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_map.h>
#include "model/DataRow.hpp"
#include "model/DeconQueue.hpp"
#include "spectral_transform.hpp"

#define CVMD_INIT_LEN 1000000 // Last N samples of the input queue used for calculating frequencies

#define OMEGA_DIVISOR 1. // 1e2 // Divides frequency by N, 1. for XAUUSD, 14 for EURUSD
#define TAU_FIDELITY 0
#define HAS_DC 0 // Has DC component is false (its removed during scaling)
#define ALPHA_BINS 100 // 2000 for EURUSD, 1600 for XAUUSD
#define MAX_VMD_ITERATIONS 500
#define DEFAULT_PHASE_STEP 1
#define EPS std::numeric_limits<double>::epsilon()
#define CVMD_TOL EPS
// #define EMOS_OMEGAS
#define FASTER_CVMD
//#define ORTHO_CVMD

namespace svr {

typedef std::tuple<std::string /* decon queue table_name */, size_t /* levels */> freq_key_t;

struct fcvmd_frequency_outputs {
    arma::vec phase_cos;
    arma::vec phase_sin;
};

class fast_cvmd final : public spectral_transform
{
    size_t K; // Number of modes/frequencies. because of DC=1 we use 1 more than the natural number of frequencies (3 in the signal above). Half of VMD levels, quarter of total levels.
    arma::vec f;
    arma::rowvec A;
    arma::mat H;
    arma::uvec even_ixs, odd_ixs, K_ixs;
    bpt::ptime timenow;
    size_t levels;
    tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> vmd_frequencies;

public:
    tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> get_vmd_frequencies() const { return vmd_frequencies; }
    void set_vmd_frequencies(const tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> &_vmd_frequencies) { vmd_frequencies = _vmd_frequencies; }

    explicit fast_cvmd(const size_t levels);

    ~fast_cvmd() final = default;

    void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double>> &decon,
            const size_t padding /* = 0 */) override { THROW_EX_FS(std::logic_error, "Not implemented!"); };

    void
    transform(
            const data_row_container &input,
            datamodel::DeconQueue &decon,
            const size_t input_colix = 0,
            const size_t test_offset = 0);

    void inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding /* = 0 */) const override;

    size_t get_residuals_length(const std::string &decon_queue_table_name);

    bool initialized(const std::string &decon_queue_table_name);
    void initialize(const datamodel::datarow_crange &input, const size_t input_column_index, const std::string &decon_queue_table_name);
};

}

#endif //SVR_FAST_CVMD_HPP
