#ifndef SVR_OEMD_COEFFICIENTS_HPP
#define SVR_OEMD_COEFFICIENTS_HPP

#include <vector>
#include <deque>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_map.h>
#include <cstddef>
#include <string>

#include "util/string_utils.hpp"

constexpr unsigned DEFAULT_SIFTINGS = 1;

constexpr unsigned MASK_FILE_MAX_VER = 0x20;
constexpr double OEMD_STRETCH_COEF = 1; // MAIN_DECON_QUEUE_RES_SECS * OMEGA_DIVISOR // 10 * MAIN_DECON_QUEUE_RES_SECS * OMEGA_DIVISOR // Fat OEMD transformer


namespace svr {

std::string mask_file_name(const size_t ctr, const size_t level, const size_t level_count, const std::string &queue_name);

const std::string C_oemd_fir_coefs_dir = getenv("BACKTEST") ? "../libs/oemd_fir_masks_xauusd_1s_backtest/" : "../libs/oemd_fir_masks_xauusd_1s/";

struct oemd_coefficients;
typedef std::shared_ptr<oemd_coefficients> t_oemd_coefficients_ptr;
typedef tbb::concurrent_map<std::pair<const size_t, const std::string>, const t_oemd_coefficients_ptr> t_coefs_cache;

struct oemd_coefficients {
    std::deque<size_t> siftings;
    std::deque<std::vector<double>> masks;

    oemd_coefficients() : siftings({}), masks({}) {};

    oemd_coefficients(
            const std::deque<size_t> &siftings_, const std::deque<std::vector<double>>& mask_) : siftings(siftings_), masks(mask_) {}

    static t_oemd_coefficients_ptr load(const size_t level_count, const std::string &queue_name);
};

}

#endif //SVR_OEMD_COEFFICIENTS_HPP
