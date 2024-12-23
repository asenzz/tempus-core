#ifndef SVR_OEMD_COEFFICIENTS_HPP
#define SVR_OEMD_COEFFICIENTS_HPP

#include <vector>
#include <deque>
#include <oneapi/tbb/concurrent_map.h>
#include <string>

namespace svr {

struct oemd_coefficients;
typedef std::shared_ptr<oemd_coefficients> t_oemd_coefficients_ptr;
typedef tbb::concurrent_map<std::pair<const uint16_t, const std::string>, const t_oemd_coefficients_ptr> t_coefs_cache;

struct oemd_coefficients {
    static constexpr uint16_t C_default_siftings = 2;
    static constexpr uint16_t C_mask_file_max_ver = 0x20;
    static constexpr double C_oemd_stretch_coef = 1;
    static const std::string C_oemd_fir_coefs_dir;

    static std::string get_mask_file_name(const uint16_t ctr, const uint16_t level, const uint16_t level_count, const std::string &queue_name);

    oemd_coefficients();

    oemd_coefficients(const std::deque<uint16_t> &siftings_, const std::deque<std::vector<double>> &mask_);

    std::deque<uint16_t> siftings;
    std::deque<std::vector<double>> masks;

    static t_oemd_coefficients_ptr load(const uint16_t level_count, const std::string &queue_name);
};

}

#endif //SVR_OEMD_COEFFICIENTS_HPP
