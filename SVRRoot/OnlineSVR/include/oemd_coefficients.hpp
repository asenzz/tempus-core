#ifndef SVR_OEMD_COEFFICIENTS_HPP
#define SVR_OEMD_COEFFICIENTS_HPP

#include <vector>
#include <map>
#include <cstddef>
#include <string>

#include "util/string_utils.hpp"


#define DEFAULT_SIFTINGS 1
#define MASK_FILE_NAME(CTR, L, LEVEL_COUNT) "mask_" << CTR << "_level_" << L << "_of_" << LEVEL_COUNT << ".txt"
#define MAX_TOKEN_SIZE 0xFF
#define MASK_FILE_MAX_VER 0x20


namespace svr {

struct oemd_coefficients {
    const size_t levels;
    std::vector<size_t> siftings;
    std::vector<std::vector<double>> masks;
    oemd_coefficients(
            const size_t levels_, const std::vector<size_t>& siftings_, const std::vector<std::vector<double>>& mask_) :
            levels(levels_), siftings(siftings_), masks(mask_){}
};

extern const std::string C_oemd_fir_coefs_dir;
extern const std::map<const size_t, const oemd_coefficients> oemd_available_levels;

}

#endif //SVR_OEMD_COEFFICIENTS_HPP
