#include "oemd_coefficients.hpp"
#include "common/Logging.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include <map>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <fstream>


namespace svr {

const std::string C_oemd_fir_coefs_dir = getenv("BACKTEST") ? "../libs/oemd_fir_masks_xauusd_1s_backtest/" : "../libs/oemd_fir_masks_xauusd_1s/";

oemd_coefficients get_oemd_level_coefficients(const size_t level_count)
{
    const std::vector<size_t> siftings (level_count - 1, DEFAULT_SIFTINGS);
	const std::vector<std::vector<double>> mask = [&]() -> std::vector<std::vector<double>> {
        std::vector<std::vector<double>> r(level_count - 1);
#pragma omp parallel for
        for (size_t l = 0; l < level_count - 1; ++l) {
            ssize_t ver = MASK_FILE_MAX_VER;
            std::string mask_full_path;
            std::ifstream ifs;
            do mask_full_path = common::formatter() << C_oemd_fir_coefs_dir << MASK_FILE_NAME(--ver, l, level_count);
            while (ver >= 0 && !(ifs = std::ifstream(mask_full_path)));
            if (!ifs || ver < 0) {
                LOG4_ERROR("Couldn't find file " << mask_full_path << " for level " << l << " in " << C_oemd_fir_coefs_dir);
                continue;
            }
            char coef_str[MAX_TOKEN_SIZE];
            while(ifs.is_open() && ifs.good()) {
                ifs.getline(coef_str, MAX_TOKEN_SIZE, ',');
                const auto val = std::strtod(coef_str, nullptr);
                if (!common::isnormalz(val)) LOG4_ERROR("FIR coefs " << val << " not normal.");
                r[l].emplace_back(val);
            }
            LOG4_DEBUG("Read " << r[l].size() << " coefficients for level " << l << " of " << level_count << " levels from " << mask_full_path);
        }
        return r;
    } ();
	return oemd_coefficients{level_count, siftings, mask};
}

std::pair<const size_t, const oemd_coefficients> load(const size_t levels)
{
    return {levels, get_oemd_level_coefficients(levels)};
}

const std::map<const size_t, const oemd_coefficients> oemd_available_levels = []() -> std::map<const size_t, const oemd_coefficients> {
    return {
        load(16),
        load(8),
        load(6),
        load(4),
        load(2),
        load(1),
    };
} ();



}
