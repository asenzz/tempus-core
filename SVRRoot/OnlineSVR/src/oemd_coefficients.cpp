#include <cctype>
#include <locale>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <atomic>
#include "oemd_coefficients.hpp"
#include "common/Logging.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"


namespace svr {


std::string mask_file_name(const size_t ctr, const size_t level, const size_t level_count, const std::string &queue_name)
{
    std::stringstream s;
    s << "mask_" << ctr << "_level_" << level << "_of_" << level_count << "_" << queue_name << ".csv";
    return s.str();
}

t_oemd_coefficients_ptr
oemd_coefficients::load(const size_t level_count, const std::string &queue_name)
{
    const std::deque<size_t> siftings (level_count - 1, DEFAULT_SIFTINGS);
	std::deque<std::vector<double>> masks(level_count - 1);
    std::atomic<bool> stop = false;
#pragma omp parallel for
    for (size_t l = 0; l < level_count - 1; ++l) {
        if (stop.load(std::memory_order::memory_order_relaxed)) continue;
        ssize_t ver = MASK_FILE_MAX_VER;
        std::string mask_full_path;
        std::ifstream ifs;
        do mask_full_path = common::formatter() << C_oemd_fir_coefs_dir << mask_file_name(--ver, l, level_count, queue_name); // Find latest version of FIR coefs
        while (ver >= 0 && !(ifs = std::ifstream(mask_full_path)));
        if (!ifs || ver < 0) {
            LOG4_ERROR("Couldn't find file " << mask_full_path << " for level " << l << " in " << C_oemd_fir_coefs_dir);
            stop.store(true, std::memory_order_relaxed);
            continue;
        }
        char coef_str[MAX_TOKEN_SIZE];
        while (ifs.is_open() && ifs.good()) {
            ifs.getline(coef_str, MAX_TOKEN_SIZE, ',');
            const auto val = std::strtod(coef_str, nullptr);
            if (!common::isnormalz(val)) LOG4_ERROR("FIR coefs " << val << " not normal.");
            masks[l].emplace_back(val);
        }
        LOG4_DEBUG("Read " << masks[l].size() << " coefficients for level " << l << " of " << level_count << " levels from " << mask_full_path);
    }
    if (stop.load(std::memory_order::memory_order_relaxed)) masks.clear();

	return std::make_shared<oemd_coefficients>(siftings, masks);
}


}
