#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <atomic>
#include "appcontext.hpp"
#include "common/logging.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "oemd_coefficients.hpp"


namespace svr {

oemd_coefficients::oemd_coefficients() : siftings({}), masks({})
{};

oemd_coefficients::oemd_coefficients(const std::deque<uint16_t> &siftings_, const std::deque <std::vector<double>> &mask_) : siftings(siftings_), masks(mask_)
{}

std::string oemd_coefficients::get_mask_file_name(const uint16_t ctr, const uint16_t level, const uint16_t level_count, const std::string &queue_name)
{
    std::stringstream s;
    s << PROPS.get_oemd_masks_dir() << "mask_v" << ctr << "_level_" << level << "_of_" << level_count << "_" << queue_name << ".csv";
    return s.str();
}

t_oemd_coefficients_ptr oemd_coefficients::load(const uint16_t level_count, const std::string &queue_name)
{
    const std::deque<uint16_t> siftings(level_count - 1, C_default_siftings);
    std::deque <std::vector<double>> masks(level_count - 1);
    std::atomic<bool> except = false;
    OMP_FOR(level_count - 1)
    for (DTYPE(level_count) l = 0; l < level_count - 1; ++l)
        if (!except) {
            ssize_t ver = C_mask_file_max_ver;
            std::string mask_full_path;
            std::ifstream ifs;
            do
                mask_full_path = get_mask_file_name(--ver, l, level_count, queue_name); // Find latest version of FIR coefficients
            while (!(ifs = std::ifstream(mask_full_path)) && ver > 0);
            if (!ifs || ver < 0) {
                LOG4_ERROR("Couldn't find file " << mask_full_path << " for level " << l << " in " << PROPS.get_oemd_masks_dir());
                except = true;
                continue;
            }
            std::array<char, common::C_max_csv_token_size> coef_str;
            coef_str.fill(0);
            while (ifs.getline(coef_str.data(), common::C_max_csv_token_size, ',')) {
                const auto val = std::strtod(coef_str.data(), nullptr);
                if (!common::isnormalz(val)) {
                    LOG4_ERROR("FIR coefficients " << val << " not normal.");
                    except = true;
                    continue;
                }
                masks[l].emplace_back(val);
            }
            LOG4_DEBUG("Read " << masks[l].size() << " coefficients for level " << l << " of " << level_count << " levels from " << mask_full_path);
        }
    if (except) masks.clear();

    return std::make_shared<oemd_coefficients>(siftings, masks);
}


}
