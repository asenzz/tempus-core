#include "common/constants.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "sobol.hpp"
#include "sobolnum.hpp"

namespace svr {


namespace {

constexpr unsigned sobol_max_dimension = 1024;
constexpr unsigned sobol_max_power = 51;
constexpr uint64_t pow2_51 = 2251799813685248; // 2^51
constexpr uint64_t pow2_52 = 2 * pow2_51;
constexpr double inversepow2_52 = 1. / pow2_52;

const auto sobol_direction_numbers_cross = []() {
    std::deque<std::deque<uint64_t>> result(sobol_max_power, std::deque<uint64_t>(sobol_max_dimension));
    OMP_FOR_(sobol_max_dimension * sobol_max_power, simd collapse(2))
    for (unsigned i = 0; i < sobol_max_power; ++i) {
        for (unsigned j = 0; j < sobol_max_dimension; ++j) {
            result[i][j] = sobol_direction_numbers[j][i];
            if (i) result[i][j] ^= sobol_direction_numbers[j][i - 1];
        }
    }
    return result;
}();

tbb::mutex sobol_mutex;

}


uint64_t init_sobol_ctr()
{
    static auto random_seed = .54321; // common::get_uniform_random_value(); // Use quasi-random seed for reproducibility
    const tbb::mutex::scoped_lock lock(sobol_mutex);
    auto res = 78786876896UL + (uint64_t) std::floor(random_seed * pow2_52);
    random_seed = std::fmod(random_seed + .123456789, 1.);
    return res;
}

double sobolnum(const unsigned dim, const uint64_t n)
{
    uint64_t P = 1;
    uint64_t result = 0;
UNROLL(sobol_max_power)
    for (unsigned i = 0; i < sobol_max_power; i++) {
        if (P > n) break;
        if (n & P) result ^= sobol_direction_numbers_cross[i][dim % sobol_max_dimension];
        P += P;
    }
    return std::fmod(double(result) * inversepow2_52, 1.);
}

} // namespace svr