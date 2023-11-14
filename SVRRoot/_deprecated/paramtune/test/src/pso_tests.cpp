#include <gtest/gtest.h>
#include <functional>
#include <cmath>

#include "pso.hpp"
#include <common/Logging.hpp>

TEST(Paramtune, PSO)
{
    std::function<double (const std::vector<double>&)> loss_fun = [](const std::vector<double>& point) -> double {
        double sum = 0;
        for (size_t i = 0; i < point.size(); i++) {
            sum += std::pow(point[i], 2);
        }
        return sum;
    };

    pso_settings_t settings;
    pso_set_default_settings(&settings);

    settings.dim = 100;
    settings.x_lo = std::vector<double>(settings.dim, -300);
    settings.x_hi = std::vector<double>(settings.dim, 300);
    settings.goal = 1e-5;

    settings.steps = 10000;
    settings.size = 30;

    settings.nhood_strategy = PSO_NHOOD_RANDOM;
    settings.nhood_size = 10;
    settings.w_strategy = PSO_W_LIN_DEC;

    std::vector<std::vector<double> > result = pso_solve(loss_fun, &settings);

    double sum_square = 0;
    for (size_t i = 0; i < result.size(); i++) {
        for (size_t j = 0; j < result[i].size(); j++) {
            sum_square += std::pow(result[i][j], 2);
            //LOG4_DEBUG("result[" << i << "][" << j << "] = " << result[i][j]);
        }
    }

    LOG4_DEBUG("sum_square = " << sum_square);
}



