#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <iostream>
#include <fstream>
#include <limits>

#define ARMA_NO_DEBUG
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include "fast_functions.hpp"

int make_trainings(std::vector<std::vector<double>> &oemd_levels, std::vector<size_t> &test_positions, size_t six_minutes, size_t ahead, size_t back_skip, size_t back_size,
                   std::vector<std::vector<double>> &xtrain, std::vector<std::vector<double>> &ytrain,
                   std::vector<double> &trimmed_mean_vals, std::vector<double> &y_mean_val)
{
    std::cout << "Training " << std::endl;
    const size_t levels = oemd_levels.size();
    const size_t test_size = test_positions.size();
    xtrain.resize(levels);
    ytrain.resize(levels);
    for (int i = 0; i < levels; i++) {
        xtrain[i].resize(test_size * (back_size));
    }
    for (size_t i = 0; i < levels; i++) {
        ytrain[i].resize(test_size, 0.);
    }
    const size_t xsize = back_size;
    for (size_t j = 0; j < test_size; j++) {
        for (int i = 0; i < levels; i++) {
            for (size_t k = 0; k < back_size; k++) {
                xtrain[i][j * (back_size) + k] = (oemd_levels[i][test_positions[j] - (back_size - 1 - k) * back_skip] - oemd_levels[i][test_positions[j] - (back_size - 1 - k + 1) * back_skip]) / trimmed_mean_vals[i];
                //xtrain[i][j*(back_size)+k] = ( oemd_levels[i][  test_positions[j] - (back_size-1-k)*back_skip]-oemd_levels[i][test_positions[j]-(back_size-1-k+1)*back_skip]);
            }
            double s = 0;
            size_t count = 0;
            for (size_t k = 1; k <= ahead; k += 1) {
                s += oemd_levels[i][test_positions[j] + six_minutes + k];
                count++;
            }
            ytrain[i][j] = s / (double) count - oemd_levels[i][test_positions[j]];
        }
    }
    const double alpha = ALPHA_ONE_PERCENT;
    y_mean_val.resize(0);
    for (int i = 0; i < levels; i++) {
        std::vector<double>::iterator begin_it, end_it;
        begin_it = ytrain[i].begin();
        end_it = ytrain[i].end();
        double left_border, right_border, mean_val;
        find_range(alpha, begin_it, end_it, left_border, right_border, mean_val);
        y_mean_val.push_back(mean_val);
        //trimmed_mean_vals.push_back(1.);
        std::cout << "y " << right_border - left_border << " " << mean_val << std::endl;
    }
    for (size_t j = 0; j < test_size; j++) {
        for (int i = 0; i < levels; i++) {
            ytrain[i][j] = ytrain[i][j] / y_mean_val[i];
        }
    }
    return 0;
}

