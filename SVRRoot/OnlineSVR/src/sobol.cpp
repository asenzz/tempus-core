#include <cmath>
#include <atomic>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cassert>
#include <mutex>

#define SOBOL_IMPLEMENTATION

#include "sobolvec.h"

uint64_t sobol_direction_numbers_cross[sobol_max_power][sobol_max_dimension];
uint64_t sobol_direction_numbers_twix[sobol_max_power][sobol_max_dimension];


int do_load_directions()
{
    static std::once_flag sobol_flag;
    std::call_once(sobol_flag, [](){
        for (int i = 0; i < sobol_max_power; i++) {
            for (int j = 0; j < sobol_max_dimension; j++) {
                sobol_direction_numbers_twix[i][j] = sobol_direction_numbers[j][i];
                if (i == 0) {
                    sobol_direction_numbers_cross[i][j] = sobol_direction_numbers[j][i];
                } else {
                    sobol_direction_numbers_cross[i][j] = sobol_direction_numbers[j][i] ^ sobol_direction_numbers[j][i - 1];
                }
            }
        }
    });
    return 0;
}


//int do_sobol_init(long seed ,int maxdim=SobolMaxDimension){
//static thread_local unsigned long SobolRandomInit[SobolMaxDimension];

int do_sobol_init()
{
    do_load_directions();
    return 0;
}


double sobolnum(const int Dim, const unsigned long N)
{
    uint64_t P = 1;
    uint64_t result = 0;
    for (size_t i = 0; i < sobol_max_power; i++) {
        if (P > N) break;
        if (N & P) result ^= sobol_direction_numbers_cross[i][Dim];
        P += P;
    }
    return double(result) * inversepow2_52;
}


int do_test()
{

    std::cout << std::setprecision(16);
    for (int N = 0; N < 64; N++) {
        for (int Dim = 0; Dim < 1024; Dim++) {
            std::cout << sobolnum(Dim, N) << std::endl;
        }
    }


    return 0;
}

int do_main_not_used()
{
    do_sobol_init();
    do_test();

    return 0;
}



