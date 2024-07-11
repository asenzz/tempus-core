//
// Created by zarko on 7/7/24.
//
#include "common/parallelism.hpp"

namespace svr {

t_omp_lock::t_omp_lock()
{
    omp_init_lock(&l);
}

t_omp_lock::operator omp_lock_t *()
{
    return &l;
}

void t_omp_lock::set()
{
    omp_set_lock(&l);
}

void t_omp_lock::unset()
{
    omp_unset_lock(&l);
}

t_omp_lock::~t_omp_lock()
{
    omp_destroy_lock(&l);
}

}