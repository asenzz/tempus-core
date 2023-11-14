//
// Created by zarko on 2/24/22.
//

#include "../include/dq_scaling_factors_service_impl.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <cublas_v2.h>


namespace svr {
namespace business {

struct abs_functor
{
    template <typename scalar_t> __host__ __device__  scalar_t
    operator()(const scalar_t& z)
    {
        return z < 0 ? -z : z;
    }
};

void calc_mean_max(
    const thrust::device_vector<double>::iterator &start_it,
    const thrust::device_vector<double>::iterator &end_it,
    const size_t alpha_n,
    const size_t rows,
    double &mean,
    double &max)
{
    mean = thrust::reduce(thrust::device, start_it, end_it, double(0), thrust::plus<double>());
    mean /= double(rows);

    thrust::transform(thrust::device, start_it, end_it, start_it, abs_functor());
    thrust::sort(thrust::device, start_it, end_it);
    max = *(start_it + alpha_n);
}

// TODO Test, not tested!
void cu_get_nth_max(
        const std::vector<double> &flat_row_matrix,
        const size_t alpha_n,
        const size_t levels,
        const size_t rows,
        std::vector<double> &means,
        std::vector<double> &scaling_factor,
        const int id)
{
    cudaSetDevice(id);
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t handle;
    thrust::device_vector<double> dev_flat_col_matrix(flat_row_matrix), dev_flat_row_matrix(flat_row_matrix);
    cublasCreate(&handle);
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, levels, &alpha, thrust::raw_pointer_cast(dev_flat_col_matrix.data()), levels, &beta,
                thrust::raw_pointer_cast(dev_flat_col_matrix.data()), rows, thrust::raw_pointer_cast(dev_flat_row_matrix.data()), rows);

    for (size_t l = 0; l < rows; ++l) {
        double mean, max;
        calc_mean_max(dev_flat_col_matrix.begin() + rows * l, dev_flat_col_matrix.begin() + rows * (l + 1), alpha_n, rows, mean, max);
        means.push_back(mean);
        scaling_factor.push_back(max);
    }
    cublasDestroy(handle);
}

}
}
