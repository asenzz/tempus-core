#ifndef SVR_KERNEL_PATH_HPP
#define SVR_KERNEL_PATH_HPP

#include <vector>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <ctime>

#ifdef VIENNACL_WITH_OPENCL

#include "common/gpu_handler.tpp"

#endif

#include "kernel_base.hpp"

namespace svr {

template<typename scalar_type>
class kernel_path : public kernel_base<scalar_type>
{
private:
public:
    explicit kernel_path(const SVRParameters &p) : kernel_base<scalar_type>(p)
    {}
#if 0 // TODO Port to arma::mat
    scalar_type operator()(const vektor<scalar_type> &va, const vektor<scalar_type> &vb)
    {
        const size_t lag_count = this->parameters.get_lag_count();
#ifndef NDEBUG
        if (sizeof(scalar_type) != sizeof(double))
            throw std::invalid_argument(
                    svr::common::formatter() << "We don't know if it works - scalar_type is not double!");

        if (lag_count <= 0)
            throw std::invalid_argument(
                    svr::common::formatter() << "Lag count <= 0 !" << this->parameters.get_lag_count());
#endif /* NDEBUG */

        const int dimvect = va.size() / lag_count;

#ifndef NDEBUG
        if (va.size() != (unsigned) (lag_count * dimvect))
            throw std::invalid_argument(
                    svr::common::formatter() << "We don't know what is happening!" << va.size() << " " << lag_count
                                             << " " << dimvect);


        if (dimvect <= 0) throw std::invalid_argument(svr::common::formatter() << "PATH dimvect <= 0 !" << dimvect);
#endif /* NDEBUG */

        // length of the first vector equals the lag_count
        const int nX = (int) lag_count;
        // length of the second vector equals the first one in our case
        // but they can be different in general
        const int nY = nX;

        std::vector<double> w(lag_count * lag_count);
        path_weights_modif(w, lag_count, lag_count);//no sum in CPU version
        const double sig = -1. / (2. * this->parameters.get_svr_kernel_param() * this->parameters.get_svr_kernel_param());
	    const double lambda = this->parameters.get_svr_kernel_param2();
        double result = 0;
#pragma omp parallel for reduction(+:result) default(shared) collapse(2)
        for (int i = 0; i < nX; i++) {
            for (int j = 0; j < nY; ++j) {
                double sum = 0;
                if (w[i * nY + j] > 0.001)
                    for (int ii = 0; ii < dimvect; ++ii)
                        sum += (va[i + ii * nX] - vb[j + ii * nY]) * (va[i + ii * nX] - vb[j + ii * nY]);
                result += w[i * nY + j] * sum;
            }
        }
        return exp(sig * pow(result, lambda));
    }

    void operator()(const vmatrix<scalar_type> &features, vmatrix<scalar_type> &kernel_matrix) // not used currently
    {
        LOG4_WARN("Kernel matrix computation on CPU is very slow!");
        omp_tpfor__(ssize_t, row, 0, features.get_length_rows(),
                    for (ssize_t row2 = 0; row2 < features.get_length_rows(); ++row2)
                kernel_matrix.set_value(row, row2, this->operator()(features[row], features[row2]));
        )
    }
#endif

    void operator()(const viennacl::matrix<scalar_type> &features, viennacl::matrix<scalar_type> &kernel_matrix)
    {
#ifdef VIENNACL_WITH_OPENCL
        common::gpu_context ctx;
        gpu_compute(features, kernel_matrix, ctx.ctx());
#else
        LOG4_THROW("CPU Path kernel not working yet!");
#endif
    }

#ifdef VIENNACL_WITH_OPENCL


    int gpu_compute(const viennacl::matrix<scalar_type> &features, viennacl::matrix<scalar_type> &kernel_matrix, viennacl::ocl::context &ctx)
    {
//TODO - block size should not be 3072, but variable
        LOG4_BEGIN();

        cl12::command_queue cq_command_queue(ctx.get_queue().handle());
        cl12::ocl_context context = ctx.handle();

        LOG4_DEBUG("Starting CPU side of OpenCL when gamma is " << this->parameters.get_svr_kernel_param() << ", lambda is " << this->parameters.get_svr_kernel_param2());

        if (kernel_matrix.size1() != features.size1() or kernel_matrix.size2() != features.size1())
            kernel_matrix.resize(features.size1(), features.size1(), false);

        cl_int err = CL_SUCCESS;
        if (features.handle().opencl_handle().get() == NULL) {
            LOG4_DEBUG("The kernel expects features matrix on the device!");
            THROW_EX_FS(std::invalid_argument, "OpenCL features NULL pointer.");
        }

        if (sizeof(scalar_type) != sizeof(double)) THROW_EX_FS(std::invalid_argument, "We don't know if it works - scalar_type is not double!");

        if (features.size1() < 1) LOG4_THROW("Empty features matrix passed for GAK computation on GPUs.");

        const auto sigma = this->parameters.get_svr_kernel_param();
        const size_t lag_count = this->parameters.get_lag_count();
        if (lag_count <= 0) THROW_EX_FS(std::invalid_argument, "Lag count <= 0 !" << this->parameters.get_lag_count());
        const int dimvect = features.size2() / lag_count;
        if (dimvect <= 0) THROW_EX_FS(std::invalid_argument, "PATH dimvect <= 0 !" << dimvect);
        const double lambda = this->parameters.get_svr_kernel_param2();
        if (lambda < 0) THROW_EX_FS(std::invalid_argument, "Invalid PATH second kernel parameter less than zero! " << lambda);
        const size_t size1 = features.size1();
        const size_t size2 = features.size2();
        const size_t block_size = 3072;
        const size_t local_work_size[2] = {common::C_cu_tile_width, common::C_cu_tile_width};
        const size_t N_groups_max_0 = block_size / local_work_size[0];
        const size_t N_groups_max_1 = block_size / local_work_size[1];

        std::vector<double> supermin_GPU(N_groups_max_0 * N_groups_max_1);

        cl::Buffer supermin_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * N_groups_max_0 * N_groups_max_1, NULL, &err);
        CL_CHECK(err);

        cl_mem features_gpu_ptr = features.handle().opencl_handle().get();
        if (features_gpu_ptr == NULL) THROW_EX_FS(std::invalid_argument, "Features are not on the GPU! - NULL pointer!");
        cl_mem kernel_gpu_ptr;
        cl::Buffer kernel_matrix_gpu_d;
        if (kernel_matrix.handle().ram_handle().get() != NULL) {
            kernel_matrix_gpu_d = cl::Buffer(
                    context, CL_MEM_READ_WRITE,
                    sizeof(cl_double) * kernel_matrix.size1() * kernel_matrix.internal_size2(),
                    NULL, &err);
            CL_CHECK(err);
            kernel_gpu_ptr = kernel_matrix_gpu_d();
        } else {
            kernel_gpu_ptr = kernel_matrix.handle().opencl_handle().get();
        }

        const size_t output_internal_size2 = kernel_matrix.internal_size2();
        err = cq_command_queue.finish();
        CL_CHECK(err);

        svr::cl12::ocl_kernel forward1(ctx.get_kernel("path_kernel_xx", "path_kernel_run").handle());

        const size_t input_internal_size2 = features.internal_size2();

        forward1.set_args(
                features_gpu_ptr,
                input_internal_size2,
                output_internal_size2,
                size2, //3 nXy
                dimvect, // 4 dimvect
                sigma, // 5 sigma
                kernel_gpu_ptr, // 6 total_result_d
                supermin_d); // 7 supermin_d*/
        err = cq_command_queue.finish();
        CL_CHECK(err);
        const size_t next_arg_ix = 8;
        path_kernel_math_run_calculations_on_OpenCL(
                features, kernel_matrix, block_size, size1, size2, size1, dimvect,
                forward1, cq_command_queue, supermin_GPU, N_groups_max_0, N_groups_max_1, supermin_d, next_arg_ix, lambda);
        CL_CHECK(err);
        err = cq_command_queue.finish();
        CL_CHECK(err);

        if (kernel_matrix.handle().ram_handle().get() == NULL) {
            LOG4_TRACE("Matrix produced on GPU as requested");
        } else {
            err = cq_command_queue.enqueueReadBuffer(
                    kernel_matrix_gpu_d, CL_TRUE, 0,
                    sizeof(cl_double) * size1 * output_internal_size2,
                    kernel_matrix.handle().ram_handle().get());
            CL_CHECK(err);
            err = cq_command_queue.finish();
            CL_CHECK(err);
        }
        LOG4_END();
        return 0;
    }

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "path_kernel_xx");
        gpu_compute(features, kernel_matrix, ctx);
    }

    using kernel_base<scalar_type>::operator();

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &x,
            const viennacl::matrix<scalar_type> &y,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        svr::common::gpu_kernel::ensure_compiled_kernel(ctx, "path_kernel_xy");
        gpu_compute(x, y, kernel_matrix, ctx);
    }

    int gpu_compute(
            const viennacl::matrix<scalar_type> &x,
            const viennacl::matrix<scalar_type> &y,
            viennacl::matrix<scalar_type> &kernel_matrix,
            viennacl::ocl::context &ctx)
    {
        LOG4_BEGIN();

        cl12::command_queue cq_command_queue(ctx.get_queue().handle());
        cl12::ocl_context context = ctx.handle();

        LOG4_DEBUG("Starting CPU side of OpenCL when gamma is " << this->parameters.get_svr_kernel_param() << ", lambda is " << this->parameters.get_svr_kernel_param2());

        if (kernel_matrix.size1() != x.size1() || kernel_matrix.size2() != y.size1())
            kernel_matrix.resize(x.size1(), y.size1(), false);
        cl_int err = CL_SUCCESS;

        if (x.handle().opencl_handle().get() == NULL) {
            THROW_EX_FS(std::invalid_argument, "OpenCL device NULL pointer " << (size_t) (x.handle().opencl_handle().get()));
        }

        if (sizeof(scalar_type) != sizeof(double))
            THROW_EX_FS(std::invalid_argument, "We don't know if it works - scalar_type is not double!");

        if (x.size1() < 1) LOG4_THROW("Empty X matrix passed for GAK computation on GPUs.");

        const auto sigma = this->parameters.get_svr_kernel_param();
        const size_t lag_count = this->parameters.get_lag_count();
        if (lag_count <= 0) THROW_EX_FS(std::invalid_argument, "Lag count <= 0 !" << this->parameters.get_lag_count());

        const int dimvect = x.size2() / lag_count;
        if (dimvect <= 0) THROW_EX_FS(std::invalid_argument, "PATH dimvect <= 0 !" << dimvect);

        const double lambda = this->parameters.get_svr_kernel_param2();
        if (lambda < 0) THROW_EX_FS(std::invalid_argument, "Invalid PATH second kernel parameter < 0 !" << lambda);

        const size_t x_size1 = x.size1();
        const size_t x_size2 = x.size2();
        const size_t y_size1 = y.size1();
        const size_t y_size2 = y.size2();

        //Number of threads (work items) per Compute Unit (work group)
        //16*16=256 - this is the maximum size for older AMD cards.
        const size_t local_work_size[2] = {common::C_cu_tile_width, common::C_cu_tile_width};
        const size_t block_size = 3072;
        const size_t N_groups_max_0 = block_size / local_work_size[0];
        const size_t N_groups_max_1 = block_size / local_work_size[1];

        std::vector<double> supermin_GPU(N_groups_max_0 * N_groups_max_1);
        const size_t output_internal_size2 = kernel_matrix.internal_size2();

        cl::Buffer supermin_d(context, CL_MEM_READ_WRITE, sizeof(cl_double) * N_groups_max_0 * N_groups_max_1, NULL, &err);
        CL_CHECK(err);

        cl_mem x_gpu_ptr = x.handle().opencl_handle().get();
        cl_mem y_gpu_ptr = y.handle().opencl_handle().get();

        cl_mem kernel_gpu_ptr;
        cl::Buffer kernel_matrix_gpu_d;
        if (kernel_matrix.handle().ram_handle().get() != NULL) {
            kernel_matrix_gpu_d = cl::Buffer(
                    context, CL_MEM_READ_WRITE,
                    sizeof(cl_double) * kernel_matrix.size1() * kernel_matrix.internal_size2(),
                    NULL, &err);
            CL_CHECK(err);
            kernel_gpu_ptr = kernel_matrix_gpu_d();
        } else {
            kernel_gpu_ptr = kernel_matrix.handle().opencl_handle().get();
        }

        err = cq_command_queue.finish();
        CL_CHECK(err);

        svr::cl12::ocl_kernel forward1(ctx.get_kernel("path_kernel_xy", "path_kernel_run").handle());
        const size_t input_internal_size2 = x.internal_size2();
        forward1.set_args(
                x_gpu_ptr, //0 X_d
                y_gpu_ptr, // 1 Y_d
                input_internal_size2, // 2 M
                x_size1, // 3 nX or internal_size1 ?
                x_size2, //4 nXy
                output_internal_size2, // 5
                y_size2, // 6
                dimvect, // 7 dimvect ???
                sigma, // 8 sigma
                kernel_gpu_ptr, // 9 Total_result_d
                supermin_d); // 10 supermin_d

        err = cq_command_queue.finish();
        CL_CHECK(err);

        const size_t next_arg_ix = 11;

        path_kernel_math_run_calculations_on_OpenCL(
                x, kernel_matrix, block_size, x_size1, x_size2, y_size1, dimvect,
                forward1, cq_command_queue, supermin_GPU, N_groups_max_0, N_groups_max_1, supermin_d, next_arg_ix,lambda);
        CL_CHECK(err);

        err = cq_command_queue.finish();
        CL_CHECK(err);
        if (kernel_matrix.handle().ram_handle().get() == NULL) {
            LOG4_DEBUG("Matrix produced on GPU as requested");
        } else {
            err = cq_command_queue.enqueueReadBuffer(
                    kernel_matrix_gpu_d, CL_TRUE, 0,
                    sizeof(cl_double) * x_size1 * output_internal_size2,
                    kernel_matrix.handle().ram_handle().get());
            CL_CHECK(err);
            err = cq_command_queue.finish();
            CL_CHECK(err);
            LOG4_DEBUG("Matrix produced on CPU as requested");
        }

        LOG4_END();
        return 0;
    }


    static double path_weights_modif_only_sum(const int M, const int N)
    {
        double w_sum_sym = 0;
#pragma omp parallel for reduction(+: w_sum_sym) collapse(2) default(shared) num_threads(adj_threads(M*N))
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                w_sum_sym += 1. / ((1. + abs(i - j)) * (1. + abs(i - j)));
        return w_sum_sym;
    }

    void path_weights_modif(std::vector<double> &w, const int M, const int N)
    {
        //type_float w = zeros(M,N);
        double w_sum = 0;
        //for ii = 1:M
#pragma omp parallel for reduction(+: w_sum) collapse(2) default(shared) num_threads(adj_threads(M*N))
        for (int ii = 0; ii < M; ++ii) {
            //for jj=1:N
            for (int jj = 0; jj < N; ++jj) {
                const auto iijj = ii * N + jj;
                w[iijj] = ii == jj ? 1. : 1. / std::pow<double>(1. + abs(ii - jj), 2);
                w_sum += w[iijj];
            }
        }

#pragma omp parallel for default(shared) num_threads(adj_threads(M*N))
        //w=w/sum(sum(w));
        for (int ii = 0; ii < M; ++ii)
            for (int jj = 0; jj < N; ++jj)
                w[ii * N + jj] /= w_sum;
    }

    void path_kernel_math_run_calculations_on_OpenCL(
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix,
            const size_t block_size, const size_t nX, const size_t nXy,
            const size_t nY,
            const int dimvect,
            svr::cl12::ocl_kernel forward1,
            cl12::command_queue cq_command_queue,
            std::vector<double> supermin_GPU,
            size_t N_groups_max_0, size_t N_groups_max_1,
            cl::Buffer supermin_d, size_t next_arg_ix, double lambda)
    {

//        const int len = nXy / dimvect; // lag
        //std::vector<double> w(len * len);
        //path_weights_modif(w, len, len); //not actually used
        const double w_sum_sym = 1; //path_weights_modif_only_sum(len, len);

        size_t number_of_blocks1 = nX / block_size;
        if (number_of_blocks1 * block_size < nX) ++number_of_blocks1;

        size_t number_of_blocks2 = nY / block_size;
        if (number_of_blocks2 * block_size < nY) ++number_of_blocks2;

        cl_int err;

        size_t changing_args_ix = next_arg_ix;
        for (size_t ii = 0; ii < number_of_blocks1; ii++) {
            for (size_t jj = 0; jj < number_of_blocks2; jj++) {
                const size_t startX = ii * block_size;
                const size_t finishX = std::min(nX, (ii + 1) * block_size);

                const size_t startY = jj * block_size;
                const size_t finishY = std::min(nY, (jj + 1) * block_size);

                const size_t numX = finishX - startX;
                const size_t numY = finishY - startY;

                err = cq_command_queue.finish();
                CL_CHECK(err);
                next_arg_ix = changing_args_ix;

                // Set kernel arguments for variables that are defined in function.
                forward1.set_arg(next_arg_ix++, startX);
                forward1.set_arg(next_arg_ix++, startY);
                forward1.set_arg(next_arg_ix++, numX);
                forward1.set_arg(next_arg_ix++, numY);
                forward1.set_arg(next_arg_ix++, w_sum_sym);
                forward1.set_arg(next_arg_ix++, lambda);

                err = cq_command_queue.finish();
                CL_CHECK(err);

                const size_t local_work_size[2] = {common::C_cu_tile_width, common::C_cu_tile_width}; // TODO replace with proper hardware checks
                const size_t global_work_size[2] = {
                        ((int) (numX / local_work_size[0]) + (bool) (numX % local_work_size[0])) *
                        local_work_size[0],
                        ((int) (numY / local_work_size[1]) + (bool) (numY % local_work_size[1])) *
                        local_work_size[1]};

                err = forward1.enqueue(
                        cq_command_queue,
                        cl::NullRange, svr::cl12::ndrange(global_work_size), svr::cl12::ndrange(local_work_size),
                        nullptr, nullptr);
                CL_CHECK(err);

                err = cq_command_queue.finish();
                CL_CHECK(err);
            }
        }
    }

#if 0 // TODO Port to armadillo
    virtual void
    operator()(viennacl::ocl::context &ctx, const vektor<double> &test_vector, const viennacl::matrix<double> &learning, vektor<double> &kernel_values)
    {
        viennacl::matrix<double> Y((size_t) 1, test_vector.size(), ctx);
        viennacl::matrix<double> kernel_matrix(learning.size1(), (size_t) 1, ctx);
        // TODO The following copy below is very slow, replace ASAP!
        for (ssize_t i = 0; i < test_vector.size(); ++i) Y(0, i) = test_vector[i];
        (*this)(ctx, learning, Y, kernel_matrix);
        kernel_values.resize(learning.size1());
        // TODO The following copy below is very slow, replace ASAP!
        for (size_t i = 0; i < learning.size1(); ++i)
            kernel_values[i] = kernel_matrix(i, 0);
    }
#endif

#endif /* #ifdef VIENNACL_WITH_OPENCL */
};

}

#endif //SVR_KERNEL_PATH_HPP
