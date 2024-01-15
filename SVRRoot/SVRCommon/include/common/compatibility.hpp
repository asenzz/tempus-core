#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <type_traits>

#include <armadillo>
#include <viennacl/matrix.hpp>

#include "types.hpp"
#include "defines.h"


namespace std {

//template<typename T, typename ...Args>
//std::unique_ptr<T> make_unique( Args&& ...args )
//{
//    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
//}

#ifdef __CYGWIN__
template<typename T>
string to_string(const T& obj){
    stringstream ss;
    ss << obj;
    return ss.str();
}
#endif
} // namespace std

namespace std {
    std::string to_string(const long double v);
    std::string to_string(const double v);
    std::string to_string(const float v);
}

namespace svr {

typedef std::shared_ptr<std::vector<arma::mat>> matrices_ptr;
typedef std::shared_ptr<arma::mat> matrix_ptr;

namespace common {

void print_stacktrace();

#define ELEMCOUNT(array) (sizeof(array)/sizeof(array[0]))


/* Return codes */
typedef size_t uint_rc_t;
#define SVR_RC_GENERAL_ERROR (std::numeric_limits<size_t>::max())


const long multi_div = 100000; // number of zeros tell the number of decimal points to assume when comparing doubles

// this is needed because there are some issues when comparing double values
// @see http://stackoverflow.com/questions/17333/most-effective-way-for-float-and-double-comparison
bool Equals(const double &lhs, const double &rhs);

double Round(const double &dbl);

template<typename T>
std::vector<std::shared_ptr<T>> inline
clone_shared_ptr_elements(const std::vector<std::shared_ptr<T>> &arg)
{
    std::vector<std::shared_ptr<T>> res;
    for (const std::shared_ptr<T> &p_elem: arg) res.push_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename T>
std::deque<std::shared_ptr<T>> inline
clone_shared_ptr_elements(const std::deque<std::shared_ptr<T>> &arg)
{
    std::deque<std::shared_ptr<T>> res;
    for (const std::shared_ptr<T> &p_elem: arg) res.push_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename K, typename T>
std::map<K, std::shared_ptr<T>> inline
clone_shared_ptr_elements(const std::map<K, std::shared_ptr<T>> &arg)
{
    std::map<K, std::shared_ptr<T>> res;
    for (const auto &pair: arg) res.emplace(K(pair.first), std::make_shared<T>(*pair.second));
    return res;
}


template<class K, class C, class A>
std::set<std::shared_ptr<K>, C, A> inline clone_shared_ptr_elements(const std::set<std::shared_ptr<K>, C, A> &arg)
{
    std::set<std::shared_ptr<K>, C, A> res;
    for (const auto &pair: arg) res.emplace(std::make_shared<K>(*pair));
    return res;
}

template<class K, class C, class A>
std::set<std::shared_ptr<K>, C, A> inline clone_shared_ptr_elements(std::set<std::shared_ptr<K>, C, A> &arg)
{
    std::set<std::shared_ptr<K>, C, A> res;
    for (const auto &pair: arg) res.emplace(std::make_shared<K>(*pair));
    return res;
}

ptimes_set_t
to_multistep_times(const ptimes_set_t &prediction_times,
                   const bpt::time_duration &resolution, const size_t &multistep_len);


ptimes_set_t
to_times(
        const boost::posix_time::time_period &prediction_range,
        const boost::posix_time::time_duration &resolution);

ptimes_set_t
to_times(
        const boost::posix_time::time_period &prediction_range,
        const boost::posix_time::time_duration &resolution,
        const size_t comb_train_ct,
        const size_t comb_validate_ct);

template<typename T> ptimes_set_t
to_times(const std::map<bpt::ptime, std::shared_ptr<T>> &data_rows)
{
    ptimes_set_t result;
    for (const auto &row: data_rows) result.insert(row.first);
    return result;
}


template<typename T> void
tovec(const arma::Mat<T> &input, std::vector<T> &output)
{
    output.resize(input.n_elem);
    output.shrink_to_fit();
    memcpy(output.data(), input.mem, input.n_elem * sizeof(T));
}

template<typename T> std::vector<T>
tovec(const arma::Mat<T> &input)
{
    std::vector<T> output(input.n_elem);
    tovec(input, output);
    return output;
}

viennacl::vector<double> tovcl(const arma::colvec &in);

template<typename T> viennacl::matrix<T>
tovcl(const arma::Mat<T> &in, const viennacl::ocl::context &cx)
{
    cl_int rc;
    auto clbuf = (T *)clCreateBuffer(cx.handle().get(), CL_MEM_READ_WRITE, in.n_elem * sizeof(T), nullptr, &rc);
    if (rc != CL_SUCCESS) throw std::runtime_error("Failed creating OpenCL buffer of size " + std::to_string(in.n_elem * sizeof(T)) + " with error " + std::to_string(rc));
    viennacl::matrix<T> r(clbuf, viennacl::OPENCL_MEMORY, in.n_cols, in.n_rows);
    viennacl::backend::memory_write(r.handle(), 0, in.n_elem * sizeof(T), in.memptr(), false);
    return viennacl::trans(r);
};

template<typename T> viennacl::matrix<T>
tovcl(const arma::Mat<T> &in)
{
    cl_int rc;
    auto hostbuf = (T *) malloc(in.n_elem * sizeof(T));
    viennacl::matrix<T> r(hostbuf, viennacl::MAIN_MEMORY, in.n_cols, in.n_rows);
    memcpy(r.handle().ram_handle().get(), in.mem, in.n_elem * sizeof(T));
    return viennacl::trans(r);
};


template<typename T> arma::Mat<T>
toarma(const viennacl::matrix<T> &in)
{
    arma::Mat<T> r(in.internal_size1(), in.internal_size2());

    switch (in.memory_domain()) {
        case viennacl::memory_types::OPENCL_MEMORY:
        case viennacl::memory_types::CUDA_MEMORY:
            viennacl::backend::memory_read(in.handle(), 0, in.internal_size() * sizeof(T), r.memptr());
            break;
        case viennacl::memory_types::MAIN_MEMORY:
            memcpy(r.memptr(), in.handle().ram_handle().get(), in.internal_size() * sizeof(T));
            break;
        default:
            throw std::invalid_argument("Unknown memory domain of matrix " + std::to_string(in.memory_domain()));
    }

    if (in.row_major()) r = r.t();
    if (in.size1() != in.internal_size1()) r.shed_rows(in.size1(), in.internal_size1() - 1);
    if (in.size2() != in.internal_size2()) r.shed_cols(in.size2(), in.internal_size2() - 1);
    return r;
};


template<typename Container, typename ConstIterator>
typename Container::iterator remove_constness(Container &c, ConstIterator it)
{
    return c.erase(it, it);
}

#if 0 // ArrayFire related routines, deprecated
af::array armat_to_af_2d(const arma::mat &input)
{
    return af::array(input.n_rows, input.n_cols, (double *) input.memptr());
}

af::array d_armat_to_af_2d(const arma::mat &input)
{
    af_array output;
    dim_t dims[2] = {dim_t(input.n_rows), dim_t(input.n_cols)};
    af_create_array(&output, (double*)input.memptr(), 2, dims, f64);
    return af::array(output);

}


arma::mat af_to_armat_2d(const af::array &input)
{
    std::vector<double> raw_data(input.elements());
    input.host(raw_data.data());
    arma::mat from_af(raw_data.data(), input.dims(0), input.dims(1));
    return from_af;
}

arma::mat af_product(const arma::mat &a, const arma::mat &b)
{
    LOG4_BEGIN();
    auto context = svr::common::gpu_context();
    arma::mat product;
    {
        af::setDevice(context.id());
        {
        af_device_mem_info()
            af::deviceGC();
            af::array af_a = armat_to_af_2d(a);
            af::array af_b = armat_to_af_2d(b);
            af::array af_c = af::matmul(af_a, af_b);
            product = af_to_armat_2d(af_c);
            af_a = af::array();
            af_b = af::array();
            af_c = af::array();
            af::sync(context.id());
        }
        af::deviceGC();
    }
    LOG4_END();
    return product;
}
#endif

} // namespace common
} // namespace svr

