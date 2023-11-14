#pragma once

#include "vektor.hpp"
#include "common/Logging.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include <cmath>
#include <limits>
#include <thread>
#include <future>
#include <unordered_set>

namespace svr {
namespace datamodel {

static const ssize_t step_size = 64;
static const std::string cm_separator{","}; // TODO Unify with one in StringUtils.cpp

// INITIALIZATION
template<class T>
vektor<T>::vektor() : vektor(step_size)
{
    length = 0;
}

template<class T>
vektor<T>::vektor(const T *p_values, const ssize_t new_length) : vektor(new_length)
{
    resize(new_length);
    std::memcpy(&(values_ptr[0]), &(p_values[0]), sizeof(T) * new_length);
}

template<class T>
vektor<T>::vektor(const vektor<T> &v)
        : vektor(&v)
{
}

template<class T>
vektor<T>::vektor(const vektor<T> *const X)
        : vektor(X, X->length)
{
}


template<class T>
vektor<T>::vektor(const vektor<T> *X, const ssize_t N)
        : // values(N, viennacl::context(viennacl::MAIN_MEMORY)),
        length(0),
        max_length(0)
{
    resize(N);
    /* cilk_ */ for (ssize_t i = 0; i < N; ++i) {
        values_ptr[i] = X->get_value(i);
    }
}


template<class T>
vektor<T>::vektor(const std::vector<T> &X) : length(0), max_length(0)
{
    resize(X.size());
    viennacl::fast_copy(X.begin(), X.end(), values.begin());
}


template<class T>
vektor<T>::vektor(const viennacl::vector<T> &vcl, const ssize_t new_length)
        : values(0, viennacl::context(viennacl::MAIN_MEMORY)), length(0), max_length(0)
{/*
    values.resize(max_length, true);
    update_values_ptr();
    std::memcpy(values_ptr, VCL_DOUBLE_PTR(vcl), length * sizeof(T));
    */
    // TODO Bug: if N < length(X), this is memory corruption?

    // This is correct usage, design specific to viennacl.
    resize(length);
    viennacl::copy(vcl, values);
}


template<class T>
vektor<T>::vektor(const arma::Row <T> &v)
        : values(0, viennacl::context(viennacl::MAIN_MEMORY)), length(0), max_length(0)
{
    resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) values_ptr[i] = v(i);
}


template<class T>
vektor<T> &vektor<T>::operator=(const vektor<T> &v)
{
    resize(v.size());
    if (!v.empty()) viennacl::copy(v.values, values);
    return *this;
}

template<class T>
vektor<T> &vektor<T>::operator=(const viennacl::vector<T> &vcl)
{
    resize(vcl.size());
    if (!vcl.empty()) viennacl::copy(vcl, values);
    return *this;
}


template<typename T> std::vector<T>
vektor<T>::to_vector()
{
    std::vector<T> result(length);
    memcpy(result.data(), values_ptr, length * sizeof(T));
    return result;
}


template<class T>
vektor<T>::vektor(const T value, const ssize_t new_length) : length(0), max_length(0)
{
    resize(new_length);
    for (ssize_t i = 0; i < new_length; ++i) values_ptr[i] = value;
}

template<class T>
vektor<T>::vektor(const size_t length_arg)
        : values(snap_new_size(length_arg), viennacl::context(viennacl::MAIN_MEMORY)),
          length(0),
          max_length(0)
{
    resize(length_arg);
}


template<class T>
vektor<T>::~vektor()
{
    //this->Clear();
}


namespace {

template<class T>
inline bool comparer(T const &t1, T const &t2);

template<>
inline bool comparer(viennacl::const_entry_proxy<double> const &t1, viennacl::const_entry_proxy<double> const &t2)
{
    const double t1_d = t1;
    const double t2_d = t2;
    return svr::common::equal_to(t1_d, t2_d);
}

template<>
inline bool comparer(double const &t1, double const &t2)
{
    return svr::common::equal_to(t1, t2);
}

template<>
inline bool comparer(viennacl::const_entry_proxy<int> const &t1, viennacl::const_entry_proxy<int> const &t2)
{
    return t1 == t2;
}

template<>
inline bool comparer(int const &t1, int const &t2)
{
    return t1 == t2;
}

template<>
inline bool comparer(ssize_t const &t1, ssize_t const &t2)
{
    return t1 == t2;
}

template<>
inline bool comparer(size_t const &t1, size_t const &t2)
{
    return t1 == t2;
}


}


template<class T>
bool vektor<T>::operator==(vektor<T> const &o) const
{
    bool result = length == o.length;
    if (!result)
        return false;

    for (size_t i = 0; i < size_t(length); ++i)
        if (!comparer(values_ptr[i], o.values_ptr[i]))
            return false;

    return true;
}

#define THIS_VCL VCL((*this))
#define VCL(V) viennacl::vector_base<T>(V.values.handle(), V.size(), 0, 1)

template<class T>
std::shared_ptr<viennacl::vector<T> > vektor<T>::vcl()
{
    // Makes a clone without actual copy (using the same memory chunk).
    return std::make_shared<viennacl::vector<T> >(THIS_VCL);
}

template<class T>
const viennacl::vector<T> &vektor<T>::get_values() const
{
    if (length != max_length)
        throw std::runtime_error(
                "Illegal operate on raw data when length not equal allocated size!");
    return values;
}


template<class T>
void vektor<T>::copy_to(std::vector<T> &stl_vec)
{
    stl_vec.resize(length);
    viennacl::vector_base<T> vcl(values.handle(), size(), 0, 1);
    viennacl::fast_copy(vcl.begin(), vcl.end(), stl_vec.begin());
}


template<typename T>
void vektor<T>::copy_to(vektor<T> &v) const
{
    v.resize(length);
    memcpy(v.values_ptr, values_ptr, sizeof(T) * length); // TODO Replace with fast memcpy
}


template<typename T>
void vektor<T>::copy_to(viennacl::vector<T> &v) const
{
    v.resize((size_t) length);
    viennacl::copy(THIS_VCL, v);
}


template<class T>
viennacl::backend::mem_handle &vektor<T>::handle()
{
    return values.handle();
}

template<class T>
const viennacl::backend::mem_handle &vektor<T>::handle() const
{
    return values.handle();
}


template<class T>
vektor<T> *vektor<T>::clone() const
{
    return new vektor<T>(this);
}


template<class T>
size_t vektor<T>::internal_size() const
{
    return values.size();
}


template<class T>
void vektor<T>::resize(const ssize_t new_length)
{
    if (new_length > max_length) memresize(new_length);
    length = new_length;
}


template<typename T>
void vektor<T>::trim()
{
    if (max_length == length) return;
    max_length = length;
    values.resize(length);
    update_values_ptr();
}

template<typename T>
void vektor<T>::reverse()
{
    std::reverse(values_ptr, values_ptr + length);
}

template<class T>
void vektor<T>::memresize()
{
    memresize((max_length + step_size) * 2); // Multiply by 2 like stl vector does.
}

template<class T>
void vektor<T>::memresize(const ssize_t new_size_)
{
    const auto modified_new_size_ = snap_new_size(new_size_);
    values.resize(modified_new_size_);
    max_length = modified_new_size_;
    update_values_ptr();
}

template<class T>
void vektor<T>::memresize(const size_t new_size, const T fill_value)
{
    const auto old_size = values.size();
    memresize(new_size);
    if (new_size > old_size) for (auto ix = old_size; ix < new_size; ++ix) values_ptr[ix] = fill_value;
}

template<class T>
size_t vektor<T>::snap_new_size(const size_t size) const
{
    return (1 + size / step_size) * step_size;
}

template<class T>
void vektor<T>::update_values_ptr()
{
    values_ptr = reinterpret_cast<T *>(this->values.handle().ram_handle().get());
}

template<class T>
T const &vektor<T>::get_value(const ssize_t sample_index) const
{
#ifndef NDEBUG
    if (sample_index < 0 || sample_index >= length)
        throw std::range_error(svr::common::formatter() << "Vector index " << sample_index << " out of bounds.");
#endif
    return values_ptr[sample_index];
}

template<class T>
bool vektor<T>::contains(const T value) const
{
    for (ssize_t i = 0; i < length; ++i) if (values_ptr[i] == value) return true;
    return false;
}

template<typename T>
size_t vektor<T>::count(const T value, const T epsilon) const
{
#if 0
    cilk::reducer<cilk::op_add<size_t>> red_count(0);
    /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)if (svr::common::equal_to(values_ptr[i], value, epsilon))
            *red_count += 1;
    return red_count.get_value();
#endif
    return 0;
}

// Add/Remove Operations
template<class T>
void vektor<T>::clear()
{
    length = 0;
}

// TODO optimize
template<class T>
void vektor<T>::add(T const &X)
{
    if (length >= max_length) memresize();
    values_ptr[length++] = X;
}

template<class T>
void vektor<T>::add_safe(T const &X)
{
    std::unique_lock<std::mutex> _(vektor_mutex);
    if (length == max_length) memresize();
    values_ptr[length++] = X;
}

template<class T>
void vektor<T>::add_fast(T &&X)
{
    values_ptr[length++] = std::move(X);
}

template<class T>
void vektor<T>::add_at(T const &X, const ssize_t index)
{
    if (index >= 0 && index <= length) {
        if (length == max_length) memresize();
        std::memmove(&(values_ptr[index + 1]), &(values_ptr[index]), sizeof(T) * (length - index));
        values_ptr[index] = X;
        length++;
    } else
        throw std::range_error(svr::common::formatter() << "Impossible to add an element in invalid index " << index);
}

template<class T>
void vektor<T>::add_at_safe(const ssize_t index, T const &X)
{
    if (index >= 0 && index <= length) {
        if (length == max_length) memresize();
        std::memmove(&(values_ptr[index + 1]), &(values_ptr[index]), sizeof(T) * (length - index));
        values_ptr[index] = X;
        length++;
    } else
        throw std::range_error(svr::common::formatter() << "Impossible to add an element in invalid index " << index);
}

template<class T>
void
vektor<T>::remove_at(const ssize_t index)
{
    if (length > 0 && index >= 0 && index < length) {
        std::memmove(&(values_ptr[index]), &(values_ptr[index + 1]), sizeof(T) * (length - index - 1));
        length--;
    } else
        throw std::range_error(svr::common::formatter() << "It's impossible to remove element " << index
                                                        << " from the vector that doesn't exist.");
}


template<class T>
void
vektor<T>::remove_range(const ssize_t start_index, const ssize_t end_index)
{
    if (length < 1 || start_index < 0 || start_index >= length || end_index < 0 || end_index >= length ||
        start_index > end_index)
        throw std::range_error(svr::common::formatter() << "Illegal range " << start_index << " to " << end_index);

    std::memmove(&(values_ptr[start_index]), &(values_ptr[end_index]), sizeof(T) * (length - end_index));
    length -= end_index - start_index;
}


template<class T>
vektor<T> *
vektor<T>::extract(const ssize_t from_index, const ssize_t to_index) const
{
    if (from_index >= 0 && to_index <= length - 1 && from_index <= to_index) {
        const auto v_length = 1 + to_index - from_index;
        auto V = new vektor<T>(0., v_length);
        //vektor<T> *V = new vektor<T>(std::vector<T>(values_ptr + from_index, values_ptr + to_index + 1));
        /* cilk_ */ for (ssize_t ix = from_index; ix <= to_index; ++ix)V->set_at(ix - from_index, values_ptr[ix]);
        return V;
    } else
        throw std::range_error(svr::common::formatter() << "Vector indexes " << from_index << " and " << to_index
                                                        << " out of bounds, length " << length);
}


// Pre-built Vectors
template<typename T>
vektor<double> *vektor<T>::zero_vector(const ssize_t length_)
{
    vektor<double> *p_zero_vec = new vektor<double>(0., length_);
    return p_zero_vec;
}

template<typename T>
vektor<double> *vektor<T>::rand_vector(const ssize_t length_)
{
    auto V = new vektor<double>(0., length_);
    /* cilk_ */ for (ssize_t i = 0; i < length_; ++i)V->set_at(i, static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
    return V;
}

template<class T>
vektor<T> *vektor<T>::get_sequence(T start, T step, T end_)
{
    vektor<T> *V = new vektor<T>();
    if (start < end_) for (T i = start; i <= end_; i += step) V->add(i);
    else for (T i = start; i >= end_; i -= step) V->add(i);
    return V;
}


// Mathematical Operations
template<class T>
void vektor<T>::sum_scalar(T X)
{
    vektor<T> v2(X, size());
    sum_vector(v2);
}

template<class T>
void vektor<T>::product_scalar(T X)
{
    THIS_VCL *= X;
}

template<class T>
void vektor<T>::divide_scalar(T X)
{
    if (X != T(0)) {
// Performance optimization: it's faster to use regular operations
//        THIS_VCL /= X;
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i) values_ptr[i] /= X;
    } else
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)values_ptr[i] = svr::common::SIGN<T>(values_ptr[i]) * INF;
}

template<class T>
void vektor<T>::pow_scalar(T X)
{
// Performance optimization: it's faster to use regular operations
//    vektor<T> v2(X, get_length());
//    THIS_VCL = viennacl::linalg::element_pow(THIS_VCL, VCL(v2));
    /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)values_ptr[i] = std::pow(values_ptr[i], X);
}

template<class T>
void vektor<T>::square_scalar()
{
    pow_scalar(2);
}


template<class T>
void vektor<T>::sum_vector(const vektor<T> *V)
{
    if (length == V->length)
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)values_ptr[i] += V->values_ptr[i];
    else
        throw std::invalid_argument("It's impossible to sum two vectors with different length.");
}


template<class T>
void vektor<T>::sum_vector(const vektor<T> &V)
{
    if (length == V.length)
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)values_ptr[i] += V.values_ptr[i];
    else
        throw std::invalid_argument("It's impossible to sum two vectors with different length.");
}


template<class T>
vektor<T> *vektor<T>::sum_vector(const vektor<T> *V1, const vektor<T> *V2)
{
    if (V1->length == V2->length) {
        vektor *V3 = new vektor(V1->values, V1->length);
        /* cilk_ */ for (decltype(V1->length) i = 0; i < V1->length; ++i)V3->values_ptr[i] += V2->values_ptr[i];
        return V3;
    } else
        throw std::invalid_argument("It's impossible to sum two vectors with different length.");
}

template<class T>
void vektor<T>::subtract_vector(const vektor<T> *V)
{
    if (length == V->length) {
// Performance optimization: it's faster to use regular operations
//        THIS_VCL -= VCL((*V));
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i) values_ptr[i] -= V->values_ptr[i];
    } else
        throw std::invalid_argument("It's impossible to subtract two vectors with different length.");
}

template<class T>
void vektor<T>::subtract_vector(const vektor<T> &V)
{
    if (length == V.length)
// Performance optimization: it's faster to use regular operations
//        THIS_VCL -= VCL((*V));
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i) values_ptr[i] -= V.values_ptr[i];
    else
        throw std::range_error("It's impossible to subtract two vectors with different length.");
}


template<class T>
vektor<T> vektor<T>::subtract_vector(const vektor<T> &V1, const vektor<T> &V2)
{
    const auto v1_len = V1.length;
    if (v1_len == V2.length) {
// Performance optimization: it's faster to use regular operations
//        vektor V3 = V1->clone();
//        VCL((*V3)) -= VCL((*V2));
        vektor<T> V3(v1_len);
        V3.resize(v1_len);
        for (ssize_t i = 0; i < v1_len; ++i) {
            __builtin_prefetch(&(V3.values_ptr[i]));
            V3.values_ptr[i] = V1.values_ptr[i] - V2.values_ptr[i];
        }
        return V3;
    } else
        throw std::range_error(svr::common::formatter() <<
                                                        "It's impossible to subtract vector two vectors with different lengths "
                                                        << V1.size() << " and " << V2.size());
}

template<class T>
void vektor<T>::product_vector(vektor<T> &V)
{
    *this = product_vector(*this, V);
}

template<typename T>
vektor<T> vektor<T>::product_vector(vektor<T> &V1, vektor<T> &V2)
{
    if (V1.length == V2.length) {
        vektor<T> V3(V1.length);
        V3.resize(V1.length);
        /* cilk_ */ for (ssize_t i = 0; i < V1.length ; ++i) {
            __builtin_prefetch(&(V3.values_ptr[i]));
            V3.values_ptr[i] = V1.values_ptr[i] * V2.values_ptr[i];
        }
        return V3;
    } else
        throw std::runtime_error("Incompatible vector lengths!");
}

template<class T>
T vektor<T>::product_vector_scalar(const vektor<T> *V) const
{
    return product_vector_scalar(this, V);
}


template<class T>
T vektor<T>::product_vector_scalar(const vektor<T> &V) const
{
    return product_vector_scalar(*this, V);
}


template<class T>
T vektor<T>::product_vector_scalar(const vektor<T> *V1, const vektor<T> *V2)
{
    return product_vector_scalar_cpu(*V1, *V2);
}


template<class T>
T vektor<T>::product_vector_scalar(const vektor<T> &V1, const vektor<T> &V2)
{
    return product_vector_scalar_cpu(V1, V2);
}


template<class T>
T vektor<T>::product_vector_scalar_cpu(const vektor<T> &V1, const vektor<T> &V2)
{
    const auto len = V1.length;
    if (len != V2.length)
        throw std::invalid_argument(
                svr::common::formatter() << "Incompatible vector lengths " << V1.length << " and " << V2.length);
//    cilk::reducer<cilk::op_add<T>> product(T{0});
//    /* cilk_ */ for (ssize_t i = 0; i < len; ++i) *product += V1.values_ptr[i] * V2.values_ptr[i];
    return 0;
}


template<class T>
T vektor<T>::product_vector_scalar_gpu(vektor<T> &V1, vektor<T> &V2)
{
    const auto length1 = V1.length;
    if (length1 == V2.length) return svr::common::dot_product(V1.values_ptr, V2.values_ptr, length1);
    else
        throw std::invalid_argument(
                svr::common::formatter() <<
                                         "It's impossible to multiply two vectors with different lengths " << V1.length
                                         << " and " << V2.length);
}


template<class T>
T vektor<T>::sum() // TODO put back const when armadilloing this code
{
#if 0
    cilk::reducer<cilk::op_add<T>> red_sum(0.);
    /* cilk_ */ for (ssize_t i = 0; i < length; ++i) *red_sum +=values_ptr[i];
    return red_sum.get_value();
#endif
    return 0;
}


template<class T>
T vektor<T>::abs_sum() // const
{
#if 0
    cilk::reducer<cilk::op_add<T>> red_sum(0.);
    /* cilk_ */ for (ssize_t i = 0; i < length; ++i) *red_sum +=std::abs(values_ptr[i]);
    return red_sum.get_value();
#endif
    return 0;
}


template<typename T>
vektor<T> *vektor<T>::abs_list() // const
{
    vektor<T> *X = new vektor<T>(this);
    X->abs();
    return X;
}


template<typename T>
void vektor<T>::abs() // const
{
    const viennacl::vector<T> v_abs = viennacl::linalg::element_fabs(THIS_VCL);
    viennacl::copy(v_abs.begin(), v_abs.end(), values.begin());
}

// Comparison Operations
template<class T>
T vektor<T>::min() // const
{
    T min_value = 0;
    if (length > 0) min_value = viennacl::linalg::min(THIS_VCL);
    return min_value;
}

template<class T>
void vektor<T>::min(T &min_value, ssize_t &min_index) const
{
#if 0
    if (length > 0) {
        cilk::reducer<cilk::op_min_index<ssize_t, T>> red_min_idx;
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)red_min_idx->calc_min(i, values_ptr[i]);
        min_index = red_min_idx->get_index_reference();
        min_value = red_min_idx->get_reference();
    } else {
        min_value = std::numeric_limits<T>::quiet_NaN();
        min_index = -1;
    }
#endif
}

template<class T>
T vektor<T>::min_abs() const
{
    if (length > 0) {
        vektor<T> V2(*this);
        V2.abs();
        T min_value = V2.min();
        return min_value;
    } else {
        T min_value;
        return min_value;
    }
}

template<class T>
void vektor<T>::min_abs(T &min_value, ssize_t &min_index) const
{
#if 0
    if (length < 1) {
        min_value = std::numeric_limits<T>::quiet_NaN();
        min_index = -1;
        return;
    }
    cilk::reducer<cilk::op_min_index<ssize_t, T>> red_min_idx;
    /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)red_min_idx->calc_min(i, svr::common::ABS<T>(values_ptr[i]));
    min_index = red_min_idx->get_index_reference();
    min_value = red_min_idx->get_reference();
#endif
}

template<class T>
void vektor<T>::min_abs_nonzero(T &min_value, ssize_t &min_index) const
{
    if (length < 1) {
        min_value = std::numeric_limits<T>::quiet_NaN();
        min_index = -1;
        return;
    }
#if 0
    cilk::reducer<cilk::op_min_index<ssize_t, T>> red_min_idx;
    /* cilk_ */ for (decltype(length) i = 0; i < length; ++i) {
        const auto cur_abs_value = svr::common::ABS<T>(values_ptr[i]);
        if (!svr::common::is_zero(cur_abs_value))
            red_min_idx->calc_min(i, cur_abs_value);
    }
    min_index = red_min_idx->get_index_reference();
    min_value = red_min_idx->get_reference();
#endif
}

template<class T>
T vektor<T>::max() // const
{
    if (length > 0) return viennacl::linalg::max(THIS_VCL);
    else return std::numeric_limits<T>::quiet_NaN();
}

template<class T>
void vektor<T>::max(T *max_value, int *max_index) const
{
#if 0
    if (length > 0) {
        cilk::reducer<cilk::op_max_index<ssize_t, T>> red_max_idx;
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)red_max_idx->calc_max(i, values_ptr[i]);
        max_index = red_max_idx->get_index_reference();
        max_value = red_max_idx->get_reference();
    } else {
        max_value = std::numeric_limits<T>::quiet_NaN();
        max_index = -1;
    }
#endif
}

template<class T>
T vektor<T>::max_abs() // const
{
    if (length > 0) {
        vektor<T> V2(*this);
        V2.abs();
        T max_value = V2.max();
        return max_value;
    } else {
        return std::numeric_limits<T>::quiet_NaN();
    }
}

template<class T>
void vektor<T>::max_abs(T *max_value, int *max_index) const
{
#if 0
    if (length > 0) {
        cilk::reducer<cilk::op_max_index<ssize_t, T>> red_max_idx;
        /* cilk_ */ for (decltype(length) i = 0; i < length; ++i)red_max_idx->calc_max(i, svr::common::ABS<T>(
                    values_ptr[i]));
        max_index = red_max_idx->get_index_reference();
        max_value = red_max_idx->get_reference();
    } else {
        max_value = std::numeric_limits<T>::quiet_NaN();
        max_index = -1;
    }
#endif
}

template<class T>
T vektor<T>::mean() // const
{
    T mean_value = 0;
    if (length > 0) mean_value = sum() / length;
    return mean_value;
}

template<class T>
T vektor<T>::mean_abs() // const
{
    T mean_value{(T) 0};
    if (length > 0) mean_value = abs_sum() / length;
    return mean_value;
}

// TODO: Population or Sample Variance?
// TODO: Previous impl probably tried Sample Variance, however it was wrong and
// the result was Population Variance with trimmed first sample.
template<class T>
T vektor<T>::variance() // const
{
    T variance{0};
    if (length > 0) {
        vektor v2(*this);
        v2.sum_scalar(-v2.mean());
        v2.pow_scalar(2);
        variance = v2.sum() / v2.size();
    }
    return variance;
}

// Sorting Operations
// TODO: Check if these could be impl entirely with viennacl...
//template<class T>
//void vektor<T>::Sort() {
//    std::sort(values_ptr, values_ptr + Length);
//}
//
//template<class T>
//void vektor<T>::RemoveDuplicates() {
//    std::vector<T> cpu_vec_unique;
//    std::unordered_set<T> uniques;
//
//    for (int i = 0; i < GetLength(); ++i) {
//       if (uniques.count(values_ptr[i]) == 0) {
//           uniques.insert(values_ptr[i]);
//           cpu_vec_unique.push_back(values_ptr[i]);
//       }
//    }
//
//    viennacl::fast_copy(cpu_vec_unique.begin(), cpu_vec_unique.end(), this->Values.begin());
//    this->Length = cpu_vec_unique.size();
//}

template<class T>
void vektor<T>::sort_and_remove_duplicates()
{
    std::sort(values_ptr, values_ptr + length);
    auto new_end = std::unique(values_ptr, values_ptr + length);
    length = std::distance(values_ptr, new_end);
}

template<class T>
ssize_t vektor<T>::find(T X) const // TODO Parallelize
{
    for (decltype(length) i = 0; i < length; ++i) if (values_ptr[i] == X) return i;
    return -1;
}

// I/O Operations
template<class T>
vektor<T> *vektor<T>::load(const char *filename)
{
    // Open the file
    std::ifstream afile(filename, std::ios::in);
    if (!afile) {
        LOG4_ERROR("It's impossible to open the file.");
        return new vektor<T>();
    }
    afile.precision(std::numeric_limits<double>::max_digits10);
    auto V = new vektor<T>();
    // Load the vector
    try {
        T v;
        while (!afile.eof()) {
            afile >> v;
            if (afile.good()) V->add(v);
        }
    } catch (const std::exception &ex) {
        LOG4_ERROR("It's impossible to complete the load. " << ex.what());
    }
    // Close the file
    afile.close();

    return V;
}


template<class T>
vektor<T> vektor<T>::load(const std::string &filename)
{
    vektor<T> res;
    // Open the file
    std::ifstream afile(filename, std::ios::in);
    if (!afile) {
        LOG4_ERROR("It's impossible to open the file.");
        return res;
    }
    afile.precision(std::numeric_limits<double>::max_digits10);
    // Load the vector
    try {
        T v;
        while (!afile.eof()) {
            afile >> v;
            if (afile.good()) res.add(v);
        }
    } catch (const std::exception &ex) {
        LOG4_ERROR("It's impossible to complete the load. " << ex.what());
    }
    // Close the file
    afile.close();

    return res;
}


template<class T>
void vektor<T>::save(const char *filename) const
{
    // Open the file
    std::ofstream afile(filename, std::ios::out);
    if (!afile) {
        LOG4_ERROR("It's impossible to create the file.");
        return;
    }
    afile.precision(std::numeric_limits<double>::max_digits10);
    // Save the vector
    try {
        for (int i = 0; i < this->length; i++) afile << this->values_ptr[i] << " ";
        afile << std::endl;
    } catch (const std::exception &ex) {
        LOG4_ERROR("It's impossible to complete the save. " << ex.what());
    }
    // Close the file
    afile.close();
}

template<class T>
void vektor<T>::print() const
{
    std::stringstream ss;
    for (int i = 0; i < length; ++i) ss << values_ptr[i] << " ";
    LOG4_INFO(ss.str());
}

template<class T>
void vektor<T>::print(const std::string &vector_name) const
{
    LOG4_INFO(vector_name);
    print();
}

// Operators Redefinition
template<class T>
T &vektor<T>::operator[](const ssize_t index)
{
#ifndef NDEBUG
    if (index < 0 || index >= length) throw std::range_error("Vector index out of bounds");
#endif
    return values_ptr[index];
}

template<class T>
const T &vektor<T>::operator[](const ssize_t index) const
{
#ifndef NDEBUG
    if (index < 0 || index >= length) throw std::range_error("Vector index out of bounds");
#endif
    return values_ptr[index];
}

template<class T>
void vektor<T>::set_at(const ssize_t index, const T value)
{
#ifndef NDEBUG
    if (index < 0 || index >= length) throw std::range_error("Vector index out of bounds");
#endif
    values_ptr[index] = value;
}

template<class T>
size_t vektor<T>::size_of() const
{
    return values.size() * sizeof(T);
}

template<typename T>
std::ostream &
operator<<(std::ostream &os, const svr::datamodel::vektor<T> &v)
{
    for (ssize_t ix = 0; ix < v.size(); ++ix) os << v.get_value(ix) << cm_separator;
    return os;
}


template<typename T> arma::Mat<T>
vektor_to_admat(const vektor<T> &input)
{
    arma::Mat<T> output(1, input.size());
    /* cilk_ */ for (decltype(input.size()) i = 0; i < input.size(); ++i) output(0, i) = input[i];
    return output;
}


template<typename T> vektor<T>
admat_to_vektor(const arma::Mat<T> &input)
{
    svr::datamodel::vektor<T> output(0., input.size());
    size_t ctr = 0;
    for (size_t row_ix = 0; row_ix < (size_t) size(input, 0); ++row_ix)
        for (size_t col_ix = 0; col_ix < (size_t) size(input, 1); ++col_ix)
            output[ctr++] = input(row_ix, col_ix);
    return output;
}


}
}
