
#pragma once

#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mutex>



#include "common.hpp"


namespace svr {
namespace datamodel {


template<class T>
class vektor
{
public:
    // Initialization
    vektor();

    explicit vektor(const size_t length_arg);

    explicit vektor(const vektor<T> * const X);

    explicit vektor(const std::vector<T> &X);

    vektor<T>& operator=(const vektor<T> &v);

    vektor<T>& operator=(const viennacl::vector<T> &vcl);

    vektor(const vektor<T> &v);

    vektor(const T *p_values, const ssize_t new_length);

    vektor(const vektor<T> *X, const ssize_t N);

    vektor(const T value, const ssize_t new_length);

    vektor(viennacl::vector<T> const &vcl, const ssize_t new_length);

    ~vektor();

    bool operator==(vektor<T> const &) const;

    void copy_to(std::vector<T> &stl_vec);

    void copy_to(vektor<T> &v) const;

    void copy_to(arma::vec<T> &v) const;

    std::shared_ptr<arma::vec<T> > vcl();

    const arma::vec<T> &get_values() const;

    //viennacl::backend::mem_handle &handle();

    //viennacl::backend::mem_handle const &handle() const;

    vektor<T> *clone() const;

    ssize_t size() const { return length; };

    void resize(const ssize_t new_length);

    size_t internal_size() const;

    T const &get_value(const ssize_t sample_index) const;

    void set_value(const ssize_t sample_index, T const &value);

    bool contains(const T value) const;

    size_t count(const T value, const T epsilon = std::numeric_limits<T>::epsilon()) const;

    // Add/Remove Operations
    void clear();

    void add(T const &X);

    void add_safe(T const &X);

    //No Lenght check
    void add_fast(T &&X);

    void add_at(T const &X, const ssize_t index);

    void add_at_safe(const ssize_t index, T const &X);

    void remove_at(const ssize_t index);

    void remove_range(const ssize_t start_index, const ssize_t end_index);

    vektor<T> *extract(const ssize_t from_index, const ssize_t to_index) const;

    // Pre-built Vectors
    static vektor<double> *zero_vector(const ssize_t length_);

    static vektor<double> *rand_vector(const ssize_t length_);

    static vektor<T> *get_sequence(T start, T step, T end_);

    // Mathematical Operations
    void sum_scalar(T X);

    void product_scalar(T X);

    void divide_scalar(T X);

    void pow_scalar(T X);

    void square_scalar();

    void sum_vector(const vektor<T> *V);

    void sum_vector(const vektor<T> &V);

    static vektor<T> *sum_vector(const vektor<T> *V1, const vektor<T> *V2);

    void subtract_vector(const vektor<T> *V);

    void subtract_vector(const vektor<T> &V);

    static vektor <T> subtract_vector(const vektor <T> &V1, const vektor <T> &V2);

    void product_vector(vektor<T> &V);

    static vektor<T> product_vector(vektor<T> &V1, vektor<T> &V2);

    T product_vector_scalar(const vektor<T> *V);
    T product_vector_scalar(const vektor<T> &V);

    static T product_vector_scalar(const vektor<T> *V1, const vektor<T> *V2);
    static T product_vector_scalar(const vektor<T> &V1, const vektor<T> &V2);
    static T product_vector_scalar_cpu(const vektor <T> &V1, const vektor <T> &V2);
    static T product_vector_scalar_gpu(vektor <T> &V1, vektor <T> &V2);

    T sum() const;

    T abs_sum() const;

    vektor<T> *abs_list() const;

    void abs();

    // Comparison Operations
    T min() const;

    void min(T &min_value, ssize_t &min_index) const;

    T min_abs() const;

    void min_abs(T &min_value, ssize_t &min_index) const;

    void min_abs_nonzero(T &min_value, ssize_t &min_index) const;

    T max() const;

    void max(T *max_value, int *max_index) const;

    T max_abs() const;

    void max_abs(T *max_value, int *max_index) const;

    // TODO Implement truncated mean!
    T mean() const;

    T mean_abs() const;

    T variance() const;

    // Sorting Operations
    void sort_and_remove_duplicates();

    ssize_t find(T X) const;

    // I/O Operations
    static vektor<T> *load(const char *filename);

    void save(const char *filename) const;

    void print() const;

    void print(const std::string &vector_name) const;

    // Operators Redefinition
    T &operator[](const ssize_t index);

    const T &operator[](const ssize_t index) const;

    void set_at(const ssize_t index, const T value);

    void trim();

    void memresize();

    void memresize(const ssize_t new_size_);

    void memresize(const size_t new_size_, const T fill_value);

    size_t size_of() const;

private:
    void update_values_ptr();

    size_t snap_new_size(const size_t size) const;

    viennacl::vector<T> values;
    ssize_t length;
    ssize_t max_length;

    // TODO: Raw ptr to viennacl vector data chunk
    // TODO: Ensure only host memory data pointe d, but not GPU memory...
    T *values_ptr;

    std::mutex vektor_mutex;

}; // vektor



template<class T> using vektor_ptr = std::shared_ptr<vektor<T>>;

template<typename T> std::ostream &operator<<(std::ostream &os, const svr::datamodel::vektor<T> &v);

} // datamodel
} // svr
