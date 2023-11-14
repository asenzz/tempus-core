/******************************************************************************
 *                       ONLINE SUPPORT VECTOR REGRESSION                      *
 *                      Copyright 2006 - Francesco Parrella                    *
 *                                                                             *
 *This program is distributed under the terms of the GNU General Public License*
 ******************************************************************************/

#pragma once
// TODO Replace references to model::vmatrix with arma::mat
#include <iostream>
#include <mutex>
#include "vektor.tcc"

namespace svr {
namespace datamodel {

template<class T>
class vmatrix
{
public:

    // Initialization
    vmatrix();

    vmatrix(const ssize_t rows);

    vmatrix(const ssize_t rows, const ssize_t cols);

    vmatrix(const double **X, const ssize_t rows, const ssize_t cols);

    vmatrix(const vmatrix<T> &m);

    vmatrix(const arma::mat &other);

    ~vmatrix();

    bool operator==(vmatrix<T> const &) const;

    void operator()(const ssize_t index, vektor<T>* V);
    
    viennacl::matrix<T> vienna_clone() const;

    void vienna_clone(viennacl::matrix<T> & vcl_matrix) const;

    void copy_to(viennacl::matrix<T> &vcl_matrix) const
    {
        // TODO Improve...

        if (vcl_matrix.size1() != size_t(get_length_rows()) ||
            vcl_matrix.size2() != size_t(get_length_cols())) {
            //vcl_matrix.resize(GetLengthRows(), GetLengthCols(), false);
            throw std::runtime_error("vcl matrix dimensions mismatch");
        }
        std::vector<std::vector<T>> std_x(get_length_rows());
        /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); i++) get_row_ref(i).copy_to(std_x[i]);
        viennacl::copy(std_x, vcl_matrix);
    }

    void copy_from(const viennacl::matrix<T> &vcl_matrix)
    {
        clear();

        // Unless copied from a std matrix the implementation, as it was,
        // is unsafe in the case of a GPU based source matrix, because of ViennaCL shortcomings
        std::vector<std::vector<double>> host_kernel_matrix(vcl_matrix.size1());
        /* cilk_ */ for (size_t i = 0; i < vcl_matrix.size1(); ++i)
            host_kernel_matrix[i].resize(vcl_matrix.size2());
        viennacl::copy(vcl_matrix, host_kernel_matrix);

        values.resize(vcl_matrix.size1());
        /* cilk_ */ for (size_t i = 0; i < vcl_matrix.size1(); ++i)
            set_row_ref_fast(new vektor<T>(host_kernel_matrix[i]), i);
    }


    vmatrix<T> *clone();

    bool empty() const;

    ssize_t get_length_rows() const;

    ssize_t get_length_cols() const;

    // Selection Operations
    vektor <T> *get_row_ptr(const int index) const;

    vektor <T> &get_row_ref(const int index) const;

    vektor <T> *get_row_copy(const int index) const;

    vektor <T> *get_col_copy(const int index) const;

    T get_value(const size_t row_index, const size_t col_index) const;

    void set_value(const size_t row_index, const size_t col_index, const T value);

    ssize_t index_of(const vektor <T> *V) const;

    ssize_t index_of(const vektor <T> &V) const;

    // Add/Remove Operations
    void clear();

    void add_row_ref(T *V, int N);

    void add_row_ref(vektor <T> *V);

    void add_row_ref_safe(vektor <T> *V); // thread safe

    //No length check
    void set_row_ref_fast(vektor<T> *p_vec, const size_t i);

    void set_row_copy_fast(const vektor<T> &vec, const size_t i);

    void add_row_ref_at(vektor <T> *V, int index);

    void add_row_copy(const vektor <T> *V);

    void add_row_copy(const vektor <T> &V);

    void add_row_copy(const arma::Row <T> &V);

    void add_row_copy(const T *V, int N);

    void add_row_copy_at(vektor <T> *V, int index);

    void add_row_copy_at(T *V, int N, int index);

    void add_col_copy(vektor <T> *V);

    void add_col_copy_safe(vektor <T> *V);

    void add_col_copy(const vektor <T> &V);

    void add_col_copy(T *V, int N);

    void add_col_copy_at(vektor <T> *V, int index);

    void add_col_copy_at(const vektor <T> &V, const int index);

    void add_col_copy_at(T *V, const ssize_t N, const ssize_t index);

    void set_col_copy_at(const vektor<T> &V, const int index);

    void set_col_copy_at_safe(const vektor<T> &V, const int index);

    void remove_row(const ssize_t index);

    void remove_rows(const ssize_t start_index, const ssize_t end_index);

    void remove_col(const ssize_t index);

    void reserve(size_t sz);

    void resize(const size_t new_size, T value, const ssize_t row_size);

    vmatrix <T> extract_rows(const int from_row_index, const int to_row_index) const;

    vmatrix <T> extract_cols(const int from_col_index, const int to_col_index) const;

    // Pre-built Matrix
    static vmatrix<T> *zero_matrix(const ssize_t rows_number, const ssize_t cols_number);

    static vmatrix<T> *single_value_matrix(const ssize_t rows_number, const ssize_t cols_number, T const &value);

    static vmatrix<double> *rand_matrix(const ssize_t rows_number, const ssize_t cols_number);

    // Mathematical Operations
    void sum_scalar(T X);

    void product_scalar(T X);

    void divide_scalar(T X);

    void pow_scalar(T X);

    void sum_matrix(vmatrix<T> *M);

    void sum_matrix(const vmatrix<T> &M);

    void subtract_matrix(vmatrix<T> *M);

    vektor<T> *product_vector(const vektor <T> *V) const;

    vektor<T> product_vector(const vektor<T> &V) const;

    static vektor <T> *product_vector(vmatrix const *M, vektor <T> const *V);

    static vmatrix<T> *product_vector_vector(vektor <T>const  *V1, vektor <T> const *V2);

    static vmatrix<T> product_vector_vector(vektor <T> const &V1, vektor <T> const &V2);

    static vmatrix<T> *product_matrix_matrix(vmatrix<T> const *M1, vmatrix<T> const *M2);

    // I/O Operations
    static vmatrix<T> *load(char const *filename);

    static vmatrix<T> load(const std::string &filename);

    static void save(char const *filename, const vmatrix<T> &v);

    void save(char const *filename) const;

    template<class Char>
    void print(std::basic_ostream<Char> &) const;

    template<class Char>
    void print(const std::string &matrix_name, std::basic_ostream<Char> &) const;

    // Operators redefinition
    const vektor <T> & operator[](const size_t index) const;

    vektor <T> & operator[](const size_t index);

    const T &operator () (const size_t row_index, const size_t col_index) const;

    T &operator () (const size_t row_index, const size_t col_index);

    size_t size_of() const;

    //operator=
//    vmatrix<T>* operator=(vmatrix<T>& vmatrix){
//    
//        return 
//    }
    
private:
    ssize_t m_step_size;
    std::vector<svr::datamodel::vektor_ptr<T>> values;

    std::mutex vmatrix_mutex;

    void resize();
    void resize(size_t new_row_number, size_t new_col_number);
};

template<class T, class Char>
std::basic_ostream<Char> & operator<<(std::basic_ostream<Char> & str, vmatrix<T> const & m)
{
    m.print(str);
    return str;
}


template<typename T> arma::mat vmatrix_to_admat(const vmatrix<T> &input);
template<typename T> arma::mat vmatrix_to_admat_fast(const vmatrix<T> &input);
template<typename T> arma::mat vmatrix_to_admat_col(const vmatrix<T> &input);


}
}
