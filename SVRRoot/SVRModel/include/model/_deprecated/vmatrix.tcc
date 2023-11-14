#pragma once

//#include <cilk/cilk.h>
#include "vmatrix.hpp"
#include "vektor.tcc"
#include "util/string_utils.hpp"

#define DEFAULT_STEP_SIZE 100

namespace svr {
namespace datamodel {


template<class T>
vmatrix<T>::vmatrix(): m_step_size(DEFAULT_STEP_SIZE)
{}

template<class T>
vmatrix<T>::vmatrix(const ssize_t rows)
: m_step_size(DEFAULT_STEP_SIZE)
{
    values.resize(rows);
}

template<class T>
vmatrix<T>::vmatrix(const double **X, const ssize_t rows, const ssize_t cols)
        : m_step_size(DEFAULT_STEP_SIZE)
{
    values.resize(rows);
    /* cilk_ */ for (decltype(values.size()) row_ix = 0; row_ix < rows; ++row_ix)
        values[row_ix] = std::make_shared<vektor<T>>(vektor<T>(X[row_ix], cols));
}


template<class T>
vmatrix<T>::vmatrix(const ssize_t rows, const ssize_t cols)
        : m_step_size(DEFAULT_STEP_SIZE)
{
    values.resize(rows);
    for (decltype(values.size()) row_ix = 0; row_ix < values.size(); ++row_ix)
        values[row_ix] = std::make_shared<vektor<T>>(vektor<T>(T(0), cols));
}


template<class T>
vmatrix<T>::vmatrix(const vmatrix<T> &m): vmatrix()
{
    for (ssize_t i = 0; i < m.get_length_rows(); ++i) add_row_copy(m.get_row_ref(i));
}

template<class T>
vmatrix<T>::vmatrix(const arma::mat &other) :  vmatrix()
{
    // TOO Parallelize
    values.resize(other.n_rows);
    for(size_t row_ix = 0; row_ix < other.n_rows; ++row_ix){
        values[row_ix] = svr::datamodel::vektor_ptr<T>(new vektor<T>(0., other.n_cols));
        for(size_t col_ix = 0; col_ix < other.n_cols; ++col_ix){
            values[row_ix]->set_at(col_ix, other(row_ix, col_ix));
        }
    }
}

template<class T>
vmatrix<T>::~vmatrix()
{
    clear();
}

template<class T>
bool vmatrix<T>::operator==(vmatrix<T> const &o) const
{
    if (o.values.size() != values.size())
        return false;

    for (size_t i = 0; i < values.size(); ++i)
        if (!(*(values[i]) == *(o.values[i])))
            return false;
    return true;
}

/* operator() as insert operator */
template<class T>
void vmatrix<T>::operator()(const ssize_t index, vektor<T>* V){
    values[index] = std::shared_ptr< vektor<T> >(V);
}

template<class T>
viennacl::matrix<T> vmatrix<T>::vienna_clone() const
{
    viennacl::matrix<T> vcl_matrix(get_length_rows(), get_length_cols());
    copy_to(vcl_matrix);
    return vcl_matrix;
}
#ifdef VIENNACL_WITH_OPENCL
template<class T>
void vmatrix<T>::vienna_clone(viennacl::matrix<T> & vcl_matrix) const
{
    if (vcl_matrix.size1() != size_t(get_length_rows()) ||
        vcl_matrix.size2() != size_t(get_length_cols())) {
        vcl_matrix.resize(get_length_rows(), get_length_cols());
    }

    // Low level copy to vcl matrix.
    for (long i = 0; i < get_length_rows(); i++) {
        datamodel::vektor<T> * row = get_row_ptr(i);
        T * row_ptr = reinterpret_cast<T*>(row->handle().ram_handle().get());

        switch (vcl_matrix.handle().get_active_handle_id()) {
            case viennacl::MAIN_MEMORY :
            {
                T* matrix_ptr = reinterpret_cast<T*>(vcl_matrix.handle().ram_handle().get());
                std::memcpy(&(matrix_ptr[i * vcl_matrix.internal_size2()]), row_ptr, get_length_cols() * sizeof(T));
                break;
            }

            case viennacl::OPENCL_MEMORY :
            {
                viennacl::backend::opencl::memory_write(
                        vcl_matrix.handle().opencl_handle(),
                        i * vcl_matrix.internal_size2() * sizeof(T),
                        get_length_cols() * sizeof(T), row_ptr);
                break;
            }

            case viennacl::MEMORY_NOT_INITIALIZED :
            case viennacl::CUDA_MEMORY :
            default:
                THROW_EX_FS(std::runtime_error, "Not supported memory type.");
        }
    }
}
#endif


template<class T> vmatrix<T> *
vmatrix<T>::clone()
{
    auto p_result_mat = new vmatrix<T>(get_length_rows());
    /* cilk_ */ for (int i = 0; i < get_length_rows(); ++i)
        p_result_mat->set_row_copy_fast(get_row_ref(i), i);
    return p_result_mat;
}

template<class T> ssize_t
vmatrix<T>::get_length_rows() const
{
    return values.size();
}

template<class T> ssize_t
vmatrix<T>::get_length_cols() const
{
    if (values.empty() || !values[0])
        return 0;
    else
        return values[0]->size();
}

// Selection Operations
template<class T>
vektor<T> *vmatrix<T>::get_row_ptr(const int index) const
{
    if (index >= 0 && index < get_length_rows())
        return values[index].get();
    else
        throw std::range_error("Row index out of bounds!");
}

template<class T>
vektor<T> &vmatrix<T>::get_row_ref(const int index) const
{
    if (index >= 0 && index < get_length_rows())
        return *this->values[index];
    else
        throw std::range_error("Row index out of bounds!");
}


template<class T>
vektor<T> *vmatrix<T>::get_row_copy(const int index) const
{
    if (index >= 0 && index < this->get_length_rows())
        return values[index]->clone();
    else
        throw std::range_error(
                "It's impossible to get an row from the matrix that doesn't exist.");
}

template<class T>vektor<T> *
vmatrix<T>::get_col_copy(const int index) const
{
    if (index >= 0 && index < get_length_cols()) {
        auto V = new vektor<T>();
        for (int i = 0; i < get_length_rows(); i++)
            V->add(values[i]->get_value(index));
        return V;
    } else 
        throw std::range_error(
                "It's impossible to get an row from the matrix that doesn't exist.");
}

template<class T> T
vmatrix<T>::get_value(const size_t row_index, const size_t col_index) const
{
#ifndef NDEBUG
    if (row_index >= decltype(row_index)(values.size()) || col_index >= decltype(col_index)(get_length_cols()))
        throw std::range_error(
                svr::common::formatter() << "Get value row index " << row_index << " row count " << values.size() <<
                                         " col index " << col_index << " col count " << values[0]->size());
#endif
    return values[row_index]->get_value(col_index);
}

template<class T> void
vmatrix<T>::set_value(const size_t row_index, const size_t col_index, const T value)
{
#ifndef NDEBUG
    if (row_index >= decltype(row_index)(values.size()) || col_index >= decltype(col_index)(get_length_cols()))
        throw std::range_error(
                svr::common::formatter() << "Set value row index " << row_index << " row count " << values.size() <<
                                         " col index " << col_index << " col count " << values[0]->size());
#endif
    values[row_index]->set_at(col_index, value);
}

template<class T> ssize_t
vmatrix<T>::index_of(const vektor<T> *V) const
{
    for (ssize_t i = 0; i < get_length_rows(); i++) {
        bool found = true;
        for (ssize_t j = 0; j < get_length_cols(); j++) {
            if (V->get_value(j) != this->get_value(i, j)) {
                found = false;
                break;
            }
        }
        if (found) return i;
    }
    return -1;
}

template<class T> ssize_t
vmatrix<T>::index_of(const vektor<T> &V) const
{
    for (ssize_t i = 0; i < get_length_rows(); i++) {
        bool found = true;
        for (ssize_t j = 0; j < get_length_cols(); j++) {
            if (V[j] != get_value(i, j)) {
                found = false;
                break;
            }
        }
        if (found) return i;
    }
    return -1;
}

// Add/Remove Operations
template<class T> void
vmatrix<T>::clear()
{
    values.clear();
}

template<class T> bool
vmatrix<T>::empty() const
{
    return values.empty();
}

template<class T>
void vmatrix<T>::add_row_ref(vektor<T> *V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V->size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V));
    } else
        THROW_EX_FS(std::invalid_argument,
                "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}

template<class T>
void vmatrix<T>::add_row_ref_safe(vektor<T> *V)
{
    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V->size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V));
    } else
        THROW_EX_FS(std::invalid_argument,
                "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::set_row_ref_fast(vektor<T> *p_vec, size_t i)
{
    values[i] = std::shared_ptr<vektor<T>>(p_vec);
}


template<class T>
void vmatrix<T>::set_row_copy_fast(const vektor<T> &vec, size_t i)
{
    values[i] = std::shared_ptr<vektor<T>>(vec.clone());
}


template<class T>
void vmatrix<T>::add_row_copy(const vektor<T> *V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V->size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V->clone()));
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::add_row_copy(const vektor <T> &V) {
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V.size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V.clone()));
    } else
        THROW_EX_FS(std::invalid_argument,
                    "It's impossible to add a row of different length " << get_length_cols() << " to " << V.size());
}


template<class T>
void vmatrix<T>::add_row_copy(const arma::Row<T> &V) {
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == (ssize_t) V.size()) {
        values.push_back(std::make_shared<svr::datamodel::vektor<T>>(V));
    } else
        THROW_EX_FS(std::invalid_argument,
                    "It's impossible to add a row of different length " << get_length_cols() << " to " << V.size());
}


template<class T>
void vmatrix<T>::add_row_copy(const T *V, int N) {
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == N) {
        values.push_back(std::make_shared<vektor<T> >(V, N));
    } else
        THROW_EX_FS(std::invalid_argument,
                    "It's impossible to add a row of different length " << get_length_cols() << " to "
                                                                        << V->size());
}


template<class T>
void vmatrix<T>::add_row_ref(T *V, int N) /* TODO Convert to shared_ptr */
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == N) {
        values.push_back(std::shared_ptr<vektor<T> >(V, N));
    } else
        THROW_EX_FS(std::invalid_argument,
                    "It's impossible to add a row of different length " << get_length_cols() << " to "
                                                                        << V->size());
}


template<class T>
void vmatrix<T>::add_row_ref_at(vektor <T> *V, int index) {
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(V);
    } else if (get_length_cols() == V->size() && index >= 0 && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), V);
    } else
        THROW_EX_FS(std::invalid_argument,
                    "It's impossible to add a row of different length " << get_length_cols() << " to "
                                                                        << V->size());
}


template<class T>
void vmatrix<T>::add_row_copy_at(vektor<T> *V, int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(V->clone());
    } else if (get_length_cols() == V->size() && index >= 0 && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), V->clone());
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_row_copy_at(T *V, int N, int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(new vektor<T>(V, N));
    } else if (get_length_cols() == V->length && index >= 0 && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), new vektor<T>(V, N));
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a column of different length or in a bad index.");
}

template<class T>
void vmatrix<T>::add_col_copy(vektor<T> *V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0)
    {
        values.resize(V->size());
        for (int i = 0; i < V->size(); ++i)
        {
            vektor<T> *V3 = new vektor<T>();
            V3->add(V->get_value(i));
            values[i] = std::shared_ptr<vektor<T> >(V3);
        }
    }
    else if (get_length_rows() == V->size())
    {
        for (int i = 0; i < V->size(); ++i)
            values[i]->add(V->get_value(i));
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a column of different length.");
}

template<class T>
void vmatrix<T>::add_col_copy_safe(vektor<T> *V)
{
    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if (get_length_rows() == 0 || get_length_cols() == 0)
    {
        for (int i = 0; i < V->size(); i++)
        {
            vektor<T> *V3 = new vektor<T>();
            V3->add(V->get_value(i));
            values.push_back(std::shared_ptr<vektor<T> >(V3));
        }
    }
    else if (get_length_rows() == V->size())
    {
        for (int i = 0; i < V->size(); ++i) values[i]->add(V->get_value(i));
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a column of different length.");
}

template<class T>
void vmatrix<T>::add_col_copy(const vektor<T> &V)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if (get_length_rows() == 0 || get_length_cols() == 0)
    {
        for (int i = 0; i < V.size(); i++)
        {
            vektor<T> *V3 = new vektor<T>();
            V3->add(V.get_value(i));
            values.push_back(std::shared_ptr<vektor<T> >(V3));
        }
    }
    else if (get_length_rows() == V.size())
    {
        for (int i = 0; i < V.size(); ++i)
            values[i]->add(V.get_value(i));
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a column of different length.");
}

template<class T>
void vmatrix<T>::add_col_copy(T *V, int N)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if (get_length_rows() == 0 || get_length_cols() == 0) {
        for (int i = 0; i < N; i++) {
            vektor<T> *V3 = new vektor<T>();
            V3->add(V[i]);
            values.push_back(std::shared_ptr<vektor<T> >(V3));
        }
    } else if (get_length_rows() == N) {
        for (int i = 0; i < N; ++i)
            values[i]->add(V[i]);
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a column of different length.");
}

template<class T>
void vmatrix<T>::add_col_copy_at(vektor<T> *V, int index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    } else if (get_length_rows() == V->size() && index >= 0 && index < get_length_cols()) {
        /* cilk_ */ for (int i = 0; i < V->size(); ++i) {
            values[i]->resize(V->size());
            values[i]->set_at(V->get_value(i), index);
        }
    } else
        THROW_EX_FS(std::invalid_argument, "It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_col_copy_at(const vektor<T> &V, const int index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    } else if (get_length_rows() == V.size() && index >= 0 && index < get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); ++i)
            values[i]->set_at(index, V.get_value(i));
    }
    else if (index > get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) {
            values[i]->resize(index + 1);
            values[i]->set_at(index, V.get_value(i));
        }
    } else
        THROW_EX_FS(std::invalid_argument,"It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_col_copy_at(T *V, const ssize_t N, const ssize_t index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V, N);
    } else if (get_length_rows() == N && index >= 0 && index < get_length_cols()) {
        /* cilk_ */ for (int i = 0; i < N; ++i)
            values[i]->set_at(V->get_value(i), index);
    } else
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::set_col_copy_at(const vektor<T> &V, const int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    }
    else if (get_length_rows() == V.size() && index >= 0 && index < get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) values[i]->set_at(index, V.get_value(i));
    }
    else if (index > get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) {
            values[i]->resize(index + 1);
            values[i]->set_at(index, V.get_value(i));
        }
    }
    else {
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to add a row of different length or in a bad index.");
    }
}


template<class T>
void vmatrix<T>::set_col_copy_at_safe(const vektor<T> &V, const int index)
{
    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    } else if (get_length_rows() == V.size() && index >= 0 && index < get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) values[i]->set_at(index, V[i]);
    } else if (index >= get_length_cols()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) {
            values[i]->resize(index + 1);
            values[i]->set_at(index, V[i]);
        }
    } else {
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to add a row of different length or in a bad index.");
    }
}


template<class T>
void vmatrix<T>::remove_row(const ssize_t index)
{
    if (index >= 0 && index < get_length_rows())
        values.erase(std::next(values.begin(), index), std::next(values.begin(), index + 1));
    else
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to remove an element from the matrix that doesn't exist.");
}

template<class T>
void vmatrix<T>::remove_rows(const ssize_t start_index, const ssize_t end_index)
{
    if (start_index < 0 || start_index >= get_length_rows() || end_index < 0 || end_index >= get_length_rows() || start_index > end_index)
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to remove an element from the matrix that doesn't exist.");
    values.erase(std::next(values.begin(), start_index), std::next(values.begin(), end_index));
}

template<class T>
void vmatrix<T>::remove_col(const ssize_t index)
{
    if (index >= 0 && index < get_length_cols())
        /* cilk_ */ for (int i = 0; i < get_length_rows(); ++i)
            values[i]->remove_at(index);
    else
        THROW_EX_FS(std::invalid_argument, 
                "It's impossible to remove an element from the matrix that doesn't exist.");
}

template<class T>
void vmatrix<T>::reserve(size_t sz)
{
    values.reserve(sz);
}

template<class T>
void vmatrix<T>::resize(const size_t new_size, T value, const ssize_t row_size)
{
    if (!values.empty() && row_size != values[0]->size()) clear();
    const auto sz = values.size();
    values.resize(new_size);
    if (sz < new_size) /* cilk_ */ for(size_t i = sz; i < new_size; ++i)
            values[i] = svr::datamodel::vektor_ptr<T>(new vektor<T>(value, row_size));
}

template<class T> vmatrix<T>
vmatrix<T>::extract_rows(const int from_row_index, const int to_row_index) const
{
    if (from_row_index >= 0 && to_row_index < get_length_rows() && from_row_index <= to_row_index) {
        vmatrix<T> M(1 + to_row_index - from_row_index);
        /* cilk_ */ for (auto i = from_row_index; i <= to_row_index; ++i)
            M.set_row_ref_fast(get_row_copy(i), i - from_row_index);
        return M;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to extract the rows, invalid indexes.");
    return {};
}

template<class T>
vmatrix <T> vmatrix<T>::extract_cols(const int from_col_index, const int to_col_index) const
{
    if (from_col_index >= 0 && to_col_index < get_length_cols() && from_col_index <= to_col_index) {
        vmatrix<T> M(get_length_rows());
        /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i)
            M.set_row_ref_fast(get_row_ptr(i)->extract(from_col_index, to_col_index), i);
        return M;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to extract the columns, invalid indexes");
    return {};
}

// Pre-built Matrix
template<class T> vmatrix<T> *
vmatrix<T>::zero_matrix(const ssize_t rows_number, const ssize_t cols_number)
{
    return single_value_matrix(rows_number, cols_number, T());
}

template<class T>vmatrix<T> *
vmatrix<T>::single_value_matrix(const ssize_t rows_number, const ssize_t cols_number, T const &value)
{
    vmatrix<T> *M = new vmatrix<T>();
    for (int i = 0; i < rows_number; i++) {
        vektor<T> *V = new vektor<T>(value, cols_number);
        M->add_row_ref(V);
    }
    return M;
}


template<class T>vmatrix<double> *
vmatrix<T>::rand_matrix(const ssize_t rows_number, const ssize_t cols_number)
{
    vmatrix<double> *M = new vmatrix<double>();
    for (int i = 0; i < rows_number; i++) {
        vektor<double> *V = vektor<T>::rand_vector(cols_number);
        M->add_row_ref(V);
    }
    return M;
}

// Mathematical Operations
template<class T> void
vmatrix<T>::sum_scalar(T X)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->sum_scalar(X);
}

template<class T> void
vmatrix<T>::product_scalar(T X)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->product_scalar(X);
}

template<class T> void
vmatrix<T>::divide_scalar(T X)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->divide_scalar(X);
}

template<class T> void
vmatrix<T>::pow_scalar(T X)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->pow_scalar(X);
}

template<class T>
void vmatrix<T>::sum_matrix(vmatrix<T> *M)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->sum_vector(M->values[i].get());
}

template<class T>
void vmatrix<T>::sum_matrix(const vmatrix<T> &M)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->sum_vector(M.values[i].get());
}

template<class T>
void vmatrix<T>::subtract_matrix(vmatrix<T> *M)
{
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i) values[i]->subtract_vector(M->values[i].get());
}

template<class T>
vektor<T> *vmatrix<T>::product_vector(const vektor<T> *V) const
{
    const auto rows = get_length_rows(), cols = get_length_cols();
    if (cols == 0 || cols == V->size())
    {
        auto V2 = new vektor<T>((T)0., rows);
        /* cilk_ */ for (ssize_t i = 0; i < rows; ++i) V2->set_at(i, V->product_vector_scalar(values[i].get(), V));
        return V2;
    }
    THROW_EX_FS(std::invalid_argument,
                "It's impossible to multiply a matrix and a vector with different length: " << cols << ", " << V->size());
    return nullptr;
}


template<class T> vektor<T>
vmatrix<T>::product_vector(const vektor<T> &V) const
{
    auto const rows = get_length_rows(), cols = get_length_cols();
    if (cols == 0 || cols == V.size())
    {
        vektor<T> V2((T)0., rows);
	    /* cilk_ */ for (ssize_t i = 0; i < rows; i++) V2.set_at(i, V.product_vector_scalar(*values[i], V));
        return V2;
    }
    THROW_EX_FS(std::invalid_argument,
                "It's impossible to multiply a matrix and a vector with different length " << cols << ", " << V.size());
    return {};
}

template<class T>
vektor<T> *vmatrix<T>::product_vector(vmatrix const *M, vektor<T> const *V)
{
    if (M->get_length_cols() == 0 || M->get_length_cols() == V->size()) {
        auto V2 = new vektor<T>((T)0., M->get_length_rows());
        /* cilk_ */ for (ssize_t i = 0; i < M->get_length_rows(); i++)
            V2->set_at(i, V->product_vector_scalar(M->values[i].get(), V));
        return V2;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to product a matrix and a vector with different length.");
    return nullptr;
}

template<class T>
vmatrix<T> *vmatrix<T>::product_vector_vector(vektor<T> const *V1, vektor<T> const *V2)
{
    if (V1->size() == V2->size()) {
        auto p_result_matrix = new vmatrix<T>(V1->size());
        /* cilk_ */ for (ssize_t i = 0; i < V1->size(); ++i) {
            auto V4 = V2->clone();
            V4->product_scalar(V1->get_value(i));
            p_result_matrix->set_row_ref_fast(V4, i);
        }
        return p_result_matrix;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to product two vectors with different length.");
    return nullptr;
}

template<class T>
vmatrix<T> vmatrix<T>::product_vector_vector(vektor<T> const &V1, vektor<T> const &V2)
{
    if (V1.size() == V2.size()) {
        vmatrix<T> result_mtx(V1.size());
        /* cilk_ */ for (ssize_t i = 0; i < V1.size(); ++i) {
            auto V4 = V2.clone();
	        V4->product_scalar(V1[i]);
            result_mtx.set_row_ref_fast(V4, i);
        }
        return result_mtx;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to product two vectors with different length.");
    return {};
}


template<class T>
vmatrix<T> *vmatrix<T>::product_matrix_matrix(vmatrix<T> const *M1, vmatrix<T> const *M2)
{
    if (M1->get_length_cols() == M2->get_length_cols()) {
        vmatrix<T> *M3 = new vmatrix<T>(M1->get_length_rows(), M2->get_length_rows());
        /* cilk_ */ for (int i = 0; i < M1->get_length_rows(); ++i) {
            vektor<T> *V = new vektor<T>(T(0), M2->get_length_rows());
            /* cilk_ */ for (int j = 0; j < M2->get_length_rows(); ++j)
                V->set_at(j, V->product_vector_scalar(M1->values[i].get(), M2->values[i].get()));
            M3->set_row_ref_fast(V, i);
        }
        return M3;
    }
    THROW_EX_FS(std::invalid_argument, "It's impossible to product matrices with different dimensions.");
    return nullptr;
}

// I/O Operations
template<class T>
vmatrix<T> *vmatrix<T>::load(char const *filename)
{
    // Open the file
    std::ifstream file(filename, std::ios::in);
    if (!file) {
        LOG4_ERROR("Error. It's impossible to open the file.");
        return new vmatrix<double>();
    }
    auto M = new vmatrix<double>();
    // Save the vector
    try {
        int rows_number, cols_number;
        double value;
        file >> rows_number >> cols_number;
        for (int i = 0; i < rows_number; i++) {
            auto V = new vektor<double>(cols_number);
            for (int j = 0; j < cols_number; j++) {
                file >> value;
                V->add(value);
            }
            M->add_row_ref(V);
        }
    } catch (...) {
        LOG4_ERROR("It's impossible to complete the load.");
    }
    // Close the file
    file.close();

    return M;
}


template<class T>
vmatrix<T> vmatrix<T>::load(const std::string &filename)
{
    vmatrix<double> M;
    // Open the file
    std::ifstream file(filename, std::ios::in);
    if (!file) {
        LOG4_ERROR("It's impossible to open the file.");
        return M;
    }

    // Save the vector
    try {
        int rows_number, cols_number;
        double value;
        file >> rows_number >> cols_number;
        for (int i = 0; i < rows_number; i++) {
            vektor<double> *V = new vektor<double>(cols_number);
            for (int j = 0; j < cols_number; j++) {
                file >> value;
                V->add(value);
            }
            M.add_row_ref(V);
        }
    } catch (...) {
        LOG4_ERROR("It's impossible to complete the load.");
    }
    // Close the file
    file.close();

    return M;
}

template<class T>
void vmatrix<T>::save(char const *filename, const vmatrix<T> &v)
{
    // Open the file
    std::ofstream file(filename, std::ios::out);
    if (!file) {
        LOG4_ERROR("Error. It's impossible to create the file.");
        return;
    }
    file.precision(std::numeric_limits<double>::max_digits10);
    // Save the matrix
    try {
        file << v.get_length_rows() << " " << v.get_length_cols() << std::endl;
        for (int i = 0; i < v.get_length_rows(); ++i) {
            for (int j = 0; j < v.get_length_cols(); ++j)
                file << v.values[i]->get_value(j) << " ";
            file << std::endl;
        }
    } catch (...) {
        LOG4_ERROR("Error. It's impossible to complete the save.");
    }
    // Close the file
    file.close();
}


template<class T>
void vmatrix<T>::save(char const *filename) const
{
    // Open the file
    std::ofstream file(filename, std::ios::out);
    if (!file) {
        LOG4_ERROR("Error. It's impossible to create the file.");
        return;
    }
    file.precision(std::numeric_limits<double>::max_digits10);
    // Save the matrix
    try {
        file << this->get_length_rows() << " " << this->get_length_cols() << std::endl;
        for (int i = 0; i < this->get_length_rows(); i++) {
            for (int j = 0; j < this->get_length_cols(); j++)
                file << this->values[i]->get_value(j) << " ";
            file << std::endl;
        }
    } catch (...) {
        LOG4_ERROR("Error. It's impossible to complete the save.");
    }
    // Close the file
    file.close();
}

template<class T> template<class Char>
void vmatrix<T>::print(std::basic_ostream<Char> & str) const
{
    str << "[" << get_length_rows() << "," << get_length_cols()<< "](";

    for(ssize_t i = 0; i < get_length_rows(); ++i)
    {
        str << "(";
        for(ssize_t j = 0; j < get_length_cols(); ++j)
            str << get_value(i, j) << (j == get_length_cols() -1 ? "" : ",");
        str << ")" << (i == get_length_rows() -1 ? "" : ",");
    }
    str << ")";
}

template<class T> template<class Char>
void vmatrix<T>::print(const std::string &matrix_name, std::basic_ostream<Char> & str) const
{
    str <<  svr::common::demangle(typeid(*this).name()) << " with name: " << matrix_name << "\n";
    print(str);
}

// Operators Redefinition
template<class T>const vektor<T> &
vmatrix<T>::operator[](const size_t index) const
{
#ifndef NDEBUG
    if (index >= decltype(index)(values.size()))
        throw std::range_error(svr::common::formatter() << "Row index " << index << " row count " << values.size());
#endif
    return *(values[index]);
}

template<class T>vektor<T> &
vmatrix<T>::operator[](const size_t index)
{
#ifndef NDEBUG
    if (index >= decltype(index)(values.size()))
        throw std::range_error(svr::common::formatter() << "Row index " << index << " row count " << values.size());
#endif
    return *(values[index]);
}

template<typename T> const T &
vmatrix<T>::operator()(const size_t row_index, const size_t col_index) const
{
#ifndef NDEBUG
    if (row_index >= decltype(row_index)(values.size()) || col_index >= (decltype(col_index)) get_length_cols())
        throw std::range_error(svr::common::formatter() << "Row index " << row_index << " row count " <<
                                                        values.size() << " column index " << col_index << " column count " << get_length_cols());
#endif
    return (*values[row_index])[col_index];
}

template<typename T> T &
vmatrix<T>::operator()(const size_t row_index, const size_t col_index)
{
#ifndef NDEBUG
    if (row_index >= decltype(row_index)(values.size()) || col_index >= (decltype(col_index)) get_length_cols())
        throw std::range_error(svr::common::formatter() << "Row index " << row_index << " row count " <<
                                                        values.size() << " column index " << col_index << " column count " << get_length_cols());
#endif
    return (*values[row_index])[col_index];
}


template<class T>
size_t vmatrix<T>::size_of() const
{
    size_t result = sizeof(vmatrix<T>);
    for (auto const &vec: values) result += vec->size_of();
    return result;
}


template<typename T>
arma::mat vmatrix_to_admat(const vmatrix<T> &input)
{
    arma::mat output(input.get_length_rows(), input.get_length_cols());
    /* cilk_ */ for (decltype(input.get_length_rows()) row_ix = 0; row_ix < input.get_length_rows(); ++row_ix)
        /* cilk_ */ for (decltype(input.get_length_cols()) col_ix = 0; col_ix < input.get_length_cols(); ++col_ix)
            output(row_ix, col_ix) = input(row_ix, col_ix);
    return output;
}


template<typename T>
arma::mat vmatrix_to_admat_fast(const vmatrix<T> &input)
{
    arma::mat output(input.get_length_rows(), input.get_length_cols());
    /* cilk_ */ for (decltype(input.get_length_rows()) row_ix = 0; row_ix < input.get_length_rows(); ++row_ix) {
        arma::rowvec row_i(input.get_row_ref(row_ix).to_vector());
        output.row(row_ix) = row_i;
    }

    return output;
}

template<typename T>
arma::mat vmatrix_to_admat_col(const vmatrix<T> &input)
{
    arma::mat output(input.get_length_cols(), input.get_length_rows());
    /* cilk_ */ for (decltype(input.get_length_rows()) row_ix = 0; row_ix < input.get_length_rows(); ++row_ix) {
        arma::colvec col_i(input.get_row_ref(row_ix).to_vector());
        output.col(row_ix) = col_i;
    }

    return output.t();
}



}
}
