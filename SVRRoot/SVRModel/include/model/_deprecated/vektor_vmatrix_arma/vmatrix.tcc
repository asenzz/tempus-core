#pragma once

//#include <cilk/cilk.h>
#include "model/_deprecated/vmatrix.tcc"
#include "model/_deprecated/vektor.tcc"
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
    for(decltype(values.size()) row_ix = 0; row_ix < values.size(); ++row_ix)
        values[row_ix] = std::make_shared<vektor<T>>(vektor<T>(std::numeric_limits<double>::quiet_NaN(), cols));
}


template<class T>
vmatrix<T>::vmatrix(const vmatrix<T> &m): vmatrix()
{
    for (ssize_t i = 0; i < m.get_length_rows(); ++i) add_row_copy(m.get_row_ref(i));
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
arma::mat<T> vmatrix<T>::arma_clone() const
{
    arma::mat<T> arma_matrix(get_length_rows(), get_length_cols());
    copy_to(arma_matrix);
    return arma_matrix;
}

template<class T>
void vmatrix<T>::arma_clone(arma::mat<T> & arma_matrix) const
{
    if (arma_matrix.size(0) != size_t(get_length_rows()) ||
        arma_matrix.size(1) != size_t(get_length_cols())) {
        arma_matrix.resize(get_length_rows(), get_length_cols());
    }

    /*
    std::vector<std::vector<T>> std_x(GetLengthRows());
    for (long i = 0; i < GetLengthRows(); i++) {
        datamodel::Vector<T> * row = GetRowRef(i);
        row->copy_to(std_x[i]);
    }
    viennacl::copy(std_x, vcl_matrix);
    */

    // Low level copy to vcl matrix.
    for (long i = 0; i < get_length_rows(); i++) {
        datamodel::vektor<T> * row = get_row_ptr(i);
        T * row_ptr = reinterpret_cast<T*>(row->handle());
	//only armadillo in main memory is supported
        T* matrix_ptr = reinterpret_cast<T*>(arma_matrix.memptr());
        std::memcpy(&(matrix_ptr[i * arma_matrix.size(0)]), row_ptr, get_length_rows() * sizeof(T));
    }
}

template<class T>
void vmatrix<T>::copy_to(arma::mat<T> &arma_matrix) const
{
    // TODO Improve...

    if (arma_matrix.size(0) != size_t(get_length_rows()) ||
        arma_matrix.size(1) != size_t(get_length_cols())) {
        //vcl_matrix.resize(GetLengthRows(), GetLengthCols(), false);
        throw std::runtime_error("vcl matrix dimensions mismatch");
    }
    std::vector<std::vector<T>> std_x(get_length_rows());
    /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); i++) get_row_ref(i).copy_to(std_x[i]);
    arma::copy(std_x, arma_matrix);
}

template<class T>
void vmatrix<T>::copy_from(const viennacl::matrix<T> &vcl_matrix)
{
    clear();

    // TODO: improve

//    for (size_t i = 0; i < vcl_matrix.size1(); i++) {
//        vektor<double> row(vcl_matrix.size2());
//        for (size_t j = 0; j < vcl_matrix.size2(); j++) {
//            row.Add(vcl_matrix(i, j));
//        }
//        AddRowCopy(row);
//    }

    reserve(arma_matrix.size(0));
    for (size_t i = 0; i < arma_matrix.size(0); ++i)
        add_row_ref(new vektor<T>(arma_matrix.row(i), arma_matrix.size(1)));
}

template<class T> vmatrix<T> *
vmatrix<T>::clone()
{
    auto p_result_mat = new vmatrix<T>(get_length_rows());
    /* cilk_ */ for (int i = 0; i < get_length_rows(); ++i)
        p_result_mat->set_row_copy_fast(get_row_ref(i), i);
    return p_result_mat;
}

template<class T> bool
vmatrix<T>::empty() const
{
    return values.empty();
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
    values[row_index]->set_value(col_index, value);
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
template<class T>
void vmatrix<T>::clear()
{
    values.clear();
}

template<class T>
void vmatrix<T>::add_row_ref(vektor<T> *V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V->size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V));
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}

template<class T>
void vmatrix<T>::add_row_ref_safe(vektor<T> *V)
{
    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V->size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V));
    } else
        throw std::invalid_argument(svr::common::formatter() << 
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
        throw std::invalid_argument(svr::common::formatter() << "It's impossible to add a row of different length " <<
                                                             get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::add_row_copy(const vektor<T> &V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == V.size()) {
        values.push_back(std::shared_ptr<vektor<T> >(V.clone()));
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                                                             "It's impossible to add a row of different length " << get_length_cols() << " to " << V.size());
}


template<class T>
void vmatrix<T>::add_row_copy(const T *V, int N)
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == N) {
        vektor<T> *new_v = new vektor<T>(V, N);
        values.push_back(std::shared_ptr<vektor<T> >(new_v));
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                                                             "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::add_row_ref(T *V, int N) /* TODO Convert to shared_ptr */
{
    if (get_length_rows() == 0 || get_length_cols() == 0 || get_length_cols() == N) {
        values.push_back(std::shared_ptr<vektor<T> >(V, N));
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                                                             "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::add_row_ref_at(vektor<T> *V, int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(V);
    } else if (get_length_cols() == V->size() && index >= 0  && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), V);
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                                                             "It's impossible to add a row of different length " << get_length_cols() << " to " << V->size());
}


template<class T>
void vmatrix<T>::add_row_copy_at(vektor<T> *V, int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(V->clone());
    } else if (get_length_cols() == V->size() && index >= 0 && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), V->clone());
    } else
        throw std::invalid_argument("It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_row_copy_at(T *V, int N, int index)
{
    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        values.push_back(new vektor<T>(V, N));
    } else if (get_length_cols() == V->length && index >= 0 && index <= get_length_rows()) {
        values.insert(std::next(values.begin(), index), new vektor<T>(V, N));
    } else
        throw std::invalid_argument("It's impossible to add a row of different length or in a bad index.");
}

template<class T>
void vmatrix<T>::add_col_copy(vektor<T> *V)
{
    if (get_length_rows() == 0 || get_length_cols() == 0)
    {
        values.resize(V->size());
        /* cilk_ */ for (int i = 0; i < V->size(); ++i)
        {
            vektor<T> *V3 = new vektor<T>();
            V3->add(V->get_value(i));
            values[i] = std::shared_ptr<vektor<T> >(V3);
        }
    }
    else if (get_length_rows() == V->size())
    {
        for (int i = 0; i < V->size(); i++)
            values[i]->add(V->get_value(i));
    } else
        throw std::invalid_argument("It's impossible to add a column of different length.");
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
        for (int i = 0; i < V->size(); i++) values[i]->add(V->get_value(i));
    } else
        throw std::invalid_argument(
                "It's impossible to add a column of different length.");
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
        for (int i = 0; i < V.size(); i++)
            values[i]->add(V.get_value(i));
    } else
        throw std::invalid_argument(
                "It's impossible to add a column of different length.");
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
        for (int i = 0; i < N; i++) {
            values[i]->add(V[i]);
        }
    } else
        throw std::invalid_argument(
                "It's impossible to add a column of different length.");
}

template<class T>
void vmatrix<T>::add_col_copy_at(vektor<T> *V, int index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    } else if (get_length_rows() == V->size() && index >= 0 && index <= get_length_rows()) {
        /* cilk_ */ for (int i = 0; i < V->size(); i++)
            values[i]->add_at(V->get_value(i), index);
    } else
        throw std::invalid_argument(
                "It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_col_copy_at(const vektor<T> &V, const int index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V);
    } else if (get_length_rows() == V.size() && index >= 0 && index <= get_length_rows()) {
        /* cilk_ */ for (auto i = 0; i < V.size(); i++) values[i]->add_at(V.get_value(i), index);
    } else
        throw std::invalid_argument(
                "It's impossible to add a row of different length or in a bad index.");
}


template<class T>
void vmatrix<T>::add_col_copy_at(T *V, const ssize_t N, const ssize_t index)
{
//    std::unique_lock<std::mutex> _(vmatrix_mutex);

    if ((get_length_rows() == 0 || get_length_cols() == 0) && index == 0) {
        add_col_copy(V, N);
    } else if (get_length_rows() == N && index >= 0 && index <= get_length_rows()) {
        for (int i = 0; i < N; i++) values[i]->add_at(V[i], index);
    } else
        throw std::invalid_argument(
                "It's impossible to add a row of different length or in a bad index.");
}

template<class T>
void vmatrix<T>::remove_row(const ssize_t index)
{
    if (index >= 0 && index < get_length_rows())
        values.erase(std::next(values.begin(), index), std::next(values.begin(), index + 1));
    else
        throw std::invalid_argument(
                "It's impossible to remove an element from the matrix that doesn't exist.");
}

template<class T>
void vmatrix<T>::remove_rows(const ssize_t start_index, const ssize_t end_index)
{
    if (start_index < 0 || start_index >= get_length_rows() || end_index < 0 || end_index >= get_length_rows() || start_index > end_index)
        throw std::invalid_argument(
                "It's impossible to remove an element from the matrix that doesn't exist.");
    values.erase(std::next(values.begin(), start_index), std::next(values.begin(), end_index));
}

template<class T>
void vmatrix<T>::remove_col(const ssize_t index)
{
    if (index >= 0 && index < get_length_cols())
        for (int i = 0; i < get_length_rows(); ++i)
            values[i]->remove_at(index);
    else
        throw std::invalid_argument(
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

template<class T>
vmatrix <T> vmatrix<T>::extract_rows(const int from_row_index, const int to_row_index) const
{
    if (from_row_index >= 0 && to_row_index < get_length_rows() && from_row_index <= to_row_index) {
        vmatrix<T> M(1 + to_row_index - from_row_index);
        /* cilk_ */ for (auto i = from_row_index; i <= to_row_index; ++i)
            M.set_row_ref_fast(get_row_copy(i), i - from_row_index);
        return M;
        
    } else
        throw std::invalid_argument("It's impossible to extract the rows, invalid indexes.");
}

template<class T>
vmatrix <T> vmatrix<T>::extract_cols(const int from_col_index, const int to_col_index) const
{
    if (from_col_index >= 0 && to_col_index < get_length_cols() && from_col_index <= to_col_index) {
        vmatrix<T> M(get_length_rows());
        /* cilk_ */ for (ssize_t i = 0; i < get_length_rows(); ++i)
            M.set_row_ref_fast(get_row_ptr(i)->extract(from_col_index, to_col_index), i);
        return M;
    } else
        throw std::invalid_argument("It's impossible to extract the columns, invalid indexes");
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
        /* cilk_ */ for (ssize_t i = 0; i < rows; ++i) V2->set_value(i, V->product_vector_scalar(values[i].get(), V));
        return V2;
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                "It's impossible to multiply a matrix and a vector with different length: " << cols << ", " << V->size());
}


template<class T> vektor<T>
vmatrix<T>::product_vector(const vektor<T> &V) const
{
    auto const rows = get_length_rows(), cols = get_length_cols();
    if (cols == 0 || cols == V.size())
    {
        vektor<T> V2((T)0., rows);
	/* cilk_ */ for (ssize_t i = 0; i < rows; i++) V2.set_value(i, V.product_vector_scalar(*values[i], V));
        return V2;
    } else
        throw std::invalid_argument(svr::common::formatter() <<
                "It's impossible to multiply a matrix and a vector with different length " << cols << ", " << V.size());
}

template<class T>
vektor<T> *vmatrix<T>::product_vector(vmatrix const *M, vektor<T> const *V)
{
    if (M->get_length_cols() == 0 || M->get_length_cols() == V->size()) {
        auto V2 = new vektor<T>((T)0., M->get_length_rows());
        /* cilk_ */ for (ssize_t i = 0; i < M->get_length_rows(); i++)
            V2->set_at(i, V->product_vector_scalar(M->values[i].get(), V));
        return V2;
    } else
        throw std::invalid_argument("It's impossible to product a matrix and a vector with different length.");
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
    throw std::invalid_argument("It's impossible to product two vectors with different length.");
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
    throw std::invalid_argument("It's impossible to multiply two vectors with different length.");
}


template<class T>
vmatrix<T> *vmatrix<T>::product_matrix_matrix(vmatrix<T> const *M1, vmatrix<T> const *M2)
{
    if (M1->get_length_cols() == M2->get_length_cols()) {
        vmatrix<T> *M3 = new vmatrix<T>();
        for (int i = 0; i < M1->get_length_rows(); i++) {
            vektor<T> *V = new vektor<T>(M2->get_length_rows());
            for (int j = 0; j < M2->get_length_rows(); j++)
                V->set_at(V->product_vector_scalar(M1->values[i].get(), M2->values[j].get()));
            M3->add_row_ref(V);

        }
        return M3;
    } else
        throw std::invalid_argument(
                "It's impossible to product matrices with different dimensions.");
}

// I/O Operations
template<class T>
vmatrix<double> *vmatrix<T>::load(char const *filename)
{
    // Open the file
    std::ifstream file(filename, std::ios::in);
    if (!file) {
        LOG4_ERROR("Error. It's impossible to open the file.");
        return new vmatrix<double>();
    }
    vmatrix<double> *M = new vmatrix<double>();
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
            M->add_row_ref(V);
        }
    } catch (...) {
        LOG4_ERROR("Error. It's impossible to complete the load.");
    }
    // Close the file
    file.close();

    return M;
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

}
}
