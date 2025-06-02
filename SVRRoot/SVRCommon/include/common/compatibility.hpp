#pragma once

#include <memory>
#include <sstream>
#include <type_traits>
#include <tuple>
#include <vector>
#include <map>
#include <set>
#include <execution>
#include <functional>
#include <armadillo>
#include <oneapi/tbb/mutex.h>
#ifdef ENABLE_OPENCL
#include <viennacl/matrix.hpp>
#endif
#include <oneapi/tbb/concurrent_map.h>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/concurrent_unordered_set.h>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <boost/functional/hash.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include "types.hpp"
#include "defines.h"
#include "constants.hpp"
#include "util/string_utils.hpp"

namespace bpt = boost::posix_time;

#define ARRAYLEN(X) (sizeof(X) / sizeof((X)[0]))
#define ARMASIZEOF(X) (X).n_elem * sizeof((X)[0])

#define DTYPE(T) std::decay_t<decltype(T)>
#define CAST2(x) (DTYPE(x))
#define PCAST(T, X) (*std::launder((T *) &(X)))
#define RELEASE_CONT(x) DTYPE((x)){}.swap((x));
#define PRAGMASTR(_STR) _Pragma(TOSTR(_STR))
#define CPTR(T) const T *const
#define RPTR(T) T *__restrict__ const
#define CRPTR(T) const T *__restrict__ const
#define CPTRd CPTR(double)
#define CRPTRd CRPTR(double)

#define ALIGN_ALLOCA(T, SZ, ALIGN) (T* const)(((uintptr_t)alloca(SZ + (ALIGN - 1)) + (ALIGN - 1)) & ~(uintptr_t)(ALIGN - 1))


// General utility macro
#define PP_CAT(A, B) A ## B
#define PP_EXPAND(...) __VA_ARGS__

// Macro overloading feature support
#define PP_VA_ARG_SIZE(...) PP_EXPAND(PP_APPLY_ARG_N((PP_ZERO_ARGS_DETECT(__VA_ARGS__), PP_RSEQ_N)))

#define PP_ZERO_ARGS_DETECT(...) PP_EXPAND(PP_ZERO_ARGS_DETECT_PREFIX_ ## __VA_ARGS__ ## _ZERO_ARGS_DETECT_SUFFIX)
#define PP_ZERO_ARGS_DETECT_PREFIX__ZERO_ARGS_DETECT_SUFFIX ,,,,,,,,,,,0

#define PP_APPLY_ARG_N(ARGS) PP_EXPAND(PP_ARG_N ARGS)
#define PP_ARG_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define PP_RSEQ_N 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define PP_OVERLOAD_SELECT(NAME, NUM) PP_CAT( NAME ## _, NUM)
#define PP_MACRO_OVERLOAD(NAME, ...) PP_OVERLOAD_SELECT(NAME, PP_VA_ARG_SIZE(__VA_ARGS__))(__VA_ARGS__)


#define PROPERTY(...) PP_MACRO_OVERLOAD(PROPERTY, __VA_ARGS__)

#define PROPERTY_PUBLISHED(T, X) \
    public:                      \
        inline T get_##X() const { return X; } \
        inline T &get_##X() { return X; } \
        inline void set_##X(const T &X##_) { X = X##_; }

#define PROPERTY_2(T, X) private: T X; PROPERTY_PUBLISHED(T, X)
#define PROPERTY_3(T, X, D) private: T X = D; PROPERTY_PUBLISHED(T, X)

template<typename Derived, typename Base>
std::unique_ptr<Derived> dynamic_ptr_cast(std::unique_ptr<Base> &&base)
{
    if (auto derived = dynamic_cast<Derived *>(base.get())) {
        base.release();
        return std::unique_ptr<Derived>(derived);
    }
    return nullptr;
}

template<typename Derived, typename Base>
std::shared_ptr<Derived> dynamic_ptr_cast(std::shared_ptr<Base> &&base)
{
    if (auto derived = dynamic_cast<Derived *>(base.get())) {
        base.release();
        return std::shared_ptr<Derived>(derived);
    }
    return nullptr;
}

template<typename T>
struct return_type : return_type<decltype(&T::operator())>
{
};

// For generic types, directly use the result of the signature of its 'operator()'

template<typename ClassType, typename ReturnType, typename... Args>
struct return_type<ReturnType(ClassType::*)(Args...) const>
{
    using type = ReturnType;
};


namespace svr {
constexpr double C_double_nan = std::numeric_limits<double>::quiet_NaN();

template<typename... T>
constexpr auto make_array(T &&... values) ->
    std::array<
        typename std::decay<
            typename std::common_type<T...>::type>::type,
        sizeof...(T)>
{
    return {std::forward<T>(values)...};
}

template<typename T, typename Enable = void>
struct is_smart_pointer
{
    enum
    {
        value = false
    };
};

template<typename T>
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::shared_ptr<typename T::element_type> >::value>::type>
{
    enum
    {
        value = true
    };
};

template<typename T>
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::unique_ptr<typename T::element_type> >::value>::type>
{
    enum
    {
        value = true
    };
};

template<typename T>
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::weak_ptr<typename T::element_type> >::value>::type>
{
    enum
    {
        value = true
    };
};


typedef std::shared_ptr<std::deque<arma::mat> > matrices_ptr;
typedef std::shared_ptr<arma::cube> cube_ptr;
typedef std::shared_ptr<arma::mat> mat_ptr;
typedef std::shared_ptr<arma::vec> vec_ptr;
typedef arma::Col<std::time_t> tvec;

std::string demangle(const std::string &name);

template<typename T>
bool operator!=(const std::set<std::shared_ptr<T> > &lhs, const std::set<std::shared_ptr<T> > &rhs)
{
    if (lhs.size() != rhs.size()) return true;
    for (auto lhs_iter = lhs.begin(), rhs_iter = rhs.begin(); lhs_iter != lhs.end(), rhs_iter != rhs.end(); ++lhs_iter, ++rhs_iter) {
        if (*lhs_iter != *rhs_iter) return true;
        ++lhs_iter;
        ++rhs_iter;
    }
    return false;
}


bool operator!=(const std::deque<std::string> &lhs, const std::deque<std::string> &rhs);

bool operator<(const arma::SizeMat &lhs, const arma::SizeMat &rhs);

bool operator<(const std::set<size_t> &lhs, const std::set<size_t> &rhs);

using type_info_ref = std::reference_wrapper<const std::type_info>;

struct type_hasher
{
    std::size_t operator()(type_info_ref code) const
    {
        return code.get().hash_code();
    }
};

struct equal_to
{
    bool operator()(type_info_ref lhs, type_info_ref rhs) const
    {
        return lhs.get() == rhs.get();
    }
};

#define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF) \
template<typename... Args> \
inline auto highLevelF(Args&&... args) -> DTYPE(lowLevelF(std::forward<Args>(args)...)) \
{ \
return lowLevelF(std::forward<Args>(args)...); \
}

template<typename I, typename T = typename I::value_type> I
before_bound(const I &begin, const I &end, const T val)
{
    auto rI = std::lower_bound(begin, end, val);
    while (*rI > val) --rI;
    return rI;
}

template<typename T> inline const T &operator^(const std::set<T> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T, typename C> inline const T &operator^(const std::set<T, C> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T> inline const T &operator^(const tbb::concurrent_set<T> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T, typename C> inline const T &operator^(const tbb::concurrent_set<T, C> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename K, typename V> inline const K &operator%(const std::map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> inline const V &operator^(const std::map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V, typename L> inline const V &operator^(const std::map<K, V, L> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> inline V &operator^(std::map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> inline const K &operator%(const std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> inline const V &operator^(const std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> inline V &operator^(std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> inline const K &operator%(const boost::unordered_flat_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> inline const V &operator^(const boost::unordered_flat_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> inline V &operator^(boost::unordered_flat_map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> inline const K &operator%(const tbb::concurrent_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> inline const V &operator^(const tbb::concurrent_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> inline V &operator^(tbb::concurrent_map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> inline const K &last(const tbb::concurrent_map<K, V> &s)
{
    if (s.size() < 2) return *s.cbegin()->first;
    return std::prev(s.end())->first;
}

template<typename K, typename V> inline const V &back(const tbb::concurrent_map<K, V> &s)
{
    if (s.size() < 2) return *s.cbegin()->second;
    return std::prev(s.cend())->second;
}

template<typename K, typename V> inline const K &first(const tbb::concurrent_map<K, V> &s)
{
    return s.cbegin()->first;
}

template<typename K, typename V> inline const V &front(const tbb::concurrent_map<K, V> &s)
{
    return s.cbegin()->second;
}

template<typename V> inline const V &front(const tbb::concurrent_set<V> &s)
{
    return *s.cbegin();
}

template<typename V> inline const V &back(const tbb::concurrent_set<V> &s)
{
    if (s.size() < 2) return *s.cbegin();
    return *std::prev(s.cend());
}

template<typename V, typename L> inline const V &front(const std::set<V, L> &s)
{
    return *s.cbegin();
}

template<typename V, typename L> inline const V &back(const std::set<V, L> &s)
{
    if (s.size() < 2) return *s.cbegin();
    return *std::prev(s.cend());
}

template<typename T> inline bool operator==(const arma::Mat<T> &lhs, const arma::Mat<T> &rhs)
{
    if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols) return false;
    for (size_t i = 0; i < lhs.n_rows; ++i)
        for (size_t j = 0; j < lhs.n_cols; ++j)
            if (lhs(i, j) != rhs(i, j)) return false;
    return true;
}

template<typename T> inline bool operator==(const std::deque<std::shared_ptr<T> > &lhs, const std::deque<std::shared_ptr<T> > &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (bool(lhs.at(i)) == bool(rhs.at(i)) && *lhs.at(i) == *rhs.at(i)) continue;
        return false;
    }
    return true;
}

template<typename T> inline bool operator==(const std::deque<arma::Col<T> > &lhs, const std::deque<arma::Col<T> > &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (arma::all(lhs.at(i) == rhs.at(i))) continue;
        return false;
    }
    return true;
}

template<typename T> inline bool operator==(const std::deque<T> &lhs, const std::deque<T> &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs.at(i) == rhs.at(i)) continue;
        return false;
    }
    return true;
}

namespace common {
template<typename C> inline size_t capacity(const C &um)
{
    return um.bucket_count() * um.max_load_factor();
}

void init_petsc();

void uninit_petsc();

void print_stacktrace();

template<typename C, typename T = typename C::value_type> auto
empty(const C &container)
{
    bool res = true;
    for (T v: container) res &= v.empty();
    return res;
}

template<typename C, typename T = typename C::value_type> auto
max_size(const C &container)
{
    size_t max = 0;
    for (const auto &item: container)
        if (item.size() > max)
            max = item.size();
    return max;
}


template<typename T>
arma::uvec lower_bound(const arma::Mat<T> &m1, const arma::Mat<T> &m2)
{
    arma::uvec r;
#pragma omp parallel for ordered schedule(static, 1 + m2.size() / C_n_cpu) num_threads(adj_threads(m2.size()))
    for (const auto e: m2) {
        const arma::uvec r_e = arma::find(m1 >= e);
#pragma omp ordered
        r.insert_rows(r.n_rows, r_e);
    }
    return r;
}

typedef struct drand48_data t_drand48_data;
typedef t_drand48_data *t_drand48_data_ptr;

t_drand48_data seedme(const size_t thread_id);

double drander(t_drand48_data_ptr buffer);

bool file_exists(const std::string &filename);

#define ELEMCOUNT(array) (sizeof(array)/sizeof(array[0]))

/* Return codes */
typedef size_t svrerr_t;
#define SVR_RC_GENERAL_ERROR (std::numeric_limits<size_t>::max())

const long multi_div = 100000; // number of zeros tell the number of decimal points to assume when comparing doubles

// this is needed because there are some issues when comparing double values
// @see http://stackoverflow.com/questions/17333/most-effective-way-for-float-and-double-comparison
bool Equals(const double &lhs, const double &rhs);

double Round(const double &dbl);

template<typename T>
std::vector<std::shared_ptr<T> > inline
clone_shared_ptr_elements(const std::vector<std::shared_ptr<T> > &arg)
{
    std::vector<std::shared_ptr<T> > res;
    for (const std::shared_ptr<T> &p_elem: arg) res.emplace_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename T>
std::deque<std::shared_ptr<T> > inline
clone_shared_ptr_elements(const std::deque<std::shared_ptr<T> > &arg)
{
    std::deque<std::shared_ptr<T> > res;
    for (const std::shared_ptr<T> &p_elem: arg) res.emplace_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename T, typename L>
tbb::concurrent_set<std::shared_ptr<T>, L> inline
clone_shared_ptr_elements(const tbb::concurrent_set<std::shared_ptr<T>, L> &arg)
{
    tbb::concurrent_set<std::shared_ptr<T>, L> res;
#pragma omp parallel num_threads(adj_threads(arg.size()))
#pragma omp single nowait
    for (const std::shared_ptr<T> &p_elem: arg)
#pragma omp task
        res.emplace(std::make_shared<T>(*p_elem));
    return res;
}

template<typename K, typename T>
std::map<K, std::shared_ptr<T> > inline
clone_shared_ptr_elements(const std::map<K, std::shared_ptr<T> > &arg)
{
    std::map<K, std::shared_ptr<T> > res;
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
to_times(const std::map<bpt::ptime, std::shared_ptr<T> > &data_rows)
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

#ifdef ENABLE_OPENCL
viennacl::vector<double> tovcl(const arma::colvec &in);

template<typename T> viennacl::matrix<T>
tovcl(const arma::Mat<T> &in, const viennacl::ocl::context &cx)
{
    cl_int rc;
    auto clbuf = (T *) clCreateBuffer(cx.handle().get(), CL_MEM_READ_WRITE, in.n_elem * sizeof(T), nullptr, &rc);
    if (rc != CL_SUCCESS)
        throw std::runtime_error("Failed creating OpenCL buffer of size " + std::to_string(in.n_elem * sizeof(T)) + " with error " + std::to_string(rc));
    viennacl::matrix<T> r(clbuf, viennacl::OPENCL_MEMORY, in.n_cols, in.n_rows);
    viennacl::backend::memory_write(r.handle(), 0, in.n_elem * sizeof(T), in.memptr(), false);
    return viennacl::trans(r);
}


template<typename T> viennacl::matrix<T>
tovcl(const arma::Mat<T> &in)
{
    auto hostbuf = (T *) malloc(in.n_elem * sizeof(T));
    viennacl::matrix<T> r(hostbuf, viennacl::MAIN_MEMORY, in.n_cols, in.n_rows);
    memcpy(r.handle().ram_handle().get(), in.mem, in.n_elem * sizeof(T));
    return viennacl::trans(r);
}
#endif

template<typename T> arma::Col<T>
toarmacol(const tbb::concurrent_vector<T> &v)
{
    arma::Col<T> r;
    r.set_size(v.size());
    std::copy(std::execution::par_unseq, v.cbegin(), v.cend(), r.begin());
    return r;
}

template<typename T> arma::Col<T>
toarmacol(const tbb::concurrent_unordered_set<T> &v)
{
    arma::Col<T> r;
    r.set_size(v.size());
    std::copy(std::execution::par_unseq, v.cbegin(), v.cend(), r.begin());
    return r;
}

#if ENABLE_OPENCL
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
    // TODO Fix bug, copy using stride1 and stride2 in consideration depending on row major or col major
    if (in.row_major()) r = r.t();
    if (in.size1() != in.internal_size1()) r.shed_rows(in.size1(), in.internal_size1() - 1);
    if (in.size2() != in.internal_size2()) r.shed_cols(in.size2(), in.internal_size2() - 1);
    return r;
};
#endif

#ifdef ENABLE_EIGEN

template<typename T> arma::Mat<T> toarma(const Eigen::MatrixX<T> &eigen_A)
{
    return arma::mat((T *)eigen_A.data(), eigen_A.rows(), eigen_A.cols(), false, true);
}

template<typename T> Eigen::MatrixX<T> toeigen(const arma::Mat<T> &arma_A)
{
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>((T *)arma_A.memptr(), arma_A.n_rows, arma_A.n_cols);
}

#endif

template<typename Container, typename ConstIterator>
typename Container::iterator remove_constness(Container &c, ConstIterator it)
{
    return c.erase(it, it);
}


template<typename T>
struct copyatomic
{
    std::atomic<T> _a;

    copyatomic()
        : _a()
    {
    }

    copyatomic(const std::atomic<T> &a)
        : _a(a.load(std::memory_order_relaxed))
    {
    }

    copyatomic(const copyatomic &other)
        : _a(other._a.load(std::memory_order_relaxed))
    {
    }

    copyatomic &operator=(const copyatomic &other)
    {
        _a.store(other._a.load(), std::memory_order_relaxed);
    }

    copyatomic &operator=(const T &other) const
    {
        _a.store(other, std::memory_order_relaxed);
    }

    bool operator==(const T &other) const
    {
        return _a.load(std::memory_order_relaxed) == other;
    }
};

size_t hash_lambda(const double param_val);

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

// Use for map and set containers
template<typename ContainerT, typename PredicateT>
void remove_if(ContainerT &items, const PredicateT &predicate)
{
    typename std::decay_t<ContainerT>::iterator it = items.begin();
    while (items.size() && it != items.end())
        if (predicate(*it)) {
            if (items.size() < 2) {
                items.clear();
                return;
            } else
                it = items.erase(it);
        } else
            ++it;
}


template<typename C> void delete_nth(C &c, const size_t N)
{
    if (N == 0 || c.empty()) return;

    for (size_t i = N - 1; i < c.size(); i += N - 1) {
        c.erase(c.begin() + i);
        if (c.size() < N) break;
    }
}


template<typename T, typename = void> struct is_iterable : std::false_type
{
};

template<typename T> struct is_iterable<T, std::void_t<
            decltype(std::begin(std::declval<T &>())),
            decltype(std::end(std::declval<T &>()))
        > > : std::true_type
{
};

template<typename T> constexpr bool is_iterable_v = is_iterable<T>::value;


template<typename T> class threadsafe_uniform_int_distribution : public std::uniform_int_distribution<T>
{
    tbb::mutex mx;

public:
    threadsafe_uniform_int_distribution(const T &a, const T &b): std::uniform_int_distribution<T>(a, b)
    {
    }

    template<typename _UniformRandomBitGenerator> T operator()(_UniformRandomBitGenerator &__urng)
    {
        const tbb::mutex::scoped_lock lock(mx);
        return std::uniform_int_distribution<T>::operator()(__urng);
    }
};

template<typename T>
class threadsafe_uniform_real_distribution : private std::uniform_real_distribution<T>
{
    tbb::mutex mx;

public:
    threadsafe_uniform_real_distribution(const T &a, const T &b) : std::uniform_real_distribution<T>(a, b)
    {
    }

    template<typename _UniformRandomBitGenerator> T operator()(_UniformRandomBitGenerator &__urng)
    {
        const tbb::mutex::scoped_lock lock(mx);
        return std::uniform_real_distribution<T>::operator()(__urng);
    }
};
}
}

#include "highwayhash/highwayhash_target.h"
#include "highwayhash/instruction_sets.h"

namespace svr {
namespace common {
template<typename T> size_t highway_hash_pod(const T v)
{
    constexpr highwayhash::HHKey highway_key = {0xbeefef0123456789ULL, 0x1122334455667788ULL, 0x99aabbccddeeff00ULL, 0x1234567890abcdefULL};
    highwayhash::HHResult64 hash;
    highwayhash::InstructionSets::Run<highwayhash::HighwayHash>(highway_key, (char *) &v, sizeof(v), &hash);
    return hash;
}

struct hash_pair
{
    template<typename... Elements> size_t operator()(const std::pair<Elements...> &k) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, k.first);
        boost::hash_combine(seed, k.second);
        return seed;
    }
};

template<typename Tuple, std::size_t Index = 0> class tuple_hash_helper
{
    template<typename E> static void apply(std::string &seed, const E &el)
    {
        if constexpr (std::is_trivial_v<E> && std::is_standard_layout_v<E>)
            seed.append((const char *) (&el), sizeof(el));
        else if constexpr (svr::common::is_iterable_v<E>)
            for (const auto &e: el) apply(seed, e);
        else if constexpr (std::is_same_v<E, boost::posix_time::time_duration>) {
            const size_t ticks = el.ticks();
            seed.append((const char *) (&ticks), sizeof(ticks));
        } else if constexpr (std::is_same_v<E, boost::posix_time::ptime>) {
            const size_t utime = to_time_t(el) + el.time_of_day().total_milliseconds() % 1000;
            seed.append((const char *) (&utime), sizeof(utime));
        } else {
            boost::hash<DTYPE(el) > hasher;
            const auto hashed = hasher(el);
            seed.append((const char *) (&hashed), sizeof(hashed));
        }
    }

public:
    static void applyt(std::string &seed, const Tuple &tuple)
    {
        if constexpr (Index < std::tuple_size_v<Tuple>) apply(seed, std::get<Index>(tuple));
        if constexpr (Index + 1 < std::tuple_size_v<Tuple>) tuple_hash_helper<Tuple, Index + 1>::applyt(seed, tuple);
    }
};

struct tuple_equal
{
    template<typename... T>
    bool operator()(const std::tuple<T...> &a, const std::tuple<T...> &b) const
    {
        return a == b;
    }
};

struct hash_tuple
{
    static constexpr highwayhash::HHKey highway_key = {0xabcdef0123456789ULL, 0x1122334455667788ULL, 0x99aabbccddeeff00ULL, 0x1234567890abcdefULL};

    template<typename... Elements> std::size_t operator()(const std::tuple<Elements...> &k) const
    {
        std::string seed;
        tuple_hash_helper<std::tuple<Elements...> >::applyt(seed, k);
        highwayhash::HHResult64 hash;
        highwayhash::InstructionSets::Run<highwayhash::HighwayHash>(highway_key, seed.data(), seed.size(), &hash);
        return hash;
    }
};

} // namespace common
} // namespace svr

#ifdef USE_MKL_MALLOC
#include <mkl.h>
#define ALIGNED_ALLOC_(align, size) mkl_malloc((size), (align))
#define ALIGNED_FREE_(ptr) mkl_free((ptr));
#else
#define ALIGNED_ALLOC_(align, size) aligned_alloc((align), CDIVI(size, align) * (align))
#define ALIGNED_FREE_(ptr) free((ptr))
#endif

namespace boost {
#include "hashing.tpp"
}

namespace std {
#include "hashing.tpp"
} // namespace std
