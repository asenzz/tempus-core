#pragma once

#include <boost/date_time/posix_time/ptime.hpp>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>
#include <map>
#include <set>
#include <armadillo>
#include <viennacl/matrix.hpp>
#include <execution>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_map.h>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_unordered_set.h>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_unordered_map.h>
#include <boost/functional/hash.hpp>
#include <functional>
#include <thread>

#include "types.hpp"
#include "defines.h"
#include "util/string_utils.hpp"

namespace bpt = boost::posix_time;

#define dtype(T) std::decay_t<decltype(T)>
#define release_cont(x) dtype((x)){}.swap((x));
#define PRAGMASTR(_STR) _Pragma(TOSTR(_STR))

template <typename T>
struct return_type : return_type<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct return_type<ReturnType(ClassType::*)(Args...) const>
{
    using type = ReturnType;
};

namespace boost {

#include "hashing.tpp"

}

namespace std {

#include "hashing.tpp"

//template<typename T, typename ...Args>
//std::unique_ptr<T> make_unique( Args&& ...args )
//{
//    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
//}
} // namespace std


namespace svr {

template <typename... T>
constexpr auto make_array(T&&... values) ->
        std::array<
            typename std::decay<
                typename std::common_type<T...>::type>::type,
            sizeof...(T)> {
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
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::shared_ptr<typename T::element_type>>::value>::type>
{
    enum
    {
        value = true
    };
};

template<typename T>
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::unique_ptr<typename T::element_type>>::value>::type>
{
    enum
    {
        value = true
    };
};

template<typename T>
struct is_smart_pointer<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::weak_ptr<typename T::element_type>>::value>::type>
{
    enum
    {
        value = true
    };
};


typedef std::shared_ptr<std::deque<arma::mat>> matrices_ptr;
typedef std::shared_ptr<arma::mat> mat_ptr;
typedef std::shared_ptr<arma::vec> vec_ptr;
typedef std::shared_ptr<std::deque<bpt::ptime>> times_ptr;

using mutex_ptr = std::shared_ptr<std::mutex>;

std::string demangle(const std::string &name);

template<typename T>
bool operator!=(const std::set<std::shared_ptr<T>> &lhs, const std::set<std::shared_ptr<T>> &rhs)
{
    if (lhs.size() != rhs.size()) return true;
    for (auto lhs_iter = lhs.begin(), rhs_iter = rhs.begin(); lhs_iter != lhs.end(), rhs_iter != rhs.end(); ++lhs_iter, ++rhs_iter) {
        if (*lhs_iter != *rhs_iter) return true;
        ++lhs_iter;
        ++rhs_iter;
    }
    return false;
}

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
inline auto highLevelF(Args&&... args) -> dtype(lowLevelF(std::forward<Args>(args)...)) \
{ \
    return lowLevelF(std::forward<Args>(args)...); \
}

template<typename T> unsigned adj_threads(const T iterations)
{
    return (unsigned) std::min<T>(iterations < 0 ? 0 : iterations, std::thread::hardware_concurrency());
}

template<typename T> const T &operator^(const std::set<T> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T, typename C> const T &operator^(const std::set<T, C> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T> const T &operator^(const tbb::concurrent_set <T> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename T, typename C> const T &operator^(const tbb::concurrent_set <T, C> &s, const size_t i)
{
    return *std::next(s.cbegin(), i);
}

template<typename K, typename V> const K &operator%(const std::map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> const V &operator^(const std::map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> V &operator^(std::map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> const K &operator%(const std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> const V &operator^(const std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> V &operator^(std::unordered_map<K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> const K &operator%(const tbb::concurrent_map <K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->first;
}

template<typename K, typename V> const V &operator^(const tbb::concurrent_map <K, V> &s, const size_t i)
{
    return std::next(s.cbegin(), i)->second;
}

template<typename K, typename V> V &operator^(tbb::concurrent_map <K, V> &s, const size_t i)
{
    return std::next(s.begin(), i)->second;
}

template<typename K, typename V> const K &last(const tbb::concurrent_map <K, V> &s)
{
    if (s.size() < 2) return *s.cbegin()->first;
    return std::prev(s.end())->first;
}

template<typename K, typename V> const V &back(const tbb::concurrent_map <K, V> &s)
{
    if (s.size() < 2) return *s.cbegin()->second;
    return std::prev(s.cend())->second;
}

template<typename K, typename V> const K &first(const tbb::concurrent_map <K, V> &s)
{
    return s.cbegin()->first;
}

template<typename K, typename V> const V &front(const tbb::concurrent_map <K, V> &s)
{
    return s.cbegin()->second;
}

template<typename V> const V &front(const tbb::concurrent_set <V> &s)
{
    return *s.cbegin();
}

template<typename V> const V &back(const tbb::concurrent_set <V> &s)
{
    if (s.size() < 2) return *s.cbegin();
    return *std::prev(s.cend());
}

template<typename V, typename L> const V &front(const std::set<V, L> &s)
{
    return *s.cbegin();
}

template<typename V, typename L> const V &back(const std::set<V, L> &s)
{
    if (s.size() < 2) return *s.cbegin();
    return *std::prev(s.cend());
}


template<typename T> auto
operator==(const std::deque<std::shared_ptr<T>> &lhs, const std::deque<std::shared_ptr<T>> &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (bool(lhs.at(i)) == bool(rhs.at(i))
            && *lhs.at(i) == *rhs.at(i))
            continue;
        return false;
    }
    return true;
}

template<typename T> auto
operator==(const std::deque<T> &lhs, const std::deque<T> &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs.at(i) == rhs.at(i)) continue;
        return false;
    }
    return true;
}


namespace common {

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
arma::uvec find(const arma::Mat <T> &m1, const arma::Mat <T> &m2)
{
    arma::uvec r;
#pragma omp parallel for ordered schedule(static, 1 + m2.size() / std::thread::hardware_concurrency()) num_threads(adj_threads(m2.size()))
    for (const auto e: m2) {
        const arma::uvec r_e = arma::find(m1 == e);
#pragma omp ordered
        r.insert_rows(r.n_rows, r_e);
    }
    return r;
}

template<typename T>
arma::uvec lower_bound(const arma::Mat <T> &m1, const arma::Mat <T> &m2)
{
    arma::uvec r;
#pragma omp parallel for ordered schedule(static, 1 + m2.size() / std::thread::hardware_concurrency()) num_threads(adj_threads(m2.size()))
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
std::vector<std::shared_ptr<T>> inline
clone_shared_ptr_elements(const std::vector<std::shared_ptr<T>> &arg)
{
    std::vector<std::shared_ptr<T>> res;
    for (const std::shared_ptr<T> &p_elem: arg) res.emplace_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename T>
std::deque<std::shared_ptr<T>> inline
clone_shared_ptr_elements(const std::deque<std::shared_ptr<T>> &arg)
{
    std::deque<std::shared_ptr<T>> res;
    for (const std::shared_ptr<T> &p_elem: arg) res.emplace_back(std::make_shared<T>(*p_elem));
    return res;
}

template<typename T, typename L>
tbb::concurrent_set <std::shared_ptr<T>, L> inline
clone_shared_ptr_elements(const tbb::concurrent_set <std::shared_ptr<T>, L> &arg)
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
tovec(const arma::Mat <T> &input, std::vector<T> &output)
{
    output.resize(input.n_elem);
    output.shrink_to_fit();
    memcpy(output.data(), input.mem, input.n_elem * sizeof(T));
}

template<typename T> std::vector<T>
tovec(const arma::Mat <T> &input)
{
    std::vector<T> output(input.n_elem);
    tovec(input, output);
    return output;
}

viennacl::vector<double> tovcl(const arma::colvec &in);

template<typename T> viennacl::matrix <T>
tovcl(const arma::Mat <T> &in, const viennacl::ocl::context &cx)
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


template<typename T> arma::Col <T>
toarmacol(const tbb::concurrent_vector <T> &v)
{
    arma::Col<T> r;
    r.set_size(v.size());
    std::copy(std::execution::par_unseq, v.cbegin(), v.cend(), r.begin());
    return r;
}

template<typename T> arma::Col <T>
toarmacol(const tbb::concurrent_unordered_set <T> &v)
{
    arma::Col<T> r;
    r.set_size(v.size());
    std::copy(std::execution::par_unseq, v.cbegin(), v.cend(), r.begin());
    return r;
}


template<typename T> arma::Mat <T>
toarma(const viennacl::matrix <T> &in)
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

template<class T>
class wrap_vector : public std::vector<T>
{
public:
    wrap_vector()
    {
        this->_M_impl._M_start = this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = NULL;
    }

    wrap_vector(const wrap_vector &o)
    {
        if (this == &o) return;
        this->_M_impl._M_start = o._M_impl._M_start;
        this->_M_impl._M_finish = o._M_impl._M_finish;
        this->_M_impl._M_end_of_storage = o._M_impl._M_end_of_storage;
    }

    wrap_vector(wrap_vector &&o) noexcept
    {
        if (this == &o) return;
        this->_M_impl._M_start = o._M_impl._M_start;
        this->_M_impl._M_finish = o._M_impl._M_finish;
        this->_M_impl._M_end_of_storage = o._M_impl._M_end_of_storage;
        o._M_impl._M_start = o._M_impl._M_finish = o._M_impl._M_end_of_storage = NULL;
    }

    wrap_vector &operator=(const wrap_vector &o)
    {
        if (this == &o) return *this;
        this->_M_impl._M_start = o._M_impl._M_start;
        this->_M_impl._M_finish = o._M_impl._M_finish;
        this->_M_impl._M_end_of_storage = o._M_impl._M_end_of_storage;
        return *this;
    }

    wrap_vector &operator=(wrap_vector &&o) noexcept
    {
        if (this == &o) return *this;
        this->_M_impl._M_start = o._M_impl._M_start;
        this->_M_impl._M_finish = o._M_impl._M_finish;
        this->_M_impl._M_end_of_storage = o._M_impl._M_end_of_storage;
        o._M_impl._M_start = o._M_impl._M_finish = o._M_impl._M_end_of_storage = NULL;
        return *this;
    }

    wrap_vector(T *sourceArray, const uint size)
    {
        this->_M_impl._M_start = sourceArray;
        this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = sourceArray + size;
    }

    ~wrap_vector()
    {
        this->_M_impl._M_start = this->_M_impl._M_finish = this->_M_impl._M_end_of_storage = NULL;
    }
};

template<typename T>
struct copyatomic
{
    std::atomic<T> _a;

    copyatomic()
            : _a()
    {}

    copyatomic(const std::atomic<T> &a)
            : _a(a.load(std::memory_order_relaxed))
    {}

    copyatomic(const copyatomic &other)
            : _a(other._a.load(std::memory_order_relaxed))
    {}

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
#ifndef __GNUC__
#pragma unroll
#endif
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

} // namespace common
} // namespace svr

