//
// Created by zarko on 2/12/24.
//

#ifndef SVR_SERIALIZATION_HPP
#define SVR_SERIALIZATION_HPP

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/throw_exception.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/collection_traits.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <tuple>
#include <armadillo>
#include <oneapi/tbb/concurrent_map.h>
#include <oneapi/tbb/concurrent_set.h>


namespace boost {
namespace serialization {

template<typename Archive, typename T> void save(Archive & ar, const arma::Mat<T> &a, const unsigned version)
{
    std::ostringstream oss;
    a.save(oss, arma::file_type::arma_binary);
    std::cout << "Saving matrix " << oss.str().size() << std::endl;
    ar & oss.str();
}

template<typename Archive, typename T> void load(Archive & ar, arma::Mat<T> &a, const unsigned version)
{
    std::string os;
    ar & BOOST_SERIALIZATION_NVP(os);
    std::cout << "Loading matrix " << os.size() << std::endl;
    std::stringstream oss(os);
    a.load(oss, arma::file_type::arma_binary);
}

template <typename Ar, typename T> inline void serialize(Ar& ar, arma::Mat<T>& t, const unsigned version)
{
    split_free(ar, t, version);
}

template <typename Ar, typename T> inline void serialize(Ar& ar, arma::Col<T>& t, const unsigned version)
{
    split_free(ar, t, version);
}

template<typename Archive> void save(Archive & ar, const boost::posix_time::ptime &a, const unsigned version)
{
    std::ostringstream oss;
    oss << boost::posix_time::to_iso_extended_string(a);
    ar & oss.str();
}

template<typename Archive> void load(Archive & ar, boost::posix_time::ptime &a, const unsigned version)
{
    std::string os;
    ar & BOOST_SERIALIZATION_NVP(os);
    std::stringstream oss(os);
    a = boost::posix_time::from_iso_extended_string(oss.str());
}

template <typename Ar> inline void serialize(Ar& ar, boost::posix_time::ptime &t, const unsigned version)
{
    split_free(ar, t, version);
}

template <typename Ar, typename... Ts>
inline void save(Ar& ar, tbb::concurrent_vector<Ts...> const& t, unsigned)
{
    boost::serialization::stl::save_collection<Ar, tbb::concurrent_vector<Ts...>>(ar, t);
}

template <typename Ar, typename... Ts>
inline void load(Ar& ar, tbb::concurrent_vector<Ts...>& t, unsigned)
{
    item_version_type    item_version(0);
    collection_size_type count;
    ar >> BOOST_SERIALIZATION_NVP(count);
    if (library_version_type{3} < ar.get_library_version())
        ar >> BOOST_SERIALIZATION_NVP(item_version);
    stl::collection_load_impl(ar, t, count, item_version);
}

template <typename Ar, typename... Ts>
inline void serialize(Ar& ar, tbb::concurrent_vector<Ts...>& t, unsigned version)
{
    split_free(ar, t, version);
}

template<typename A, typename... Ts> void
save(A &ar, const tbb::concurrent_set<Ts...> &t, const unsigned ver)
{
    std::deque<std::decay_t<typename tbb::concurrent_set<Ts...>::value_type>> tmp;
    for (const auto &e: t) tmp.emplace_back(e);
    ar & tmp;
}

template<typename A, typename... Ts>
void load(A &ar, tbb::concurrent_set<Ts...> &t, const unsigned ver)
{
    std::deque<std::decay_t<typename tbb::concurrent_set<Ts...>::value_type>> tmp;
    ar & tmp;
    for (auto &e: tmp) t.emplace(e);
}

template <typename Ar, typename... Ts>
inline void serialize(Ar& ar, tbb::concurrent_set<Ts...>& t, unsigned version)
{
    split_free(ar, t, version);
}

template<typename A, typename... Ts> void
save(A &ar, const tbb::concurrent_map<Ts...> &t, const unsigned ver)
{
    std::deque<std::pair<
            typename std::decay_t<typename tbb::concurrent_map<Ts...>::key_type>,
            typename tbb::concurrent_map<Ts...>::mapped_type>> tmp;
    for (const auto &e: t) tmp.emplace_back(e);
    ar & tmp;
}

template<typename A, typename... Ts>
void load(A &ar, tbb::concurrent_map<Ts...> &t, const unsigned ver)
{
    std::deque<std::pair<
            std::decay_t<typename tbb::concurrent_map<Ts...>::key_type>,
            typename tbb::concurrent_map<Ts...>::mapped_type>> tmp;
    ar & tmp;
    for (auto &e: tmp) t.emplace(e);
}

template <typename Ar, typename... Ts>
inline void serialize(Ar& ar, tbb::concurrent_map<Ts...>& t, unsigned version)
{
    split_free(ar, t, version);
}

template <typename Ar, typename... Ts>
inline void serialize(Ar& ar, std::pair<Ts...> &t, unsigned version)
{
    ar & t.first;
    ar & t.second;
}

template <typename Ar, typename... Ts>
inline void serialize(Ar& ar, std::tuple<Ts...> &t, unsigned version)
{
    for (size_t i = 0; i < std::tuple_size<std::tuple<Ts...>>{}; ++i) ar & std::get<i>(t);
}

} // namespace serialization
} // namespace boost

BOOST_SERIALIZATION_SPLIT_FREE(arma::mat);
BOOST_SERIALIZATION_SPLIT_FREE(arma::uvec);

#endif //SVR_SERIALIZATION_HPP
