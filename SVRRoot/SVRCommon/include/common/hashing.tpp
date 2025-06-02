//
// Created by Zarko on 2/26/24.
//

template<>
struct hash<boost::posix_time::time_duration>
{
    std::size_t operator()(const boost::posix_time::time_duration &k) const
    {
        return svr::common::highway_hash_pod((std::size_t) k.total_milliseconds());
    }
};

template<>
struct hash<boost::posix_time::ptime>
{
    std::size_t operator()(const boost::posix_time::ptime &k) const
    {
        return svr::common::highway_hash_pod((std::size_t) to_time_t(k));
    }
};

template<>
struct hash<arma::SizeMat>
{
    std::size_t operator()(const arma::SizeMat &k) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, svr::common::highway_hash_pod(k.n_rows));
        boost::hash_combine(seed, svr::common::highway_hash_pod(k.n_cols));
        return seed;
    }
};

template<>
struct hash<std::set<size_t> >
{
    std::size_t operator()(const std::set<size_t> &k) const
    {
        std::size_t seed = boost::hash_range(k.begin(), k.end());
        return seed;
    }
};

template<>
struct hash<std::set<uint16_t> >
{
    std::size_t operator()(const std::set<uint16_t> &k) const
    {
        std::size_t seed = boost::hash_range(k.begin(), k.end());
        return seed;
    }
};


template<typename... Elements> struct hash<std::tuple<Elements...> >
{
    std::size_t operator() (const std::tuple<Elements...> &k) const
    {
        std::size_t seed = 0;
        svr::common::tuple_hash_helper<std::tuple<Elements...> >::apply(seed, k);
        return seed;
    }
};

template<typename... Elements> struct hash<std::pair<Elements...> >
{
    std::size_t operator()(const std::pair<Elements...> &k) const
    {
        svr::common::hash_pair hp;
        return hp(k);
    }
};
