//
// Created by Zarko on 2/26/24.
//

template<>
struct hash<boost::posix_time::time_duration>
{
    std::size_t operator()(const boost::posix_time::time_duration &k) const
    {
        return boost::hash_value(static_cast<std::size_t>(k.ticks()));
    }
};

template<>
struct hash<boost::posix_time::ptime>
{
    std::size_t operator()(const boost::posix_time::ptime &k) const
    {
        return boost::hash_value(static_cast<std::size_t>(to_time_t(k)));
    }
};

template<>
struct hash<arma::SizeMat>
{
    std::size_t operator()(const arma::SizeMat &k) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, k.n_rows);
        boost::hash_combine(seed, k.n_cols);
        return seed;
    }
};

template<>
struct hash<std::set<size_t>>
{
    std::size_t operator()(const std::set<size_t> &k) const
    {
        std::size_t seed = boost::hash_range(k.begin(), k.end());
        return seed;
    }
};

template<typename ...Elements>
struct hash<std::tuple<Elements...>>
{
    std::size_t operator()(const std::tuple<Elements...> &k) const
    {
        std::size_t seed = 0;
        apply([&seed](const auto &... el) { (boost::hash_combine(seed, el), ...); }, k);
        return seed;
    }
};

template<typename ...Elements>
struct hash<std::pair<Elements...>>
{
    size_t operator()(const std::pair<Elements...> &k) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, k.first);
        boost::hash_combine(seed, k.second);
        return seed;
    }
};

